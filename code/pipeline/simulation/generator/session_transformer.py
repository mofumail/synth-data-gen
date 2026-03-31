"""
Autoregressive transformer for synthetic e-commerce session generation.
Three factored prediction heads: ActionHead, ItemHead, TemporalHead.

Architecture:
    x_t = event_emb(action_t) + item_emb(sku_t) + delta_emb(bin_t) + pos_emb(t)
    h   = CausalTransformerDecoder(x, memory=CrossSessionHistory)
    action_t+1 ~ ActionHead(h_t)
    item_t+1   ~ ItemHead(h_t, action_t+1)
    delta_t+1  ~ TemporalHead(h_t, action_t+1, item_t+1)

Training  : teacher-forced; ItemHead/TemporalHead receive gold next-action.
Inference : cascaded sampling, action -> item -> temporal.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from config import N_TEMPORAL_BINS, TEMPORAL_MAX_S, TEMPORAL_MIN_S, VOCAB_K

# Action vocabulary, matches ALL_EVENT_TYPES + EOS
ACTION_TYPES = [
    "page_visit",        # 0
    "search_query",      # 1
    "add_to_cart",       # 2
    "remove_from_cart",  # 3
    "product_buy",       # 4
    "EOS",               # 5
]
ACTION2IDX: Dict[str, int] = {a: i for i, a in enumerate(ACTION_TYPES)}
IDX2ACTION: Dict[int, str] = {i: a for a, i in ACTION2IDX.items()}
EOS_IDX = ACTION2IDX["EOS"]
ITEM_BEARING_IDX = {ACTION2IDX["add_to_cart"], ACTION2IDX["remove_from_cart"], ACTION2IDX["product_buy"]}
N_ACTIONS = len(ACTION_TYPES)

# Temporal bins: 64 log-spaced edges between TEMPORAL_MIN_S and TEMPORAL_MAX_S
BIN_EDGES = np.logspace(
    np.log10(TEMPORAL_MIN_S),
    np.log10(TEMPORAL_MAX_S),
    N_TEMPORAL_BINS + 1,
)


def seconds_to_bin(secs: float) -> int:
    secs = float(np.clip(secs, TEMPORAL_MIN_S, TEMPORAL_MAX_S))
    return min(int(np.searchsorted(BIN_EDGES[1:], secs)), N_TEMPORAL_BINS - 1)


def bin_to_seconds(bin_idx: int) -> float:
    lo = BIN_EDGES[bin_idx]
    hi = BIN_EDGES[min(bin_idx + 1, N_TEMPORAL_BINS)]
    return float(np.sqrt(lo * hi))   # geometric midpoint


# TransitionConstraintMask, this should be removed and maybe learned innately

class TransitionConstraintMask:
    """
    Pre-sampling mask: zeroes logits for illegal next actions.
    Applied inside ActionHead at inference time.

    valid_transitions : action_str ->set[action_str]
                        EOS is always allowed from any state.

    Stores a [N_ACTIONS+1, N_ACTIONS] matrix where row N_ACTIONS is the
    unconstrained row (all ones), used when last_action_idx is None / -1.
    """

    def __init__(self, valid_transitions: dict):
        # Default: all actions allowed (row = all ones)
        mat = torch.ones(N_ACTIONS + 1, N_ACTIONS)
        for src_str, tgt_set in valid_transitions.items():
            src_idx = ACTION2IDX.get(src_str)
            if src_idx is None:
                continue
            row = torch.zeros(N_ACTIONS)
            for tgt_str in tgt_set:
                tgt_idx = ACTION2IDX.get(tgt_str)
                if tgt_idx is not None:
                    row[tgt_idx] = 1.0
            row[EOS_IDX] = 1.0   # always allow EOS
            mat[src_idx] = row
        self._allowed_matrix = mat   # [N_ACTIONS+1, N_ACTIONS]

    def apply(
        self,
        logits: torch.Tensor,           # [..., N_ACTIONS]
        last_action_idx: Optional[int],
    ) -> torch.Tensor:
        """Return logits with -inf at disallowed positions. No-op if no constraint."""
        if last_action_idx is None:
            return logits
        row = self._allowed_matrix[last_action_idx].bool().to(logits.device)
        return logits.masked_fill(~row, float("-inf"))

    def apply_batch(
        self,
        logits: torch.Tensor,           # [B, N_ACTIONS]
        last_action_idxs: torch.Tensor, # [B] long --1 means unconstrained
    ) -> torch.Tensor:
        """Vectorised batch version. -1 in last_action_idxs ->no constraint."""
        idx = torch.where(
            last_action_idxs < 0,
            torch.full_like(last_action_idxs, N_ACTIONS),   # unconstrained row
            last_action_idxs,
        )
        allowed = self._allowed_matrix.to(logits.device)[idx].bool()  # [B, N_ACTIONS]
        return logits.masked_fill(~allowed, float("-inf"))


# CrossSessionHistory

class CrossSessionHistory(nn.Module):
    """
    Encodes up to window_H past events as cross-attention memory.

    Receives pre-embedded history (embedded by the parent SessionTransformer
    using its shared embedding tables) and produces a [B, 1, d_model] context
    vector via learned attention pooling.
    """

    def __init__(self, window_H: int, d_model: int, n_heads: int = 2):
        super().__init__()
        self.window_H   = window_H
        self.d_model    = d_model
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn       = nn.MultiheadAttention(d_model, num_heads=n_heads, batch_first=True)

    def encode(
        self,
        hist_emb: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        hist_emb         : [B, H, d_model]  -pre-embedded history events
        key_padding_mask : [B, H] bool, True = padded position (ignored in attn)
        Returns          : [B, 1, d_model]  -pooled context for cross-attention
        """
        B   = hist_emb.size(0)
        q   = self.pool_query.expand(B, -1, -1)
        ctx, _ = self.attn(q, hist_emb, hist_emb,
                           key_padding_mask=key_padding_mask.float() if key_padding_mask is not None else None)
        return ctx


# Prediction heads

class ActionHead(nn.Module):
    """
    Predicts next action (5 event types + EOS).
    TransitionConstraintMask is applied at inference time (last_action_idx != None).
    During training last_action_idx is None ->no masking ->standard cross-entropy.
    """

    def __init__(self, d_model: int, n_actions: int, valid_transitions: dict):
        super().__init__()
        self.fc              = nn.Linear(d_model, n_actions)
        self.constraint_mask = TransitionConstraintMask(valid_transitions)

    def forward(
        self,
        h_t: torch.Tensor,                      # [B, d_model] or [B, T, d_model]
        last_action_idx: Optional[int] = None,  # scalar; None during training
    ) -> torch.Tensor:                           # [B, n_actions] or [B, T, n_actions]
        logits = self.fc(h_t)
        return self.constraint_mask.apply(logits, last_action_idx)


class ItemHead(nn.Module):
    """
    Predicts next SKU. Conditioned on action_t via a small action embedding
    (avoids coupling the full 630K-embedding to this head).
    Top-k is enforced at inference time in SessionTransformer.infer().
    """

    def __init__(self, d_model: int, vocab_size: int, top_k: int = 100):
        super().__init__()
        self.fc          = nn.Linear(d_model, vocab_size)
        self.top_k       = top_k
        self.action_cond = nn.Embedding(N_ACTIONS, d_model)

    def forward(
        self,
        h_t: torch.Tensor,       # [B, d_model] or [B, T, d_model]
        action_t: torch.Tensor,  # [B] or [B, T] -action indices
    ) -> torch.Tensor:           # [B, vocab_size] or [B, T, vocab_size]
        cond = h_t + self.action_cond(action_t)
        return self.fc(cond)


class TemporalHead(nn.Module):
    """
    Predicts inter-event delta_t as a categorical over N_TEMPORAL_BINS log bins.

    Fully factored conditioning:
        delta_{t+1} ~ TemporalHead(h_t, action_{t+1}, item_emb_{t+1})

    item_emb_t is the pre-embedded item vector from SessionTransformer.item_emb,
    passed in rather than re-embedded here to avoid a duplicate 630k-row table.

    Training: action_t and item_emb_t use gold next tokens (teacher-forced).
    Inference: they use the sampled outputs of the upstream heads.
    """

    def __init__(self, d_model: int, n_bins: int = N_TEMPORAL_BINS):
        super().__init__()
        self.fc          = nn.Linear(d_model, n_bins)
        self.n_bins      = n_bins
        self.action_cond = nn.Embedding(N_ACTIONS, d_model)

    def forward(
        self,
        h_t: torch.Tensor,           # [B, d_model] or [B, T, d_model]
        action_t: torch.Tensor,      # [B] or [B, T]  -action indices
        item_emb_t: torch.Tensor,    # [B, d_model] or [B, T, d_model] -pre-embedded item
    ) -> torch.Tensor:               # [B, n_bins] or [B, T, n_bins]
        cond = h_t + self.action_cond(action_t) + item_emb_t
        return self.fc(cond)


# SessionTransformer

class SessionTransformer(nn.Module):
    """
    Full autoregressive session generator.

    Training
    --------
    loss = CE(action_logits, tgt_actions)
         + CE(item_logits[item-bearing positions], tgt_items[item-bearing])
         + CE(temporal_logits, tgt_deltas)

    Inference
    ---------
    Use SessionTransformer.infer(client_id, sku, start_dt, history).

    Parameters
    ----------
    vocab_size        : unique SKUs -VOCAB_K = 630,052
    n_actions         : 5 event types + EOS = 6
    d_model           : transformer hidden dimension (default 256)
    n_layers          : decoder layers (default 4)
    n_heads           : attention heads (default 8)
    n_temporal_bins   : delta_t bins (default 64)
    window_H          : cross-session history window size
    valid_transitions : dict action_str ->set[action_str] for constraint mask
    """

    def __init__(
        self,
        vocab_size: int,
        n_actions: int = N_ACTIONS,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        n_temporal_bins: int = N_TEMPORAL_BINS,
        window_H: int = 50,
        valid_transitions: dict = None,
    ):
        super().__init__()
        self.d_model    = d_model
        self.vocab_size = vocab_size

        # Shared embedding tables
        self.event_emb = nn.Embedding(n_actions, d_model)
        self.item_emb  = nn.Embedding(vocab_size + 1, d_model, padding_idx=0)
        self.delta_emb = nn.Embedding(n_temporal_bins, d_model)
        self.pos_emb   = nn.Embedding(512, d_model)

        # Causal decoder backbone (cross-attention to history memory)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Cross-session history encoder
        self.history = CrossSessionHistory(window_H, d_model, n_heads=n_heads)

        # Prediction heads
        self.action_head   = ActionHead(d_model, n_actions, valid_transitions or {})
        self.item_head     = ItemHead(d_model, vocab_size, top_k=100)
        self.temporal_head = TemporalHead(d_model, n_temporal_bins)

    # Training forward (teacher-forced, gold conditioning)

    def forward(
        self,
        events: torch.Tensor,                         # [B, T] action indices
        items: torch.Tensor,                          # [B, T] item indices (0=no item)
        deltas: torch.Tensor,                         # [B, T] temporal bin indices
        history: Optional[dict] = None,               # keys: events/items/deltas ->[B, H]
        tgt_key_padding_mask: Optional[torch.Tensor] = None,  # [B, T-1] True=pad
        item_mask: Optional[torch.Tensor] = None,     # [B, T-1] bool -item positions only
    ):
        """
        Teacher-forced training pass.

        Input tokens  : t = 0 .. T-2
        Target tokens : t = 1 .. T-1   (shifted by 1)

        Returns
        -------
        action_logits   : [B, T-1, n_actions]
        item_logits     : [M, vocab_size]   if item_mask provided (M = item_mask.sum())
                          [B, T-1, vocab_size]  otherwise (may OOM for large vocabs)
        temporal_logits : [B, T-1, n_bins]

        item_mask should be precomputed in the training loop (valid & tgt_is_item)
        to avoid allocating the full [B, T-1, vocab_size] tensor.
        """
        in_e  = events[:, :-1]   # [B, T-1] -model input
        in_i  = items[:, :-1]
        in_d  = deltas[:, :-1]
        tgt_e = events[:, 1:]    # [B, T-1] -gold next actions (for conditioning)
        tgt_i = items[:, 1:]     # [B, T-1] -gold next items

        B, T = in_e.shape
        device = in_e.device

        pos = torch.arange(T, device=device).unsqueeze(0)   # [1, T]
        x   = (
            self.event_emb(in_e)
            + self.item_emb(in_i)
            + self.delta_emb(in_d)
            + self.pos_emb(pos)
        )  # [B, T, d_model]

        memory   = self._encode_history(history, B, device)                         # [B, >=1, d_model]
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device) # [T, T]

        h = self.decoder(
            x, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )  # [B, T, d_model]

        action_logits = self.action_head(h)                      # [B, T, n_actions]

        # Pre-embed gold next items (shared table) for item + temporal heads
        tgt_item_emb = self.item_emb(tgt_i)                     # [B, T, d_model]

        temporal_logits = self.temporal_head(h, tgt_e, tgt_item_emb)  # [B, T, n_bins]

        if item_mask is not None:
            # Sparse: only compute item logits at item-bearing positions ->[M, vocab_size]
            h_item      = h[item_mask]       # [M, d_model]
            tgt_e_item  = tgt_e[item_mask]   # [M]
            item_logits = self.item_head(h_item, tgt_e_item)   # [M, vocab_size]
        else:
            item_logits = self.item_head(h, tgt_e)             # [B, T, vocab_size]

        return action_logits, item_logits, temporal_logits

    # Autoregressive inference

    @torch.no_grad()
    def infer(
        self,
        client_id: int,
        sku: int,
        start_dt,
        history: Optional[List[dict]] = None,
        max_steps: int = 50,
        temperature: float = 1.0,
    ) -> List[dict]:
        """
        Generate one session autoregressively.

        Parameters
        ----------
        client_id   : user id (from SimpleIdentitySampler)
        sku         : seed item (0-indexed, from SimpleIdentitySampler)
        start_dt    : session start time (datetime or pd.Timestamp)
        history     : past event dicts for cross-session conditioning
        max_steps   : max events before forced stop
        temperature : sampling temperature

        Returns
        -------
        List of event dicts: {client_id, event_type, sku, timestamp}
        """
        self.eval()
        device = next(self.parameters()).device

        if not isinstance(start_dt, pd.Timestamp):
            start_dt = pd.Timestamp(start_dt)

        # Encode cross-session history ->cross-attention memory
        if history:
            hist_emb = self._history_dicts_to_emb(history, device)  # [1, H, d_model]
            memory   = self.history.encode(hist_emb)                 # [1, 1, d_model]
        else:
            memory = torch.zeros(1, 1, self.d_model, device=device)

        # Load sku2idx mapping for correct embedding lookup (trained with remapped indices)
        sku2idx: dict = getattr(self, "_sku2idx", {})
        idx2sku: dict = getattr(self, "_idx2sku", {})

        # Seed token: primes the generator with the seed item identity.
        # Treated as a synthetic "exposure" at t=0; not emitted in output.
        seed_item_idx = sku2idx.get(int(sku), 0) if sku2idx else min(int(sku) + 1, self.vocab_size - 1)
        cur_e = [ACTION2IDX["page_visit"]]
        cur_i = [seed_item_idx]
        cur_d = [0]

        events_out: List[dict] = []
        last_action_idx: Optional[int] = None
        current_dt = start_dt

        for _ in range(max_steps):
            e_t = torch.tensor([cur_e], dtype=torch.long, device=device)
            i_t = torch.tensor([cur_i], dtype=torch.long, device=device)
            d_t = torch.tensor([cur_d], dtype=torch.long, device=device)

            T   = e_t.size(1)
            pos = torch.arange(T, device=device).unsqueeze(0)
            x   = (
                self.event_emb(e_t)
                + self.item_emb(i_t)
                + self.delta_emb(d_t)
                + self.pos_emb(pos)
            )

            tgt_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)
            h   = self.decoder(x, memory, tgt_mask=tgt_mask)  # [1, T, d_model]
            h_t = h[:, -1, :]                                  # [1, d_model]

            # Action
            a_logits = self.action_head(h_t, last_action_idx) / temperature
            a_probs  = F.softmax(a_logits, dim=-1)
            a_idx    = int(torch.multinomial(a_probs, 1).item())

            if a_idx == EOS_IDX:
                break

            # Item (item-bearing actions only)
            a_t = torch.tensor([a_idx], dtype=torch.long, device=device)
            if a_idx in ITEM_BEARING_IDX:
                i_logits = self.item_head(h_t, a_t) / temperature
                k        = min(self.item_head.top_k, self.vocab_size)
                topk_v, topk_ids = torch.topk(i_logits, k, dim=-1)
                i_probs  = F.softmax(topk_v, dim=-1)
                chosen   = int(torch.multinomial(i_probs, 1).item())
                i_idx    = int(topk_ids[0, chosen].item())
            else:
                i_idx = 0   # padding for non-item-bearing events

            # Temporal bin ->seconds
            i_t_s      = torch.tensor([i_idx], dtype=torch.long, device=device)
            i_emb_t    = self.item_emb(i_t_s)                           # [1, d_model]
            d_logits   = self.temporal_head(h_t, a_t, i_emb_t) / temperature
            d_probs  = F.softmax(d_logits, dim=-1)
            bin_idx  = int(torch.multinomial(d_probs, 1).item())
            delta_s  = bin_to_seconds(bin_idx)

            current_dt = current_dt + pd.Timedelta(seconds=delta_s)
            # Decode item index ->raw SKU via idx2sku (inverse of sku2idx mapping)
            if i_idx > 0:
                sku_out = idx2sku.get(i_idx, i_idx - 1) if idx2sku else i_idx - 1
            else:
                sku_out = None

            events_out.append({
                "client_id":  client_id,
                "event_type": IDX2ACTION[a_idx],
                "sku":        sku_out,
                "timestamp":  current_dt,
            })

            cur_e.append(a_idx)
            cur_i.append(i_idx)
            cur_d.append(bin_idx)
            last_action_idx = a_idx

        return events_out

    # Batched autoregressive inference, dont ask me how this works, I don't know either

    @torch.no_grad()
    def infer_batch(
        self,
        batch_inputs: List[tuple],  # [(client_id, sku, start_dt, history), ...]
        max_steps: int = 50,
        temperature: float = 1.0,
    ) -> List[List[dict]]:
        """
        Generate B sessions in parallel.

        All sessions share a batch dimension and advance one token per step.
        Done sessions (EOS emitted) are masked out and their tokens are ignored.

        Parameters are
        ----------
        batch_inputs : list of (client_id, sku, start_dt, history) tuples
        max_steps    : max events per session before forced stop
        temperature  : sampling temperature

        Returns
        -------
        List[List[dict]] -one event-dict list per input
        """
        self.eval()
        device  = next(self.parameters()).device
        B       = len(batch_inputs)
        sku2idx = getattr(self, "_sku2idx", {})
        idx2sku = getattr(self, "_idx2sku", {})

        # Per-session history ->memory [B, 1, d_model] This fucking sucks
        memories = []
        for _, _, _, history in batch_inputs:
            if history:
                hist_emb = self._history_dicts_to_emb(history, device)  # [1, H, d_model]
                mem      = self.history.encode(hist_emb)                 # [1, 1, d_model]
            else:
                mem = torch.zeros(1, 1, self.d_model, device=device)
            memories.append(mem)
        memory = torch.cat(memories, dim=0)  # [B, 1, d_model]

        # Seed tokens
        seed_items = [
            sku2idx.get(int(sku), 0) if sku2idx else min(int(sku) + 1, self.vocab_size - 1)
            for _, sku, _, _ in batch_inputs
        ]
        cur_e = torch.tensor([[ACTION2IDX["page_visit"]] for _ in range(B)],
                             dtype=torch.long, device=device)   # [B, 1]
        cur_i = torch.tensor([[s] for s in seed_items],
                             dtype=torch.long, device=device)   # [B, 1]
        cur_d = torch.zeros(B, 1, dtype=torch.long, device=device)  # [B, 1]

        done             = torch.zeros(B, dtype=torch.bool, device=device)
        last_action_idxs = torch.full((B,), -1, dtype=torch.long, device=device)
        # Ok
        events_out  = [[] for _ in range(B)]
        current_dts = []
        for _, _, start_dt, _ in batch_inputs:
            if not isinstance(start_dt, pd.Timestamp):
                start_dt = pd.Timestamp(start_dt)
            current_dts.append(start_dt)

        for _ in range(max_steps):
            if done.all():
                break

            T   = cur_e.size(1)
            pos = torch.arange(T, device=device).unsqueeze(0)   # [1, T] broadcast
            x   = (
                self.event_emb(cur_e)
                + self.item_emb(cur_i)
                + self.delta_emb(cur_d)
                + self.pos_emb(pos)
            )  # [B, T, d_model]

            tgt_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)
            h   = self.decoder(x, memory, tgt_mask=tgt_mask)  # [B, T, d_model]
            h_t = h[:, -1, :]                                  # [B, d_model]

            #  Action
            a_logits = self.action_head.fc(h_t) / temperature  # [B, N_ACTIONS]
            a_logits = self.action_head.constraint_mask.apply_batch(a_logits, last_action_idxs)
            a_probs  = F.softmax(a_logits, dim=-1)
            a_idxs   = torch.multinomial(a_probs, 1).squeeze(1)  # [B]

            newly_done = (a_idxs == EOS_IDX) | done

            # Item (item-bearing, non-done sessions only)
            is_item = torch.zeros(B, dtype=torch.bool, device=device)
            for idx in ITEM_BEARING_IDX:
                is_item |= (a_idxs == idx)
            is_item &= ~done

            i_idxs = torch.zeros(B, dtype=torch.long, device=device)
            if is_item.any():
                h_item   = h_t[is_item]       # [M, d_model]
                a_item   = a_idxs[is_item]    # [M]
                i_logits = self.item_head(h_item, a_item) / temperature  # [M, vocab_size]
                k        = min(self.item_head.top_k, self.vocab_size)
                topk_v, topk_ids = torch.topk(i_logits, k, dim=-1)      # [M, k]
                i_probs  = F.softmax(topk_v, dim=-1)
                chosen   = torch.multinomial(i_probs, 1).squeeze(1)     # [M]
                i_idxs[is_item] = topk_ids.gather(1, chosen.unsqueeze(1)).squeeze(1)
            # This is bad but works for now
            # Temporal
            i_emb_t  = self.item_emb(i_idxs)  # [B, d_model]
            d_logits = self.temporal_head(h_t, a_idxs, i_emb_t) / temperature
            d_probs  = F.softmax(d_logits, dim=-1)
            bin_idxs = torch.multinomial(d_probs, 1).squeeze(1)  # [B]

            # Collect outputs
            for b in range(B):
                if done[b] or a_idxs[b] == EOS_IDX:
                    continue
                a_idx   = int(a_idxs[b].item())
                i_idx   = int(i_idxs[b].item())
                bin_idx = int(bin_idxs[b].item())
                current_dts[b] = current_dts[b] + pd.Timedelta(seconds=bin_to_seconds(bin_idx))
                sku_out = None
                if i_idx > 0:
                    sku_out = idx2sku.get(i_idx, i_idx - 1) if idx2sku else i_idx - 1
                events_out[b].append({
                    "client_id":  batch_inputs[b][0],
                    "event_type": IDX2ACTION[a_idx],
                    "sku":        sku_out,
                    "timestamp":  current_dts[b],
                })

            # Grow sequence tensors (all sessions, incl. done -keeps shape uniform)
            cur_e = torch.cat([cur_e, a_idxs.unsqueeze(1)], dim=1)
            cur_i = torch.cat([cur_i, i_idxs.unsqueeze(1)], dim=1)
            cur_d = torch.cat([cur_d, bin_idxs.unsqueeze(1)], dim=1)

            last_action_idxs = torch.where(done, last_action_idxs, a_idxs)
            done = newly_done

        return events_out

    # Checkpoint I/O

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "config": {
                "vocab_size":      self.vocab_size,
                "d_model":         self.d_model,
                "n_actions":       self.action_head.fc.out_features,
                "n_layers":        len(self.decoder.layers),
                "n_heads":         self.decoder.layers[0].self_attn.num_heads,
                "n_temporal_bins": self.temporal_head.n_bins,
                "window_H":        self.history.window_H,
            },
        }, path)
        print(f"  SessionTransformer saved ->{path}")

    @classmethod
    def load(
        cls,
        path,
        valid_transitions: dict = None,
        device: str = "cpu",
    ) -> "SessionTransformer":
        path       = Path(path)
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        cfg        = checkpoint["config"]
        model      = cls(
            vocab_size        = cfg["vocab_size"],
            d_model           = cfg["d_model"],
            n_actions         = cfg["n_actions"],
            n_layers          = cfg["n_layers"],
            n_heads           = cfg["n_heads"],
            n_temporal_bins   = cfg["n_temporal_bins"],
            window_H          = cfg["window_H"],
            valid_transitions = valid_transitions or {},
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        print(f"  SessionTransformer loaded ← {path}")
        return model

    # Private helpers

    def _encode_history(
        self,
        history: Optional[dict],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Produce cross-attention memory from history tensors (training) or
        a zero dummy tensor when history is None.

        history : dict with keys 'events', 'items', 'deltas' ->[B, H] tensors
        Returns : [B, 1, d_model]
        """
        if history is None:
            return torch.zeros(batch_size, 1, self.d_model, device=device)

        h_e  = history["events"].to(device)   # [B, H]
        h_i  = history["items"].to(device)
        h_d  = history["deltas"].to(device)
        h_pad = history.get("padding_mask")
        if h_pad is not None:
            h_pad = h_pad.to(device)
        H   = h_e.size(1)
        pos = torch.arange(H, device=device).unsqueeze(0)
        hist_emb = (
            self.event_emb(h_e)
            + self.item_emb(h_i)
            + self.delta_emb(h_d)
            + self.pos_emb(pos)
        )  # [B, H, d_model]
        return self.history.encode(hist_emb, key_padding_mask=h_pad)

    def _history_dicts_to_emb(
        self,
        history: List[dict],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Convert a list of event dicts (past session history) to an embedded
        tensor for CrossSessionHistory.encode(). Truncates to window_H.

        Returns [1, H, d_model].
        """
        history  = history[-self.history.window_H :]
        sku2idx  = getattr(self, "_sku2idx", {})
        h_e, h_i, h_d = [], [], []
        prev_ts = None
        for ev in history:
            a_str = ev.get("event_type", "page_visit")
            h_e.append(ACTION2IDX.get(a_str, 0))
            sku = ev.get("sku")
            if sku is not None:
                h_i.append(sku2idx.get(int(sku), 0) if sku2idx else min(int(sku) + 1, self.vocab_size))
            else:
                h_i.append(0)
            ts = ev.get("timestamp")
            if ts is not None and prev_ts is not None:
                delta_s = (pd.Timestamp(ts) - pd.Timestamp(prev_ts)).total_seconds()
                h_d.append(seconds_to_bin(max(delta_s, TEMPORAL_MIN_S)))
            else:
                h_d.append(0)
            prev_ts = ts

        H   = len(h_e)
        pos = torch.arange(H, device=device).unsqueeze(0)
        e_t = torch.tensor([h_e], dtype=torch.long, device=device)
        i_t = torch.tensor([h_i], dtype=torch.long, device=device)
        d_t = torch.tensor([h_d], dtype=torch.long, device=device)
        return (
            self.event_emb(e_t)
            + self.item_emb(i_t)
            + self.delta_emb(d_t)
            + self.pos_emb(pos)
        )  # [1, H, d_model]
