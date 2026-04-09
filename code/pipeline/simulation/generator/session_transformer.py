"""
Autoregressive transformer for synthetic e-commerce session generation.
Three factored prediction heads: ActionHead, ItemHeadRQVAE, TemporalHead.

Architecture:
    x_t = event_emb(action_t) + sum_k(code_embs[k](code_t_k)) + delta_emb(bin_t) + pos_emb(t)
    h   = CausalTransformerDecoder(x, memory=CrossSessionHistory)
    action_t+1 ~ ActionHead(h_t)
    item_t+1   ~ ItemHeadRQVAE(h_t)  -- 3 sequential 128-way heads (RQ-VAE codes)
    delta_t+1  ~ TemporalHead(h_t, action_t+1, item_emb_t+1)

Training  : teacher-forced; ItemHeadRQVAE/TemporalHead receive gold next codes.
Inference : cascaded sampling, action -> item (3 code levels) -> temporal.
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

from config import (
    N_TEMPORAL_BINS, TEMPORAL_MAX_S, TEMPORAL_MIN_S,
    RQVAE_CODEBOOK_SIZE, RQVAE_N_LEVELS,
)

CODEBOOK_SIZE = RQVAE_CODEBOOK_SIZE   # 128
N_CODE_LEVELS = RQVAE_N_LEVELS        # 3

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
    Applied inside ActionHead at inference time (last_action_idx != None).
    During training last_action_idx is None ->no masking ->standard cross-entropy.

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


class ItemHeadRQVAE(nn.Module):
    """
    Predicts next item as N_CODE_LEVELS sequential 128-way softmaxes.

    Head k receives [h_t, gold_emb_0, ..., gold_emb_{k-1}] concatenated,
    so each level is conditioned on all previous gold code embeddings
    (teacher-forced at train time; sampled codes at inference time).

    This is ~1600× cheaper in memory than a single 630k-way head.
    """

    def __init__(self, d_model: int, codebook_size: int, n_levels: int):
        super().__init__()
        self.n_levels = n_levels
        self.heads = nn.ModuleList([
            nn.Linear(d_model * (1 + k), codebook_size)
            for k in range(n_levels)
        ])

    def forward(
        self,
        h_t: torch.Tensor,              # [M, d_model]
        code_embs_gold: List[Tensor],   # list of (n_levels-1) tensors [M, d_model]
    ) -> List[Tensor]:                  # list of n_levels tensors [M, codebook_size]
        """
        code_embs_gold : gold embeddings for levels 0 .. n_levels-2.
        Head k uses code_embs_gold[:k] as conditioning (none for k=0).
        """
        logits = []
        for k, head in enumerate(self.heads):
            inp = torch.cat([h_t] + code_embs_gold[:k], dim=-1)   # [M, d*(1+k)]
            logits.append(head(inp))                               # [M, codebook_size]
        return logits


class TemporalHead(nn.Module):
    """
    Predicts inter-event delta_t as a categorical over N_TEMPORAL_BINS log bins.

    Fully factored conditioning:
        delta_{t+1} ~ TemporalHead(h_t, action_{t+1}, item_emb_{t+1})

    item_emb_t is the sum of RQ-VAE code embeddings for the next item,
    passed in rather than re-embedded here to avoid duplicate tables.

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
        item_emb_t: torch.Tensor,    # [B, d_model] or [B, T, d_model] -sum of code embeddings
    ) -> torch.Tensor:               # [B, n_bins] or [B, T, n_bins]
        cond = h_t + self.action_cond(action_t) + item_emb_t
        return self.fc(cond)


# SessionTransformer

class SessionTransformer(nn.Module):
    """
    Full autoregressive session generator with RQ-VAE item tokenisation.

    Training:
    loss = CE(action_logits, tgt_actions)
         + mean_k CE(item_logits_list[k][item_positions], tgt_codes[k][item_positions])
         + CE(temporal_logits, tgt_deltas)

    Items are represented as N_CODE_LEVELS=3 codes from a codebook of size 128.
    Each item head is a 128-way softmax conditioned on previous code levels
    (teacher-forced during training, sequential sampling at inference).

    Inference:
    Use SessionTransformer.infer(client_id, sku, start_dt, history).

    Args:
    codebook_size     : RQ-VAE codebook size per level (default CODEBOOK_SIZE=128)
    n_code_levels     : number of RQ-VAE residual levels (default N_CODE_LEVELS=3)
    n_actions         : 5 event types + EOS = 6
    d_model           : transformer hidden dimension (default 128)
    n_layers          : decoder layers (default 4)
    n_heads           : attention heads (must divide d_model)
    n_temporal_bins   : delta_t bins (default 64)
    window_H          : cross-session history window size
    valid_transitions : dict action_str ->set[action_str] for constraint mask
    """

    def __init__(
        self,
        codebook_size: int = CODEBOOK_SIZE,
        n_code_levels: int = N_CODE_LEVELS,
        n_actions: int = N_ACTIONS,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        n_temporal_bins: int = N_TEMPORAL_BINS,
        window_H: int = 50,
        valid_transitions: dict = None,
    ):
        super().__init__()
        self.d_model      = d_model
        self.codebook_size = codebook_size
        self.n_code_levels = n_code_levels

        # Shared embedding tables
        self.event_emb = nn.Embedding(n_actions, d_model)
        # One codebook embedding per RQ-VAE level; item repr = sum of N_CODE_LEVELS lookups
        self.code_embs = nn.ModuleList([
            nn.Embedding(codebook_size, d_model, padding_idx=0)
            for _ in range(n_code_levels)
        ])
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
        self.item_head     = ItemHeadRQVAE(d_model, codebook_size, n_code_levels)
        self.temporal_head = TemporalHead(d_model, n_temporal_bins)

    # Training forward (teacher-forced, gold conditioning)

    def forward(
        self,
        events: torch.Tensor,                         # [B, T] action indices
        codes: torch.Tensor,                          # [B, T, N_CODE_LEVELS] item codes
        deltas: torch.Tensor,                         # [B, T] temporal bin indices
        history: Optional[dict] = None,               # keys: events/codes/deltas ->[B, H]
        tgt_key_padding_mask: Optional[torch.Tensor] = None,  # [B, T-1] True=pad
        item_mask: Optional[torch.Tensor] = None,     # [B, T-1] bool -item positions only
    ):
        """
        Teacher-forced training pass.

        Input tokens  : t = 0 .. T-2
        Target tokens : t = 1 .. T-1   (shifted by 1)

        Returns:
        action_logits   : [B, T-1, n_actions]
        item_logits_list: list of N_CODE_LEVELS tensors [M, codebook_size]
                          (M = item_mask.sum(); only item-bearing positions)
        temporal_logits : [B, T-1, n_bins]

        item_mask should be precomputed in the training loop (valid & tgt_is_item).
        """
        in_e  = events[:, :-1]            # [B, T-1] -model input
        in_c  = codes[:, :-1]             # [B, T-1, N_CODE_LEVELS]
        in_d  = deltas[:, :-1]
        tgt_e = events[:, 1:]             # [B, T-1] -gold next actions
        tgt_c = codes[:, 1:]              # [B, T-1, N_CODE_LEVELS] -gold next codes

        B, T = in_e.shape
        device = in_e.device

        pos = torch.arange(T, device=device).unsqueeze(0)   # [1, T]
        # Item representation: sum of code embeddings across levels
        item_repr = sum(self.code_embs[k](in_c[:, :, k]) for k in range(self.n_code_levels))
        x   = (
            self.event_emb(in_e)
            + item_repr
            + self.delta_emb(in_d)
            + self.pos_emb(pos)
        )  # [B, T, d_model]

        memory   = self._encode_history(history, B, device)  # [B, >=1, d_model]
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)

        h = self.decoder(
            x, memory,
            tgt_mask=tgt_mask,
            tgt_is_causal=True,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )  # [B, T, d_model]

        action_logits = self.action_head(h)                      # [B, T, n_actions]

        # Pre-embed gold next items (sum of code embeddings) for temporal head
        tgt_item_emb = sum(
            self.code_embs[k](tgt_c[:, :, k]) for k in range(self.n_code_levels)
        )  # [B, T, d_model]
        temporal_logits = self.temporal_head(h, tgt_e, tgt_item_emb)  # [B, T, n_bins]

        # Item logits: sparse over item-bearing positions only
        # Gold embeddings for levels 0..n_levels-2 used as sequential conditioning
        if item_mask is not None:
            h_item      = h[item_mask]       # [M, d_model]
            tgt_c_embs  = [
                self.code_embs[k](tgt_c[:, :, k])[item_mask]
                for k in range(self.n_code_levels)
            ]  # list of N_LEVELS tensors [M, d_model]
            # Pass all but last as gold conditioning (last head predicts level N-1)
            item_logits_list = self.item_head(h_item, tgt_c_embs[:-1])
        else:
            # Dense path (not used in normal training; kept for debugging)
            B_, T_ = h.shape[:2]
            h_flat = h.reshape(B_ * T_, self.d_model)
            tgt_c_embs = [
                self.code_embs[k](tgt_c[:, :, k]).reshape(B_ * T_, self.d_model)
                for k in range(self.n_code_levels)
            ]
            item_logits_list = self.item_head(h_flat, tgt_c_embs[:-1])

        return action_logits, item_logits_list, temporal_logits

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

        Args:
        client_id   : user id (from SimpleIdentitySampler)
        sku         : seed item SKU integer
        start_dt    : session start time (datetime or pd.Timestamp)
        history     : past event dicts for cross-session conditioning
        max_steps   : max events before forced stop
        temperature : sampling temperature

        Returns:
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

        sku2codes: dict = getattr(self, "_sku2codes", {})
        codes2sku: dict = getattr(self, "_codes2sku", {})

        # Seed token: primes the generator with the seed item identity.
        seed_codes = sku2codes.get(int(sku), (0, 0, 0))
        cur_e = [ACTION2IDX["page_visit"]]
        cur_c = [list(seed_codes)]   # list of [c0, c1, c2] triples, length T
        cur_d = [0]

        events_out: List[dict] = []
        last_action_idx: Optional[int] = None
        current_dt = start_dt

        for _ in range(max_steps):
            e_t = torch.tensor([cur_e], dtype=torch.long, device=device)     # [1, T]
            c_t = torch.tensor([cur_c], dtype=torch.long, device=device)     # [1, T, 3]
            d_t = torch.tensor([cur_d], dtype=torch.long, device=device)     # [1, T]

            T   = e_t.size(1)
            pos = torch.arange(T, device=device).unsqueeze(0)
            item_repr = sum(self.code_embs[k](c_t[:, :, k]) for k in range(self.n_code_levels))
            x   = (
                self.event_emb(e_t)
                + item_repr
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

            a_t = torch.tensor([a_idx], dtype=torch.long, device=device)

            # Item: sequential N_CODE_LEVELS 128-way samplings
            if a_idx in ITEM_BEARING_IDX:
                sampled_codes = []
                gold_embs = []   # grows with each level: [1, d_model] tensors
                for k, head in enumerate(self.item_head.heads):
                    inp    = torch.cat([h_t] + gold_embs, dim=-1)   # [1, d*(1+k)]
                    logits = head(inp) / temperature
                    c_k    = torch.multinomial(F.softmax(logits, dim=-1), 1)  # [1, 1]
                    gold_embs.append(self.code_embs[k](c_k.squeeze(1)))       # [1, d]
                    sampled_codes.append(int(c_k.item()))
                item_code_triple = tuple(sampled_codes)
                bucket           = codes2sku.get(item_code_triple, None)
                if bucket is None:
                    sku_out = None
                else:
                    sku_arr, prob_arr = bucket
                    if len(sku_arr) == 1:
                        sku_out = int(sku_arr[0])
                    else:
                        sku_out = int(np.random.choice(sku_arr, p=prob_arr))
                i_emb_t          = sum(gold_embs)   # [1, d_model]
            else:
                item_code_triple = (0, 0, 0)
                sku_out = None
                zero_c = torch.zeros(1, dtype=torch.long, device=device)
                i_emb_t = sum(self.code_embs[k](zero_c) for k in range(self.n_code_levels))

            # Temporal bin ->seconds
            d_logits = self.temporal_head(h_t, a_t, i_emb_t) / temperature
            d_probs  = F.softmax(d_logits, dim=-1)
            bin_idx  = int(torch.multinomial(d_probs, 1).item())
            delta_s  = bin_to_seconds(bin_idx)

            current_dt = current_dt + pd.Timedelta(seconds=delta_s)

            events_out.append({
                "client_id":  client_id,
                "event_type": IDX2ACTION[a_idx],
                "sku":        sku_out,
                "timestamp":  current_dt,
            })

            cur_e.append(a_idx)
            cur_c.append(list(item_code_triple))
            cur_d.append(bin_idx)
            last_action_idx = a_idx

        return events_out

    # Batched autoregressive inference

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

        Args:
        batch_inputs : list of (client_id, sku, start_dt, history) tuples
        max_steps    : max events per session before forced stop
        temperature  : sampling temperature

        Returns:
        List[List[dict]] -one event-dict list per input
        """
        self.eval()
        device    = next(self.parameters()).device
        B         = len(batch_inputs)
        sku2codes = getattr(self, "_sku2codes", {})
        codes2sku = getattr(self, "_codes2sku", {})

        # Per-session history ->memory [B, 1, d_model]
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
        seed_code_list = [
            list(sku2codes.get(int(sku), (0, 0, 0)))
            for _, sku, _, _ in batch_inputs
        ]
        cur_e = torch.tensor([[ACTION2IDX["page_visit"]] for _ in range(B)],
                             dtype=torch.long, device=device)       # [B, 1]
        cur_c = torch.tensor([[s] for s in seed_code_list],
                             dtype=torch.long, device=device)       # [B, 1, 3]
        cur_d = torch.zeros(B, 1, dtype=torch.long, device=device)  # [B, 1]

        done             = torch.zeros(B, dtype=torch.bool, device=device)
        last_action_idxs = torch.full((B,), -1, dtype=torch.long, device=device)

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
            item_repr = sum(self.code_embs[k](cur_c[:, :, k]) for k in range(self.n_code_levels))
            x   = (
                self.event_emb(cur_e)
                + item_repr
                + self.delta_emb(cur_d)
                + self.pos_emb(pos)
            )  # [B, T, d_model]

            tgt_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)
            h   = self.decoder(x, memory, tgt_mask=tgt_mask)  # [B, T, d_model]
            h_t = h[:, -1, :]                                  # [B, d_model]

            # Action
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

            # Default codes (0,0,0) for non-item events
            i_code_triples = torch.zeros(B, self.n_code_levels, dtype=torch.long, device=device)

            if is_item.any():
                h_item = h_t[is_item]   # [M, d_model]
                M      = h_item.size(0)
                sampled_codes_m = torch.zeros(M, self.n_code_levels, dtype=torch.long, device=device)
                gold_embs = []
                for k, head in enumerate(self.item_head.heads):
                    inp    = torch.cat([h_item] + gold_embs, dim=-1)   # [M, d*(1+k)]
                    logits = head(inp) / temperature                   # [M, K]
                    c_k    = torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(1)  # [M]
                    gold_embs.append(self.code_embs[k](c_k))           # [M, d]
                    sampled_codes_m[:, k] = c_k
                i_code_triples[is_item] = sampled_codes_m

            # Item embedding for temporal head (recomputed from final code triples)
            i_embs = sum(
                self.code_embs[k](i_code_triples[:, k]) for k in range(self.n_code_levels)
            )  # [B, d_model]

            # Temporal
            d_logits = self.temporal_head(h_t, a_idxs, i_embs) / temperature
            d_probs  = F.softmax(d_logits, dim=-1)
            bin_idxs = torch.multinomial(d_probs, 1).squeeze(1)  # [B]

            # Collect outputs
            for b in range(B):
                if done[b] or a_idxs[b] == EOS_IDX:
                    continue
                a_idx   = int(a_idxs[b].item())
                bin_idx = int(bin_idxs[b].item())
                current_dts[b] = current_dts[b] + pd.Timedelta(seconds=bin_to_seconds(bin_idx))
                sku_out = None
                if is_item[b]:
                    triple = tuple(i_code_triples[b].tolist())
                    bucket = codes2sku.get(triple, None)
                    if bucket is not None:
                        sku_arr, prob_arr = bucket
                        if len(sku_arr) == 1:
                            sku_out = int(sku_arr[0])
                        else:
                            sku_out = int(np.random.choice(sku_arr, p=prob_arr))
                events_out[b].append({
                    "client_id":  batch_inputs[b][0],
                    "event_type": IDX2ACTION[a_idx],
                    "sku":        sku_out,
                    "timestamp":  current_dts[b],
                })

            # Grow sequence tensors (all sessions incl. done -keeps shape uniform)
            cur_e = torch.cat([cur_e, a_idxs.unsqueeze(1)], dim=1)
            cur_c = torch.cat([cur_c, i_code_triples.unsqueeze(1)], dim=1)
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
                "codebook_size":   self.codebook_size,
                "n_code_levels":   self.n_code_levels,
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
            codebook_size     = cfg["codebook_size"],
            n_code_levels     = cfg["n_code_levels"],
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
        print(f"  SessionTransformer loaded <- {path}")
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

        history : dict with keys 'events', 'codes' [B,H,N_LEVELS], 'deltas' ->[B, H]
        Returns : [B, 1, d_model]
        """
        if history is None:
            return torch.zeros(batch_size, 1, self.d_model, device=device)

        h_e  = history["events"].to(device)   # [B, H]
        h_c  = history["codes"].to(device)    # [B, H, N_LEVELS]
        h_d  = history["deltas"].to(device)   # [B, H]
        h_pad = history.get("padding_mask")
        if h_pad is not None:
            h_pad = h_pad.to(device)
        H   = h_e.size(1)
        pos = torch.arange(H, device=device).unsqueeze(0)
        hist_item_repr = sum(
            self.code_embs[k](h_c[:, :, k]) for k in range(self.n_code_levels)
        )
        hist_emb = (
            self.event_emb(h_e)
            + hist_item_repr
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
        history   = history[-self.history.window_H :]
        sku2codes = getattr(self, "_sku2codes", {})
        h_e, h_c, h_d = [], [], []
        prev_ts = None
        for ev in history:
            a_str = ev.get("event_type", "page_visit")
            h_e.append(ACTION2IDX.get(a_str, 0))
            sku = ev.get("sku")
            if sku is not None:
                h_c.append(list(sku2codes.get(int(sku), (0, 0, 0))))
            else:
                h_c.append([0, 0, 0])
            ts = ev.get("timestamp")
            if ts is not None and prev_ts is not None:
                delta_s = (pd.Timestamp(ts) - pd.Timestamp(prev_ts)).total_seconds()
                h_d.append(seconds_to_bin(max(delta_s, TEMPORAL_MIN_S)))
            else:
                h_d.append(0)
            prev_ts = ts

        H   = len(h_e)
        pos = torch.arange(H, device=device).unsqueeze(0)
        e_t = torch.tensor([h_e], dtype=torch.long, device=device)   # [1, H]
        c_t = torch.tensor([h_c], dtype=torch.long, device=device)   # [1, H, 3]
        d_t = torch.tensor([h_d], dtype=torch.long, device=device)   # [1, H]
        hist_item_repr = sum(
            self.code_embs[k](c_t[:, :, k]) for k in range(self.n_code_levels)
        )
        return (
            self.event_emb(e_t)
            + hist_item_repr
            + self.delta_emb(d_t)
            + self.pos_emb(pos)
        )  # [1, H, d_model]
