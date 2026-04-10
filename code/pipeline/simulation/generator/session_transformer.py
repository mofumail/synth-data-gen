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

from config import N_TEMPORAL_BINS, TEMPORAL_MAX_S, TEMPORAL_MIN_S, VOCAB_K, N_CATEGORIES

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


# Precomputed lookup: avoids np.sqrt on every generated event
BIN_SECONDS_LUT: List[float] = [bin_to_seconds(i) for i in range(N_TEMPORAL_BINS)]


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


class CategoryHead(nn.Module):
    """
    Predicts next item category.
    Conditioned on action_t: category = fc(h_t + action_cond(action_t))

    Only applied at item-bearing positions (add_to_cart, remove_from_cart, product_buy).
    n_categories: 0=PAD, 1=RARE, 2..N=regular categories (built by preprocess.py).
    """

    def __init__(self, d_model: int, n_categories: int):
        super().__init__()
        self.fc          = nn.Linear(d_model, n_categories)
        self.action_cond = nn.Embedding(N_ACTIONS, d_model)
        self.n_categories = n_categories

    def forward(
        self,
        h_t: torch.Tensor,       # [M, d_model]
        action_t: torch.Tensor,  # [M]
    ) -> torch.Tensor:           # [M, n_categories]
        cond = h_t + self.action_cond(action_t)
        return self.fc(cond)


class ItemHead(nn.Module):
    """
    Predicts next SKU. Conditioned on action_t and (optionally) category_t.
    Top-k is enforced at inference time in SessionTransformer.infer().

    Hierarchical conditioning (when category_cond is set):
        cond = h_t + action_cond(action_t) + category_cond(category_t)
    Standard conditioning:
        cond = h_t + action_cond(action_t)
    """

    def __init__(self, d_model: int, vocab_size: int, n_categories: int = 0, top_k: int = 100):
        super().__init__()
        self.fc          = nn.Linear(d_model, vocab_size)
        self.top_k       = top_k
        self.action_cond = nn.Embedding(N_ACTIONS, d_model)
        if n_categories > 0:
            self.category_cond = nn.Embedding(n_categories, d_model, padding_idx=0)
        else:
            self.category_cond = None

    def forward(
        self,
        h_t: torch.Tensor,                        # [B, d_model] or [B, T, d_model]
        action_t: torch.Tensor,                   # [B] or [B, T]
        category_t: Optional[torch.Tensor] = None,  # [B] or [B, T], None for non-hierarchical
    ) -> torch.Tensor:                             # [B, vocab_size] or [B, T, vocab_size]
        cond = h_t + self.action_cond(action_t)
        if self.category_cond is not None and category_t is not None:
            cond = cond + self.category_cond(category_t)
        return self.fc(cond)

    def sampled_logits(
        self,
        h_t: torch.Tensor,            # [M, d_model]
        action_t: torch.Tensor,       # [M]
        positive_ids: torch.Tensor,   # [M]    long, true targets
        negative_ids: torch.Tensor,   # [K]    long, shared negatives for this step
        log_q: torch.Tensor,          # [V]    fp32, log of the proposal Q(w) ∝ U(w)^alpha
        category_t: Optional[torch.Tensor] = None,  # [M] for hierarchical mode
    ) -> torch.Tensor:                # [M, 1 + K]  (positive is column 0)
        """
        Sampled-softmax logits with log-Q correction and accidental-hit masking.

        Target class for the downstream CE is always column 0 (the positive).
        Shared negatives: one draw per step is ~100x faster than per-example and
        is statistically fine because the M positives stratify the shared noise
        through their distinct h_t vectors.
        """
        cond = h_t + self.action_cond(action_t)           # [M, d_model]
        if self.category_cond is not None and category_t is not None:
            cond = cond + self.category_cond(category_t)
        W = self.fc.weight                                # [V, d_model]
        b = self.fc.bias                                  # [V]

        w_pos = W.index_select(0, positive_ids)           # [M, d_model]
        b_pos = b.index_select(0, positive_ids)           # [M]
        w_neg = W.index_select(0, negative_ids)           # [K, d_model]
        b_neg = b.index_select(0, negative_ids)           # [K]

        # Positive logit per row: (cond_m . w_pos_m) + b_pos_m
        logit_pos = (cond * w_pos).sum(dim=-1) + b_pos    # [M]
        # Negative logits: same negatives shared across rows
        logit_neg = cond @ w_neg.t() + b_neg              # [M, K]

        # log-Q correction: s'(w) = s(w) - log Q(w). log_q is fp32; subtraction
        # promotes fp16 logits to fp32 automatically, which CE is happy with.
        logit_pos = logit_pos - log_q.index_select(0, positive_ids)
        logit_neg = logit_neg - log_q.index_select(0, negative_ids).unsqueeze(0)

        # Accidental-hit masking: wherever a sampled negative equals the row's
        # positive, suppress it from the softmax. Per-row mask because each row
        # has a different positive.
        hit = negative_ids.unsqueeze(0).eq(positive_ids.unsqueeze(1))   # [M, K]
        logit_neg = logit_neg.masked_fill(hit, torch.finfo(logit_neg.dtype).min)

        return torch.cat([logit_pos.unsqueeze(1), logit_neg], dim=1)    # [M, 1+K]


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

    Training:
    loss = CE(action_logits, tgt_actions)
         + CE(item_logits[item-bearing positions], tgt_items[item-bearing])
         + CE(temporal_logits, tgt_deltas)

    Inference:
    Use SessionTransformer.infer(client_id, sku, start_dt, history).

    Args:
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
        n_categories: int = 0,    # 0 = no category head (backward-compat)
        sku_price_table: Optional[np.ndarray] = None,  # [V] long, 0=PAD, real=bucket+1
        sku_name_table:  Optional[np.ndarray] = None,  # [V, 16] long, 0=PAD
        price_bins: int = 100,
        name_vocab: int = 256,
        name_len: int = 16,
    ):
        super().__init__()
        self.d_model      = d_model
        self.vocab_size   = vocab_size
        self.n_categories = n_categories

        # Shared embedding tables
        self.event_emb = nn.Embedding(n_actions, d_model)
        self.item_emb  = nn.Embedding(vocab_size + 1, d_model, padding_idx=0)
        self.delta_emb = nn.Embedding(n_temporal_bins, d_model)
        self.pos_emb   = nn.Embedding(512, d_model)

        # --- Input-side auxiliary feature embeddings ---
        # Category as an *input* signal (what the user is currently browsing/buying).
        # Separate from the CategoryHead's `action_cond` — that one is for the target
        # conditional; this one enters the encoder alongside event/item/delta.
        if n_categories > 0:
            self.cat_input_emb = nn.Embedding(n_categories, d_model, padding_idx=0)
        else:
            self.cat_input_emb = None

        # Price bucket as an input signal. Indexed by SKU via lookup table.
        self.price_bins = price_bins
        if sku_price_table is not None:
            self.price_emb = nn.Embedding(price_bins + 1, d_model, padding_idx=0)
            self.register_buffer(
                "sku_price",
                torch.as_tensor(sku_price_table, dtype=torch.long),
                persistent=False,   # not in state_dict; rebuilt from joblib at load
            )
        else:
            self.price_emb = None

        # Quantized name tokens per SKU (16 tokens of 256-way vocab). Mean-pooled
        # into a single d_model vector per item and added to the input sum.
        self.name_vocab = name_vocab
        self.name_len   = name_len
        if sku_name_table is not None:
            self.name_tok_emb = nn.Embedding(name_vocab, d_model)
            self.register_buffer(
                "sku_name",
                torch.as_tensor(sku_name_table, dtype=torch.long),
                persistent=False,
            )
        else:
            self.name_tok_emb = None

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
        self.category_head = CategoryHead(d_model, n_categories) if n_categories > 0 else None
        self.item_head     = ItemHead(d_model, vocab_size, n_categories=n_categories, top_k=100)
        self.temporal_head = TemporalHead(d_model, n_temporal_bins)

        # Populated at inference time via set_cat_sku_pools()
        self._cat_sku_pools = None

    # Shared input builder (forward + infer)
    def _build_input(
        self,
        events: torch.Tensor,      # [B, T] long
        items:  torch.Tensor,      # [B, T] long
        deltas: torch.Tensor,      # [B, T] long
        categories: Optional[torch.Tensor],  # [B, T] long or None
        pos:    torch.Tensor,      # [1, T] or [B, T] long
    ) -> torch.Tensor:              # [B, T, d_model]
        """Sum of all input-side embeddings (event/item/delta/pos + aux features)."""
        x = (
            self.event_emb(events)
            + self.item_emb(items)
            + self.delta_emb(deltas)
            + self.pos_emb(pos)
        )
        if self.cat_input_emb is not None and categories is not None:
            x = x + self.cat_input_emb(categories)
        if self.price_emb is not None:
            x = x + self.price_emb(self.sku_price[items])
        if self.name_tok_emb is not None:
            # [B, T, name_len] -> embed -> [B, T, name_len, d] -> mean over tokens
            name_ids = self.sku_name[items]                        # [B, T, L]
            x = x + self.name_tok_emb(name_ids).mean(dim=-2)       # [B, T, d]
        return x

    # Training forward (teacher-forced, gold conditioning)

    def forward(
        self,
        events: torch.Tensor,                         # [B, T] action indices
        items: torch.Tensor,                          # [B, T] item indices (0=no item)
        deltas: torch.Tensor,                         # [B, T] temporal bin indices
        categories: Optional[torch.Tensor] = None,   # [B, T] dense cat indices (0=PAD)
        history: Optional[dict] = None,               # keys: events/items/deltas ->[B, H]
        tgt_key_padding_mask: Optional[torch.Tensor] = None,  # [B, T-1] True=pad
        item_mask: Optional[torch.Tensor] = None,     # [B, T-1] bool -item positions only
        item_head_mode: str = "full",                 # "full" | "ctx" | "hier"
    ):
        """
        Teacher-forced training pass.

        Input tokens  : t = 0 .. T-2
        Target tokens : t = 1 .. T-1   (shifted by 1)

        Returns (5-tuple):
        action_logits   : [B, T-1, n_actions]
        category_logits : [M, n_categories] at item-bearing positions, or None
        item_logits     : [M, vocab_size] if "full"; None if "ctx"/"hier"
        temporal_logits : [B, T-1, n_bins]
        item_ctx        : (h_item, tgt_e_item, tgt_c_item) if "hier"/"ctx"; else None
                          tgt_c_item is None when category head is not active.

        item_head_mode:
            "full"  - full V-way CE; used for val and non-sampled training.
            "ctx"   - return (h_item, tgt_e_item, None) for global sampled softmax.
            "hier"  - return (h_item, tgt_e_item, tgt_c_item) for within-category
                      sampled softmax; requires categories != None.
        """
        in_e  = events[:, :-1]   # [B, T-1] -model input
        in_i  = items[:, :-1]
        in_d  = deltas[:, :-1]
        in_c  = categories[:, :-1] if categories is not None else None  # input-side categories
        tgt_e = events[:, 1:]    # [B, T-1] -gold next actions (for conditioning)
        tgt_i = items[:, 1:]     # [B, T-1] -gold next items
        tgt_c = categories[:, 1:] if categories is not None else None  # [B, T-1] gold next cats

        B, T = in_e.shape
        device = in_e.device

        pos = torch.arange(T, device=device).unsqueeze(0)   # [1, T]
        x   = self._build_input(in_e, in_i, in_d, in_c, pos)   # [B, T, d_model]

        memory = self._encode_history(history, B, device)  # [B, >=1, d_model]

        # Newer torch requires an explicit tgt_mask when tgt_is_causal=True
        # (otherwise F.multi_head_attention_forward raises under torch.compile).
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)

        h = self.decoder(
            x, memory,
            tgt_mask=tgt_mask,
            tgt_is_causal=True,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )  # [B, T, d_model]

        action_logits = self.action_head(h)                      # [B, T, n_actions]

        # Pre-embed gold next items (shared table) for temporal head
        tgt_item_emb = self.item_emb(tgt_i)                     # [B, T, d_model]
        temporal_logits = self.temporal_head(h, tgt_e, tgt_item_emb)  # [B, T, n_bins]

        category_logits = None
        item_logits     = None
        item_ctx        = None

        if item_mask is not None:
            # Sparse: only compute heads at item-bearing positions -> [M, ...]
            h_item      = h[item_mask]       # [M, d_model]
            tgt_e_item  = tgt_e[item_mask]   # [M]
            tgt_c_item  = tgt_c[item_mask] if tgt_c is not None else None  # [M]

            # Category head (always runs when available, regardless of item_head_mode)
            if self.category_head is not None:
                category_logits = self.category_head(h_item, tgt_e_item)  # [M, n_cats]

            if item_head_mode == "full":
                item_logits = self.item_head(h_item, tgt_e_item, tgt_c_item)
            elif item_head_mode in ("ctx", "hier"):
                item_logits = None
                item_ctx    = (h_item, tgt_e_item, tgt_c_item)
            else:
                raise ValueError(f"unknown item_head_mode={item_head_mode!r}")
        else:
            if item_head_mode == "full":
                item_logits = self.item_head(h, tgt_e, tgt_c)
            else:
                raise ValueError(
                    f"item_head_mode={item_head_mode!r} requires item_mask"
                )

        return action_logits, category_logits, item_logits, temporal_logits, item_ctx

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
        sku         : seed item (0-indexed, from SimpleIdentitySampler)
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

    # Batched autoregressive inference with KV cache

    @torch.no_grad()
    def infer_batch(
        self,
        batch_inputs: List[tuple],  # [(client_id, sku, start_dt, history), ...]
        max_steps: int = 50,
        temperature: float = 1.0,
    ) -> List[List[dict]]:
        """
        KV-cached batch inference.

        Each step processes only the single new token [B, 1, d_model] through the
        decoder, using per-layer caches for all past positions as K/V in self-attention.
        This reduces per-step attention from O(T^2) to O(T) and eliminates the
        growing-tensor torch.cat from the hot path.

        KV cache layout: kv_caches[l] = layer-l inputs for positions 0..t-1, [B, t, d].
        At step t the new token is the query; cat([cache, x_new]) is key/value.

        Args:
        batch_inputs : list of (client_id, sku, start_dt, history) tuples
        max_steps    : max events per session before forced stop
        temperature  : sampling temperature

        Returns:
        List[List[dict]] - one event-dict list per input
        """
        self.eval()
        device   = next(self.parameters()).device
        B        = len(batch_inputs)
        sku2idx  = getattr(self, "_sku2idx", {})
        idx2sku  = getattr(self, "_idx2sku", {})
        n_layers = len(self.decoder.layers)
        use_cuda = device.type == "cuda"

        # bf16 autocast: halves memory bandwidth, enables tensor cores on 4070 Ti.
        # multinomial/indexing stay in fp32 automatically via autocast rules.
        autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_cuda else torch.autocast("cpu", enabled=False)

        with autocast_ctx:
            # Cross-attention memory [B, 1, d_model]
            memories = []
            for _, _, _, history in batch_inputs:
                if history:
                    hist_emb = self._history_dicts_to_emb(history, device)
                    mem      = self.history.encode(hist_emb)
                else:
                    mem = torch.zeros(1, 1, self.d_model, device=device)
                memories.append(mem)
            memory = torch.cat(memories, dim=0)  # [B, 1, d_model]

            # Seed token at position 0
            seed_items = [
                sku2idx.get(int(sku), 0) if sku2idx else min(int(sku) + 1, self.vocab_size - 1)
                for _, sku, _, _ in batch_inputs
            ]
            e0 = torch.tensor([[ACTION2IDX["page_visit"]]] * B, dtype=torch.long, device=device)
            i0 = torch.tensor([[s] for s in seed_items],        dtype=torch.long, device=device)
            d0 = torch.zeros(B, 1, dtype=torch.long, device=device)
            p0 = torch.zeros(1, 1, dtype=torch.long, device=device)
            c0 = torch.zeros(B, 1, dtype=torch.long, device=device)   # seed: PAD category
            x  = self._build_input(e0, i0, d0, c0, p0)

                # Pre-allocate KV buffers [B, max_steps+1, d] — no torch.cat in the hot loop
            kv_dtype = torch.bfloat16 if use_cuda else torch.float32
            kv_buf: List[Tensor] = [
                torch.zeros(B, max_steps + 1, self.d_model, device=device, dtype=kv_dtype)
                for _ in range(n_layers)
            ]
            x = self._run_decoder_prealloc(x, kv_buf, t=0, memory=memory)

        done             = torch.zeros(B, dtype=torch.bool, device=device)
        last_action_idxs = torch.full((B,), -1, dtype=torch.long, device=device)
        events_out       = [[] for _ in range(B)]
        current_dts      = [
            pd.Timestamp(sdt) if not isinstance(sdt, pd.Timestamp) else sdt
            for _, _, sdt, _ in batch_inputs
        ]
        client_ids = [bi[0] for bi in batch_inputs]

        for step in range(max_steps):
            if done.all():
                break

            with autocast_ctx:
                h_t = x[:, 0, :]   # [B, d_model]

                # Action
                a_logits = self.action_head.fc(h_t) / temperature
                a_logits = self.action_head.constraint_mask.apply_batch(a_logits, last_action_idxs)
                a_probs  = F.softmax(a_logits.float(), dim=-1)
                a_idxs   = torch.multinomial(a_probs, 1).squeeze(1)   # [B]

                newly_done = (a_idxs == EOS_IDX) | done

                # Item (hierarchical: sample category first, then item within category)
                is_item = torch.zeros(B, dtype=torch.bool, device=device)
                for idx in ITEM_BEARING_IDX:
                    is_item |= (a_idxs == idx)
                is_item &= ~done

                i_idxs = torch.zeros(B, dtype=torch.long, device=device)
                c_full = torch.zeros(B, dtype=torch.long, device=device)   # category per session for input-side feed
                if is_item.any():
                    h_item = h_t[is_item]
                    a_item = a_idxs[is_item]

                    # --- Category sampling (hierarchical mode) ---
                    if self.category_head is not None and self._cat_sku_pools is not None:
                        cat_logits = self.category_head(h_item, a_item) / temperature
                        cat_probs  = F.softmax(cat_logits.float(), dim=-1)
                        c_idxs     = torch.multinomial(cat_probs, 1).squeeze(1)  # [M]

                        # Per-session: sample item from within-category pool
                        pools = self._cat_sku_pools   # list[Tensor] on device
                        c_list = c_idxs.cpu().tolist()
                        chosen_items = []
                        for ci in c_list:
                            pool = pools[ci] if ci < len(pools) else None
                            if pool is not None and len(pool) > 0:
                                k_pool = min(self.item_head.top_k, len(pool))
                                # Score only items in this category's pool
                                w_pool = self.item_head.fc.weight.index_select(0, pool[:k_pool])
                                b_pool = self.item_head.fc.bias.index_select(0, pool[:k_pool])
                                # Use h_item for the current session (indexed via loop pos)
                                pass   # will be handled per-item below
                                chosen_items.append((pool, ci))
                            else:
                                chosen_items.append((None, ci))

                        # Vectorized per-session within-category item sampling
                        item_result = torch.zeros(is_item.sum(), dtype=torch.long, device=device)
                        for m, (pool, ci) in enumerate(chosen_items):
                            h_m = h_item[m:m+1]   # [1, d_model]
                            a_m = a_item[m:m+1]
                            c_m = c_idxs[m:m+1]
                            if pool is not None and len(pool) > 0:
                                k_pool = min(self.item_head.top_k, len(pool))
                                pool_k = pool[:k_pool]
                                w_pool = self.item_head.fc.weight.index_select(0, pool_k)
                                b_pool = self.item_head.fc.bias.index_select(0, pool_k)
                                cond   = h_m + self.item_head.action_cond(a_m) + self.item_head.category_cond(c_m)
                                logits = (cond @ w_pool.t() + b_pool) / temperature
                                probs  = F.softmax(logits.float(), dim=-1)
                                chosen = int(torch.multinomial(probs, 1).item())
                                item_result[m] = pool_k[chosen]
                            else:
                                # Fallback: global top-k
                                i_logits = self.item_head(h_m, a_m, c_m) / temperature
                                k_fb = min(self.item_head.top_k, self.vocab_size)
                                topk_v, topk_ids = torch.topk(i_logits, k_fb, dim=-1)
                                probs = F.softmax(topk_v.float(), dim=-1)
                                chosen = int(torch.multinomial(probs, 1).item())
                                item_result[m] = topk_ids[0, chosen]
                        i_idxs[is_item] = item_result
                        c_full[is_item] = c_idxs
                    else:
                        # Standard (non-hierarchical) item sampling
                        i_logits = self.item_head(h_item, a_item) / temperature
                        k        = min(self.item_head.top_k, self.vocab_size)
                        topk_v, topk_ids = torch.topk(i_logits, k, dim=-1)
                        i_probs  = F.softmax(topk_v.float(), dim=-1)
                        chosen   = torch.multinomial(i_probs, 1).squeeze(1)
                        i_idxs[is_item] = topk_ids.gather(1, chosen.unsqueeze(1)).squeeze(1)

                # Temporal
                i_emb_t  = self.item_emb(i_idxs)
                d_logits = self.temporal_head(h_t, a_idxs, i_emb_t) / temperature
                d_probs  = F.softmax(d_logits.float(), dim=-1)
                bin_idxs = torch.multinomial(d_probs, 1).squeeze(1)

            # Bulk CPU transfer — one sync per step instead of 3×B individual .item() calls
            a_list    = a_idxs.cpu().tolist()
            i_list    = i_idxs.cpu().tolist()
            bin_list  = bin_idxs.cpu().tolist()
            done_list = done.cpu().tolist()

            for b in range(B):
                if done_list[b] or a_list[b] == EOS_IDX:
                    continue
                a_idx   = a_list[b]
                i_idx   = i_list[b]
                bin_idx = bin_list[b]
                current_dts[b] += pd.Timedelta(seconds=BIN_SECONDS_LUT[bin_idx])
                sku_out = None
                if i_idx > 0:
                    sku_out = idx2sku.get(i_idx, i_idx - 1) if idx2sku else i_idx - 1
                events_out[b].append({
                    "client_id":  client_ids[b],
                    "event_type": IDX2ACTION[a_idx],
                    "sku":        sku_out,
                    "timestamp":  current_dts[b],
                })

            with autocast_ctx:
                pos_t = torch.tensor([[step + 1]], dtype=torch.long, device=device)
                x = self._build_input(
                    a_idxs.unsqueeze(1),
                    i_idxs.unsqueeze(1),
                    bin_idxs.unsqueeze(1),
                    c_full.unsqueeze(1),
                    pos_t,
                )
                x = self._run_decoder_prealloc(x, kv_buf, t=step + 1, memory=memory)

            last_action_idxs = torch.where(done, last_action_idxs, a_idxs)
            done = newly_done

        return events_out

    def _run_decoder_prealloc(
        self,
        x: Tensor,
        kv_buf: List[Tensor],  # [n_layers] of [B, max_steps+1, d_model] pre-allocated
        t: int,                # current time step (0 = seed)
        memory: Tensor,
    ) -> Tensor:
        """
        Pass x [B, 1, d_model] through all decoder layers using pre-allocated KV buffers.
        Writes layer-l input into kv_buf[l][:, t, :] then attends to kv_buf[l][:, :t+1, :].
        No torch.cat — just in-place writes + contiguous slice views.
        """
        for l, layer in enumerate(self.decoder.layers):
            x_in = x
            kv_buf[l][:, t:t+1, :] = x_in           # write layer-l input at step t
            kv_full = kv_buf[l][:, :t+1, :]          # view: positions 0..t (includes x_in)
            x = self._decoder_layer_step(layer, x_in, kv_full, memory)
        return x

    def _decoder_layer_step(
        self,
        layer: nn.TransformerDecoderLayer,
        x_new: Tensor,    # [B, 1, d_model]  — query (current token)
        kv_full: Tensor,  # [B, t+1, d_model] — key/value (all positions 0..t, including x_new)
        memory: Tensor,   # [B, m, d_model]
    ) -> Tensor:          # [B, 1, d_model]
        """
        Single-step post-norm decoder layer forward.
        kv_full already includes x_new at the last position — no torch.cat needed.
        """
        # Self-attention + residual + norm
        sa = layer.self_attn(x_new, kv_full, kv_full, need_weights=False)[0]
        x  = layer.norm1(x_new + layer.dropout1(sa))

        # Cross-attention + residual + norm
        ca = layer.multihead_attn(x, memory, memory, need_weights=False)[0]
        x  = layer.norm2(x + layer.dropout2(ca))

        # FFN + residual + norm
        ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
        x  = layer.norm3(x + layer.dropout3(ff))

        return x

    # Checkpoint I/O

    def set_cat_sku_pools(self, cat_sku_pools: list, device) -> None:
        """
        Register per-category SKU index pools for hierarchical inference.
        cat_sku_pools: list[np.ndarray] indexed by dense_cat_idx (from get_cat_vocab).
        Stored as a list of tensors on the given device.
        """
        self._cat_sku_pools = [
            torch.from_numpy(p).long().to(device) if len(p) > 0
            else torch.tensor([], dtype=torch.long, device=device)
            for p in cat_sku_pools
        ]

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
                "n_categories":    self.n_categories,
                "has_price":       self.price_emb is not None,
                "has_name":        self.name_tok_emb is not None,
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

        # Rebuild SKU property tables (not in state_dict — persistent=False buffers).
        sku_price_tbl = sku_name_tbl = None
        if cfg.get("has_price", False) or cfg.get("has_name", False):
            try:
                from ingestion.dataset import get_sku_properties
                props = get_sku_properties()
                if cfg.get("has_price", False):
                    sku_price_tbl = props["price"]
                if cfg.get("has_name", False):
                    sku_name_tbl  = props["name"]
            except Exception as e:
                print(f"  [warn] failed to load sku_properties for inference: {e}")

        model = cls(
            vocab_size        = cfg["vocab_size"],
            d_model           = cfg["d_model"],
            n_actions         = cfg["n_actions"],
            n_layers          = cfg["n_layers"],
            n_heads           = cfg["n_heads"],
            n_temporal_bins   = cfg["n_temporal_bins"],
            window_H          = cfg["window_H"],
            valid_transitions = valid_transitions or {},
            n_categories      = cfg.get("n_categories", 0),
            sku_price_table   = sku_price_tbl,
            sku_name_table    = sku_name_tbl,
        )
        model._cat_sku_pools = None
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        model.to(device)
        model.eval()

        # Auto-load per-category SKU pools so hierarchical inference actually fires
        # (infer_batch falls back to global top-k when self._cat_sku_pools is None).
        if cfg.get("n_categories", 0) > 0:
            try:
                from config import CAT_SKU_POOLS_PATH
                import joblib as _jl
                if CAT_SKU_POOLS_PATH.exists():
                    pools = _jl.load(CAT_SKU_POOLS_PATH)
                    model.set_cat_sku_pools(pools, device)
                    print(f"  cat_sku_pools loaded ({len(pools)} categories)")
                else:
                    print(f"  [warn] n_categories={cfg['n_categories']} but {CAT_SKU_POOLS_PATH} missing — hierarchical inference disabled")
            except Exception as e:
                print(f"  [warn] failed to auto-load cat_sku_pools: {e}")

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
