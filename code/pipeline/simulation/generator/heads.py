"""
Prediction heads + shared vocabularies for SessionTransformer.

Contents:
    Action vocabulary constants  : ACTION_TYPES, ACTION2IDX, IDX2ACTION,
                                   EOS_IDX, ITEM_BEARING_IDX, N_ACTIONS
    Temporal bin helpers         : BIN_EDGES, BIN_SECONDS_LUT,
                                   seconds_to_bin, bin_to_seconds
    Five prediction heads        : ActionHead, CategoryHead, ItemHead,
                                   SVDPQItemHead, TemporalHead

Split out of session_transformer.py; no behavior changes. The heads and
constants are imported by session_transformer.py (the nn.Module that hosts
them), plus a few downstream consumers (ingestion/dataset.py,
evaluation/markov.py, train.py) that need the action/temporal vocabularies
to build matching inputs.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import N_TEMPORAL_BINS, TEMPORAL_MAX_S, TEMPORAL_MIN_S


# Action vocabulary, matches ALL_EVENT_TYPES + EOS.
# Note: index 0 is `page_visit` (a real action), NOT a padding slot — PAD is
# handled via `tgt_key_padding_mask` at the transformer level. Keep this in
# mind when reading ACTION2IDX.get(..., 0) lookups downstream.
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


# Temporal bins: N_TEMPORAL_BINS log-spaced edges between TEMPORAL_MIN_S and TEMPORAL_MAX_S
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


# Prediction heads

class ActionHead(nn.Module):
    """Predicts next action (5 event types + EOS)."""

    def __init__(self, d_model: int, n_actions: int):
        super().__init__()
        self.fc = nn.Linear(d_model, n_actions)

    def forward(self, h_t: torch.Tensor) -> torch.Tensor:
        return self.fc(h_t)


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

    def __init__(self, d_model: int, vocab_size: int, n_categories: int = 0):
        super().__init__()
        self.fc          = nn.Linear(d_model, vocab_size)
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

    def register_sku_cat_map(
        self, cat_sku_pools: List[torch.Tensor], vocab_size: int,
    ) -> None:
        """
        Build and register a [V] tensor mapping each SKU embedding index to its
        dense category index.  SKUs not in any pool get category 0 (PAD), which
        will never match a real target category (real cats are >= 1).
        """
        sku_cat = torch.zeros(vocab_size, dtype=torch.long)
        for cat_idx, pool in enumerate(cat_sku_pools):
            if pool.numel() > 0:
                sku_cat[pool.cpu().long()] = cat_idx
        self.register_buffer("sku_cat", sku_cat.to(self.fc.weight.device), persistent=False)

    def hierarchical_loss(
        self,
        h_t: torch.Tensor,            # [M, d_model]
        action_t: torch.Tensor,       # [M]
        category_t: torch.Tensor,     # [M] dense cat indices
        sku_ids: torch.Tensor,        # [M] target sku indices (global)
        sample_weights: Optional[torch.Tensor] = None,  # [M] per-event CE weights
    ) -> torch.Tensor:
        """
        Vectorized masked cross-entropy: compute full logits over the entire
        vocab, mask out-of-category items to -inf, then standard CE with global
        SKU targets.  Softmax normalizes only over in-category items.

        sample_weights: optional [M] tensor of per-event CE weights. None -> mean
        reduction (vanilla CE). Used for inverse-frequency reweighting.
        """
        cond = h_t + self.action_cond(action_t)
        if self.category_cond is not None:
            cond = cond + self.category_cond(category_t)
        logits = self.fc(cond)                                           # [M, V]
        mask = self.sku_cat.unsqueeze(0) == category_t.unsqueeze(1)      # [M, V]
        logits = logits.masked_fill(~mask, float('-inf'))
        if sample_weights is None:
            return F.cross_entropy(logits, sku_ids)
        per = F.cross_entropy(logits, sku_ids, reduction='none')         # [M]
        return (per * sample_weights).sum() / sample_weights.sum().clamp_min(1e-12)

    # Polymorphic alias so train.py can call item_head.loss(...) for both
    # ItemHead (flat hierarchical) and SVDPQItemHead (token factored) uniformly.
    def loss(self, h_t, action_t, category_t, sku_ids, sample_weights=None):
        return self.hierarchical_loss(h_t, action_t, category_t, sku_ids, sample_weights)


class SVDPQItemHead(nn.Module):
    """
    SVD Product-Quantization item head.

    Replaces the flat [d_model, V] softmax with t independent v-way softmaxes
    predicted in parallel from (h_t + action_cond(a) + category_cond(c)).
    Each item is represented offline as a t-tuple of tokens in [0, v-1] via
    per-dim quantile binning of truncated SVD item embeddings
    (see ingestion/svdpq.py).

    Loss: mean cross-entropy across t*M token positions.

    sku_tokens : [V, t] long buffer, populated via register_sku_tokens().
    """

    def __init__(
        self,
        d_model: int,
        t: int,
        v: int,
        n_categories: int = 0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.t = t
        self.v = v
        # Label smoothing bounds per-dim peak probability at (1 - ε + ε/v),
        # which caps the t-factor joint and stops log_prob inference from
        # collapsing to winner-take-all. Training-time only; not a parameter.
        self.label_smoothing = float(label_smoothing)
        # One matmul producing all t*v logits; reshape to [M, t, v].
        self.fc            = nn.Linear(d_model, t * v)
        self.action_cond   = nn.Embedding(N_ACTIONS, d_model)
        if n_categories > 0:
            self.category_cond = nn.Embedding(n_categories, d_model, padding_idx=0)
        else:
            self.category_cond = None

    def register_sku_tokens(self, sku_tokens: torch.Tensor) -> None:
        """sku_tokens: [V, t] integer tensor, values in [0, v-1]."""
        assert sku_tokens.ndim == 2 and sku_tokens.size(1) == self.t, (
            f"sku_tokens must be [V, {self.t}], got {tuple(sku_tokens.shape)}"
        )
        self.register_buffer(
            "sku_tokens",
            sku_tokens.long().to(self.fc.weight.device),
            persistent=False,
        )

    def _cond(self, h_t, action_t, category_t):
        cond = h_t + self.action_cond(action_t)
        if self.category_cond is not None and category_t is not None:
            cond = cond + self.category_cond(category_t)
        return cond

    def loss(
        self,
        h_t: torch.Tensor,        # [M, d_model]
        action_t: torch.Tensor,   # [M]
        category_t: torch.Tensor, # [M] dense cat indices
        sku_ids: torch.Tensor,    # [M] global SKU indices
        sample_weights: Optional[torch.Tensor] = None,  # [M] per-event CE weights
    ) -> torch.Tensor:
        """
        Mean CE over M*t token positions.

        sample_weights: optional [M] per-event weights. Replicated across the t
        token positions of each event so all t tokens for SKU s share weight w_s.
        None -> uniform mean reduction (vanilla CE).
        """
        cond = self._cond(h_t, action_t, category_t)
        logits = self.fc(cond).view(-1, self.t, self.v)        # [M, t, v]
        targets = self.sku_tokens[sku_ids]                     # [M, t]
        if sample_weights is None:
            return F.cross_entropy(
                logits.reshape(-1, self.v),
                targets.reshape(-1),
                label_smoothing=self.label_smoothing,
            )
        per = F.cross_entropy(
            logits.reshape(-1, self.v),
            targets.reshape(-1),
            reduction='none',
            label_smoothing=self.label_smoothing,
        )                                                      # [M*t]
        w = sample_weights.unsqueeze(1).expand(-1, self.t).reshape(-1)  # [M*t]
        return (per * w).sum() / w.sum().clamp_min(1e-12)

    def predict_tokens(
        self,
        h_t: torch.Tensor,
        action_t: torch.Tensor,
        category_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Argmax per dim. Returns [M, t] long token tuple."""
        cond = self._cond(h_t, action_t, category_t)
        logits = self.fc(cond).view(-1, self.t, self.v)
        return logits.argmax(dim=-1)

    def sample_tokens(
        self,
        h_t: torch.Tensor,
        action_t: torch.Tensor,
        category_t: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Sample [M, t] tokens from per-dim softmax(logits / temperature)."""
        cond = self._cond(h_t, action_t, category_t)
        logits = self.fc(cond).view(-1, self.t, self.v)
        probs  = F.softmax(logits.float() / max(temperature, 1e-6), dim=-1)
        M, t, v = probs.shape
        return torch.multinomial(probs.reshape(M * t, v), 1).view(M, t)


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
