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
TODO: imports uit funcs halen
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import N_TEMPORAL_BINS, TEMPORAL_MAX_S, TEMPORAL_MIN_S

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

# Break for refactor; put heads in separate heads.py file.
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
    ) -> torch.Tensor:
        """
        Vectorized masked cross-entropy: compute full logits over the entire
        vocab, mask out-of-category items to -inf, then standard CE with global
        SKU targets.  Softmax normalizes only over in-category items.
        """
        cond = h_t + self.action_cond(action_t)
        if self.category_cond is not None:
            cond = cond + self.category_cond(category_t)
        logits = self.fc(cond)                                           # [M, V]
        mask = self.sku_cat.unsqueeze(0) == category_t.unsqueeze(1)      # [M, V]
        logits = logits.masked_fill(~mask, float('-inf'))
        return F.cross_entropy(logits, sku_ids)

    # Polymorphic alias so train.py can call item_head.loss(...) for both
    # ItemHead (flat hierarchical) and SVDPQItemHead (token factored) uniformly.
    def loss(self, h_t, action_t, category_t, sku_ids):
        return self.hierarchical_loss(h_t, action_t, category_t, sku_ids)


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

    def __init__(self, d_model: int, t: int, v: int, n_categories: int = 0, top_k: int = 100):
        super().__init__()
        self.t = t
        self.v = v
        self.top_k = top_k
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
    ) -> torch.Tensor:
        """Mean CE over M*t token positions."""
        cond = self._cond(h_t, action_t, category_t)
        logits = self.fc(cond).view(-1, self.t, self.v)        # [M, t, v]
        targets = self.sku_tokens[sku_ids]                     # [M, t]
        return F.cross_entropy(
            logits.reshape(-1, self.v),
            targets.reshape(-1),
        )

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

# Break for refactoring
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
        n_categories: int = 0,    # 0 = no category head (backward-compat)
        sku_price_table: Optional[np.ndarray] = None,  # [V] long, 0=PAD, real=bucket+1
        sku_name_table:  Optional[np.ndarray] = None,  # [V, 16] long, 0=PAD
        price_bins: int = 100,
        name_vocab: int = 256,
        name_len: int = 16,
        sku_tokens_table: Optional[np.ndarray] = None, # [V, svdpq_t] int, 0=PAD row
        svdpq_t: int = 0,                              # tokens per item (0 = disabled)
        svdpq_v: int = 0,                              # bins per dim
        item_head_top_k: int = 100,                    # items kept from item head at inference
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
        self.action_head   = ActionHead(d_model, n_actions)
        self.category_head = CategoryHead(d_model, n_categories) if n_categories > 0 else None
        # SVD-PQ head when sku_tokens_table is supplied; otherwise flat 630K head.
        self.svdpq_enabled = sku_tokens_table is not None and svdpq_t > 0
        if self.svdpq_enabled:
            self.item_head = SVDPQItemHead(
                d_model, t=svdpq_t, v=svdpq_v,
                n_categories=n_categories, top_k=item_head_top_k,
            )
            # Test
            if sku_tokens_table is not None:
                self.item_head.register_sku_tokens(torch.as_tensor(sku_tokens_table, dtype=torch.long))
                # end test
        else:
            self.item_head = ItemHead(d_model, vocab_size, n_categories=n_categories, top_k=item_head_top_k)
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
    ):
        """
        Teacher-forced training pass.

        Input tokens  : t = 0 .. T-2
        Target tokens : t = 1 .. T-1   (shifted by 1)

        Returns (4-tuple):
        action_logits   : [B, T-1, n_actions]
        category_logits : [M, n_categories] at item-bearing positions
        temporal_logits : [B, T-1, n_bins]
        item_ctx        : (h_item, tgt_e_item, tgt_c_item) for hierarchical loss
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

        # Sparse: only compute item/category heads at item-bearing positions
        h_item     = h[item_mask]        # [M, d_model]
        tgt_e_item = tgt_e[item_mask]    # [M]
        tgt_c_item = tgt_c[item_mask]    # [M]
        category_logits = self.category_head(h_item, tgt_e_item)  # [M, n_cats]
        item_ctx = (h_item, tgt_e_item, tgt_c_item)

        return action_logits, category_logits, temporal_logits, item_ctx


    # Break for refactor; inference should maybe be moved to the generator class?
    # class SessionGenerator;
    #   def init (self, model: ses.Transf.)
    #       self.model = model,
    #       self.model.eval()

    #   @torch.no_grad()
    #   def geenrate_batch(sef, batch_input, max_steps, temp)
    #       #Infer batch func


    # Autoregressive inference
    # Batched autoregressive inference with KV cache

    @torch.no_grad()
    def infer_batch(
        self,
        batch_inputs: List[tuple],  # [(client_id, sku, start_dt, history), ...]
        max_steps: int = 50,
        temperature: float = 1.0,
    ) -> List[List[dict]]:
        """
        Batched autoregressive inference.

        Each step concatenates the new token onto the growing token stream and
        runs the full sequence through ``self.decoder``. PyTorch's SDPA backend
        auto-dispatches to FlashAttention / memory-efficient attention when the
        inputs are bf16 and ``need_weights=False`` (the nn.TransformerDecoder
        default), so no hand-rolled KV cache is needed at this sequence length.

        Args:
        batch_inputs : list of (client_id, sku, start_dt, history) tuples
        max_steps    : max events per session before forced stop
        temperature  : sampling temperature

        Returns:
        List[List[dict]] - one event-dict list per input
        """
        self.eval()
        device  = next(self.parameters()).device
        B       = len(batch_inputs)
        sku2idx = getattr(self, "_sku2idx", {})
        idx2sku = getattr(self, "_idx2sku", {})

        autocast_ctx = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else torch.autocast("cpu", enabled=False)
        )

        item_action_ids = torch.tensor(sorted(ITEM_BEARING_IDX), dtype=torch.long, device=device)

        with autocast_ctx:
            # Cross-attention memory [B, 1, d_model]
            memories = []
            for _, _, _, history in batch_inputs:
                if history:
                    mem = self.history.encode(self._history_dicts_to_emb(history, device))
                else:
                    mem = torch.zeros(1, 1, self.d_model, device=device)
                memories.append(mem)
            memory = torch.cat(memories, dim=0)

        # Seed token at position 0: synthetic page_visit on the seed item
        seed_items = [
            sku2idx.get(int(sku), 0) if sku2idx else min(int(sku) + 1, self.vocab_size - 1)
            for _, sku, _, _ in batch_inputs
        ]
        tok_e = torch.full((B, 1), ACTION2IDX["page_visit"], dtype=torch.long, device=device)
        tok_i = torch.tensor([[s] for s in seed_items], dtype=torch.long, device=device)
        tok_d = torch.zeros(B, 1, dtype=torch.long, device=device)
        tok_c = torch.zeros(B, 1, dtype=torch.long, device=device)

        done        = torch.zeros(B, dtype=torch.bool, device=device)
        events_out  = [[] for _ in range(B)]
        current_dts = [
            pd.Timestamp(sdt) if not isinstance(sdt, pd.Timestamp) else sdt
            for _, _, sdt, _ in batch_inputs
        ]
        client_ids = [bi[0] for bi in batch_inputs]

        for step in range(max_steps):
            if done.all():
                break

            T = step + 1
            pos = torch.arange(T, device=device).unsqueeze(0)

            with autocast_ctx:
                x = self._build_input(tok_e, tok_i, tok_d, tok_c, pos)
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)
                h = self.decoder(x, memory, tgt_mask=tgt_mask, tgt_is_causal=True)
                h_t = h[:, -1, :]                                 # [B, d_model]

                # Action
                a_logits = self.action_head(h_t) / temperature
                a_probs  = F.softmax(a_logits.float(), dim=-1)
                a_idxs   = torch.multinomial(a_probs, 1).squeeze(1)
                newly_done = (a_idxs == EOS_IDX) | done

                # Item mask (item-bearing, not already-done)
                is_item = torch.isin(a_idxs, item_action_ids) & ~done

                i_idxs = torch.zeros(B, dtype=torch.long, device=device)
                c_full = torch.zeros(B, dtype=torch.long, device=device)
                if is_item.any():
                    h_item = h_t[is_item]
                    a_item = a_idxs[is_item]

                    cat_logits = self.category_head(h_item, a_item) / temperature
                    cat_probs  = F.softmax(cat_logits.float(), dim=-1)
                    c_idxs     = torch.multinomial(cat_probs, 1).squeeze(1)

                    pools = self._cat_sku_pools
                    c_list = c_idxs.cpu().tolist()
                    item_result = torch.zeros(is_item.sum(), dtype=torch.long, device=device)

                    if isinstance(self.item_head, SVDPQItemHead):
                        # SVD-PQ inference: predict t tokens, then pick the in-category
                        # SKU with highest token-match count (Hamming-based score).
                        pred_tokens = self.item_head.predict_tokens(
                            h_item, a_item, c_idxs,
                        )  # [M, t]
                        for m, ci in enumerate(c_list):
                            pool = pools[ci] if pools is not None and ci < len(pools) else None
                            if pool is not None and len(pool) > 0:
                                pool_tokens = self.item_head.sku_tokens[pool]    # [P, t]
                                match = (pool_tokens == pred_tokens[m]).sum(dim=-1)  # [P]
                                k_fb  = min(self.item_head.top_k, len(pool))
                                top_v, top_idx = match.topk(k_fb)
                                probs = F.softmax(top_v.float() / max(temperature, 1e-6), dim=-1)
                                item_result[m] = pool[top_idx[int(torch.multinomial(probs, 1).item())]]
                            else:
                                # No pool for predicted category: fall back to PAD
                                item_result[m] = 0
                    else:
                        for m, ci in enumerate(c_list):
                            pool = pools[ci] if pools is not None and ci < len(pools) else None
                            h_m  = h_item[m:m+1]
                            a_m  = a_item[m:m+1]
                            c_m  = c_idxs[m:m+1]
                            if pool is not None and len(pool) > 0:
                                pool_k = pool[: min(self.item_head.top_k, len(pool))]
                                w_pool = self.item_head.fc.weight.index_select(0, pool_k)
                                b_pool = self.item_head.fc.bias.index_select(0, pool_k)
                                cond   = (
                                    h_m
                                    + self.item_head.action_cond(a_m)
                                    + self.item_head.category_cond(c_m)
                                )
                                logits = (cond @ w_pool.t() + b_pool) / temperature
                                probs  = F.softmax(logits.float(), dim=-1)
                                item_result[m] = pool_k[int(torch.multinomial(probs, 1).item())]
                            else:
                                # Fallback: global top-k
                                i_logits = self.item_head(h_m, a_m, c_m) / temperature
                                k_fb     = min(self.item_head.top_k, self.vocab_size)
                                topk_v, topk_ids = torch.topk(i_logits, k_fb, dim=-1)
                                probs    = F.softmax(topk_v.float(), dim=-1)
                                item_result[m] = topk_ids[0, int(torch.multinomial(probs, 1).item())]
                    i_idxs[is_item] = item_result
                    c_full[is_item] = c_idxs

                # Temporal
                i_emb_t  = self.item_emb(i_idxs)
                d_logits = self.temporal_head(h_t, a_idxs, i_emb_t) / temperature
                d_probs  = F.softmax(d_logits.float(), dim=-1)
                bin_idxs = torch.multinomial(d_probs, 1).squeeze(1)

            # Bulk CPU transfer, one sync per step instead of 3×B individual .item() calls
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

            # Append the new token to the growing streams
            tok_e = torch.cat([tok_e, a_idxs.unsqueeze(1)],   dim=1)
            tok_i = torch.cat([tok_i, i_idxs.unsqueeze(1)],   dim=1)
            tok_d = torch.cat([tok_d, bin_idxs.unsqueeze(1)], dim=1)
            tok_c = torch.cat([tok_c, c_full.unsqueeze(1)],   dim=1)
            done  = newly_done

        return events_out

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
                "svdpq_enabled":   self.svdpq_enabled,
                "svdpq_t":         self.item_head.t if self.svdpq_enabled else 0,
                "svdpq_v":         self.item_head.v if self.svdpq_enabled else 0,
                "item_head_top_k": self.item_head.top_k,
            },
        }, path)
        print(f"  SessionTransformer saved ->{path}")

    @classmethod
    def load(
        cls,
        path,
        device: str = "cpu",
        **_legacy,   # accept and ignore removed kwargs (e.g. valid_transitions)
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

        # Rebuild SVD-PQ token table (not in state_dict — persistent=False buffer).
        sku_tokens_tbl = None
        if cfg.get("svdpq_enabled", False):
            try:
                from ingestion.dataset import get_sku_tokens
                sku_tokens_tbl = get_sku_tokens()
            except Exception as e:
                print(f"  [warn] failed to load sku_tokens for SVD-PQ inference: {e}")

        model = cls(
            vocab_size        = cfg["vocab_size"],
            d_model           = cfg["d_model"],
            n_actions         = cfg["n_actions"],
            n_layers          = cfg["n_layers"],
            n_heads           = cfg["n_heads"],
            n_temporal_bins   = cfg["n_temporal_bins"],
            window_H          = cfg["window_H"],
            n_categories      = cfg.get("n_categories", 0),
            sku_price_table   = sku_price_tbl,
            sku_name_table    = sku_name_tbl,
            sku_tokens_table  = sku_tokens_tbl,
            svdpq_t           = cfg.get("svdpq_t", 0),
            svdpq_v           = cfg.get("svdpq_v", 0),
            item_head_top_k   = cfg.get("item_head_top_k", 100),
        )

        # test
        if sku_tokens_tbl is not None:
            model.item_head.register_sku_tokens(torch.as_tensor(sku_tokens_tbl))
        # end test
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
