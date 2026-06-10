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

The head modules and shared action/temporal vocabularies now live in
`simulation/generator/heads.py`; the cross-session attention-pool encoder
lives in `simulation/generator/cross_session_history.py`. This file owns
only the nn.Module that wires those pieces together plus save/load.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import N_TEMPORAL_BINS, TEMPORAL_MIN_S
from simulation.generator.cross_session_history import CrossSessionHistory
from simulation.generator.heads import (
    ACTION2IDX,
    IDX2ACTION,
    EOS_IDX,
    ITEM_BEARING_IDX,
    N_ACTIONS,
    BIN_SECONDS_LUT,
    seconds_to_bin,
    ActionHead,
    CategoryHead,
    ItemHead,
    SVDPQItemHead,
    TemporalHead,
)


class SessionTransformer(nn.Module):
    """
    Full autoregressive session generator.

    Training:
    loss = CE(action_logits, tgt_actions)
         + CE(item_logits[item-bearing positions], tgt_items[item-bearing])
         + CE(temporal_logits, tgt_deltas)

    Inference:
    Use SessionTransformer.infer_batch(batch_inputs, ...).

    Args:
    vocab_size        : unique SKUs (VOCAB_K, typically 200,000)
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
        svdpq_label_smoothing: float = 0.0,            # per-dim CE smoothing ε (train-time only)
        pool_temperature: float = 1.0,                 # SVD-PQ in-pool softmax temperature (eval-time)
        item_head_mode: str = "hier",                  # flat | hier | svdpq
    ):
        super().__init__()
        if item_head_mode not in {"flat", "hier", "svdpq"}:
            raise ValueError(
                f"item_head_mode must be flat|hier|svdpq, got {item_head_mode!r}"
            )
        self.item_head_mode = item_head_mode
        # Iter-1 baseline drops the category entirely (no input emb, no head,
        # no item-head conditioning). Caller may still pass n_categories for
        # backward-compat; we override it locally.
        if item_head_mode == "flat":
            n_categories = 0
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
        # Item head: dispatch on iteration mode.
        #   flat  -> ItemHead with no category cond, plain K-way softmax (iter 1)
        #   hier  -> ItemHead with category cond + hierarchical masked CE (iter 2)
        #   svdpq -> SVDPQItemHead t-way factored softmax (iter 3)
        self.svdpq_enabled = (item_head_mode == "svdpq")
        if item_head_mode == "svdpq":
            self.item_head = SVDPQItemHead(
                d_model, t=svdpq_t, v=svdpq_v,
                n_categories=n_categories,
                label_smoothing=svdpq_label_smoothing,
            )
            if sku_tokens_table is not None:
                self.item_head.register_sku_tokens(torch.as_tensor(sku_tokens_table, dtype=torch.long))
        else:
            # n_categories already forced to 0 above for flat mode, so this
            # builds an ItemHead with category_cond=None and flat_loss is used.
            self.item_head = ItemHead(d_model, vocab_size, n_categories=n_categories)
        self.temporal_head = TemporalHead(d_model, n_temporal_bins)

        # Populated at inference time via set_cat_sku_pools()
        self._cat_sku_pools = None
        self.pool_temperature = pool_temperature

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
        # Iter-1 flat mode has no category supervision at all; tgt_c may be None.
        tgt_c_item = tgt_c[item_mask] if tgt_c is not None else None
        if self.category_head is not None:
            category_logits = self.category_head(h_item, tgt_e_item)  # [M, n_cats]
        else:
            category_logits = None
        item_ctx = (h_item, tgt_e_item, tgt_c_item)

        return action_logits, category_logits, temporal_logits, item_ctx

    # Autoregressive inference

    @torch.no_grad()
    def infer_batch(
        self,
        batch_inputs: List[tuple],  # [(client_id, sku, start_dt, history), ...]
        max_steps: int = 50,
        temperature: float = 1.0,
        item_temperature: float = 1.0,
        svdpq_scorer: str = "hamming",
        pool_temperature: Optional[float] = None,
    ) -> List[List[dict]]:
        """
        Batched autoregressive inference.

        Each step concatenates the new token onto the growing token stream and
        runs the full sequence through ``self.decoder``. PyTorch's SDPA backend
        auto-dispatches to FlashAttention / memory-efficient attention when the
        inputs are bf16 and ``need_weights=False`` (the nn.TransformerDecoder
        default), so no hand-rolled KV cache is needed at this sequence length.

        The per-step body is split into small `_sample_*` helpers defined below;
        this method is the orchestration loop.

        Args:
        batch_inputs     : list of (client_id, sku, start_dt, history) tuples
        max_steps        : max events per session before forced stop
        temperature      : sampling temperature for action/category/temporal heads
        item_temperature : sampling temperature for the item head
        svdpq_scorer     : 'hamming' (sample t tokens, rank pool SKUs by match count)
                           or 'log_prob' (rank pool SKUs by factored log-likelihood).
                           Ignored when item_head is not SVDPQItemHead.
        pool_temperature : softmax temperature applied when sampling the final
                           in-pool SKU under SVD-PQ. None falls back to
                           ``self.pool_temperature`` (set at construction /
                           restored from the checkpoint config).

        Returns:
        List[List[dict]] - one event-dict list per input
        """
        assert svdpq_scorer in ("hamming", "log_prob"), (
            f"svdpq_scorer must be 'hamming' or 'log_prob', got {svdpq_scorer!r}"
        )
        pool_T = pool_temperature if pool_temperature is not None else self.pool_temperature

        self.eval()
        device  = next(self.parameters()).device
        B       = len(batch_inputs)
        idx2sku = getattr(self, "_idx2sku", {})

        autocast_ctx = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else torch.autocast("cpu", enabled=False)
        )

        item_action_ids = torch.tensor(sorted(ITEM_BEARING_IDX), dtype=torch.long, device=device)

        with autocast_ctx:
            memory = self._encode_batch_memories(batch_inputs, device)

        tok_e, tok_i, tok_d, tok_c, client_ids, current_dts = self._seed_tokens(
            batch_inputs, device
        )

        done       = torch.zeros(B, dtype=torch.bool, device=device)
        events_out = [[] for _ in range(B)]

        for step in range(max_steps):
            if done.all():
                break

            T   = step + 1
            pos = torch.arange(T, device=device).unsqueeze(0)

            with autocast_ctx:
                x = self._build_input(tok_e, tok_i, tok_d, tok_c, pos)
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)
                h   = self.decoder(x, memory, tgt_mask=tgt_mask, tgt_is_causal=True)
                h_t = h[:, -1, :]                                 # [B, d_model]

                a_idxs     = self._sample_action(h_t, temperature)
                newly_done = (a_idxs == EOS_IDX) | done
                is_item    = torch.isin(a_idxs, item_action_ids) & ~done

                i_idxs, c_full = self._sample_items(
                    h_t, a_idxs, is_item, temperature, item_temperature, svdpq_scorer, pool_T,
                )

                bin_idxs = self._sample_temporal(h_t, a_idxs, i_idxs, temperature)

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
                # Iter-1 flat mode emits no category at all.
                if self.item_head_mode == "flat":
                    cat_out = None
                else:
                    cat_out = int(c_full[b].item()) if is_item[b] else None
                events_out[b].append({
                    "client_id":  client_ids[b],
                    "event_type": IDX2ACTION[a_idx],
                    "sku":        sku_out,
                    "timestamp":  current_dts[b],
                    "category":   cat_out,
                })

            # Append the new token to the growing streams
            tok_e = torch.cat([tok_e, a_idxs.unsqueeze(1)],   dim=1)
            tok_i = torch.cat([tok_i, i_idxs.unsqueeze(1)],   dim=1)
            tok_d = torch.cat([tok_d, bin_idxs.unsqueeze(1)], dim=1)
            tok_c = torch.cat([tok_c, c_full.unsqueeze(1)],   dim=1)
            done  = newly_done

        return events_out

    # infer_batch per-step helpers

    def _encode_batch_memories(
        self,
        batch_inputs: List[tuple],
        device: torch.device,
    ) -> torch.Tensor:
        """Cross-attention memory [B, 1, d_model] from per-row history lists."""
        memories = []
        for _, _, _, history in batch_inputs:
            if history:
                mem = self.history.encode(self._history_dicts_to_emb(history, device))
            else:
                mem = torch.zeros(1, 1, self.d_model, device=device)
            memories.append(mem)
        return torch.cat(memories, dim=0)

    def _seed_tokens(
        self,
        batch_inputs: List[tuple],
        device: torch.device,
    ):
        """
        Seed token at position 0: one synthetic page_visit per row on the seed
        SKU. Also materializes per-row client_ids and current_dts used by the
        event-emission loop.
        """
        B = len(batch_inputs)
        sku2idx = getattr(self, "_sku2idx", {})
        seed_items = [
            sku2idx.get(int(sku), 0) if sku2idx else min(int(sku) + 1, self.vocab_size - 1)
            for _, sku, _, _ in batch_inputs
        ]
        tok_e = torch.full((B, 1), ACTION2IDX["page_visit"], dtype=torch.long, device=device)
        tok_i = torch.tensor([[s] for s in seed_items], dtype=torch.long, device=device)
        tok_d = torch.zeros(B, 1, dtype=torch.long, device=device)
        tok_c = torch.zeros(B, 1, dtype=torch.long, device=device)
        current_dts = [
            pd.Timestamp(sdt) if not isinstance(sdt, pd.Timestamp) else sdt
            for _, _, sdt, _ in batch_inputs
        ]
        client_ids = [bi[0] for bi in batch_inputs]
        return tok_e, tok_i, tok_d, tok_c, client_ids, current_dts

    def _sample_action(
        self,
        h_t: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """Action head → softmax → multinomial. Returns [B] action indices."""
        a_logits = self.action_head(h_t) / temperature
        a_probs  = F.softmax(a_logits.float(), dim=-1)
        return torch.multinomial(a_probs, 1).squeeze(1)

    def _sample_items(
        self,
        h_t: torch.Tensor,
        a_idxs: torch.Tensor,
        is_item: torch.Tensor,
        temperature: float,
        item_temperature: float,
        svdpq_scorer: str,
        pool_temperature: float,
    ):
        """
        Category + item sampling at item-bearing rows.

        Returns (i_idxs, c_full), both [B] with 0 at non-item rows. Category
        masking via `empty_cat_mask` is applied here once; the per-pool helpers
        trust that every sampled category has a non-empty SKU pool.
        """
        B = h_t.size(0)
        device = h_t.device
        i_idxs = torch.zeros(B, dtype=torch.long, device=device)
        c_full = torch.zeros(B, dtype=torch.long, device=device)
        if not is_item.any():
            return i_idxs, c_full

        h_item = h_t[is_item]
        a_item = a_idxs[is_item]

        # Iter-1 flat baseline: no category, sample directly from full-vocab softmax.
        if self.item_head_mode == "flat":
            i_idxs[is_item] = self._sample_flat_unrestricted(
                h_item, a_item, item_temperature,
            )
            return i_idxs, c_full

        cat_logits = self.category_head(h_item, a_item) / temperature
        # Mask categories with no SKU pool (otherwise we'd emit a PAD-sku event
        # and ValidityLayer would drop the session).
        if hasattr(self, "empty_cat_mask"):
            cat_logits = cat_logits.masked_fill(
                self.empty_cat_mask.unsqueeze(0), float("-inf"),
            )
        cat_probs = F.softmax(cat_logits.float(), dim=-1)
        c_idxs    = torch.multinomial(cat_probs, 1).squeeze(1)

        pools  = self._cat_sku_pools
        c_list = c_idxs.cpu().tolist()
        if isinstance(self.item_head, SVDPQItemHead):
            item_result = self._sample_svdpq_pool(
                h_item, a_item, c_idxs, c_list, pools, item_temperature, svdpq_scorer,
                pool_temperature,
            )
        else:
            item_result = self._sample_flat_pool(
                h_item, a_item, c_idxs, c_list, pools, item_temperature,
            )

        i_idxs[is_item] = item_result
        c_full[is_item] = c_idxs
        return i_idxs, c_full

    def _sample_svdpq_pool(
        self,
        h_item: torch.Tensor,
        a_item: torch.Tensor,
        c_idxs: torch.Tensor,
        c_list: list,
        pools,
        item_temperature: float,
        scorer: str,
        pool_temperature: float,
    ) -> torch.Tensor:
        """
        SVD-PQ pool scoring. Two scorers share the per-dim logits but differ in
        how they map them to a single pool SKU:
          hamming  : sample tokens, rank pool by match count (coarse — integer counts)
          log_prob : rank pool by sum of per-dim log-probs of its tokens (factored likelihood)
        """
        cond   = self.item_head._cond(h_item, a_item, c_idxs)
        t_, v_ = self.item_head.t, self.item_head.v
        logits = self.item_head.fc(cond).view(-1, t_, v_)   # [M, t, v]
        inv_T  = 1.0 / max(item_temperature, 1e-6)
        M_     = logits.size(0)
        item_result = torch.zeros(M_, dtype=torch.long, device=h_item.device)

        if scorer == "hamming":
            probs_tok   = F.softmax(logits.float() * inv_T, dim=-1)
            pred_tokens = torch.multinomial(
                probs_tok.reshape(M_ * t_, v_), 1
            ).view(M_, t_)                                  # [M, t]
            for m, ci in enumerate(c_list):
                pool = pools[ci] if pools is not None and ci < len(pools) else None
                assert pool is not None and len(pool) > 0, (
                    f"empty_cat_mask failed: category {ci} has no SKU pool"
                )
                pool_tokens = self.item_head.sku_tokens[pool]           # [P, t]
                match = (pool_tokens == pred_tokens[m]).sum(dim=-1)     # [P]
                k_fb = len(pool)
                top_v, top_idx = match.topk(k_fb)
                pick_probs = F.softmax(top_v.float() * inv_T, dim=-1)
                item_result[m] = pool[top_idx[int(torch.multinomial(pick_probs, 1).item())]]
        else:  # log_prob
            log_probs = F.log_softmax(logits.float() * inv_T, dim=-1)   # [M, t, v]
            for m, ci in enumerate(c_list):
                pool = pools[ci] if pools is not None and ci < len(pools) else None
                assert pool is not None and len(pool) > 0, (
                    f"empty_cat_mask failed: category {ci} has no SKU pool"
                )
                pool_tokens = self.item_head.sku_tokens[pool]           # [P, t]
                # scores[p] = sum_k log p(pool_tokens[p, k] | ...)
                # Test Test Test
                # scores = log_probs[m].gather(1, pool_tokens.T).sum(dim=0)  # [P]
                # pick_probs = F.softmax(scores, dim=-1)

                scores = log_probs[m].gather(1, pool_tokens.T).sum(dim=0) / self.item_head.t
                # pick_probs = F.softmax(scores, dim=-1)
                # pick_probs = F.softmax(scores / self.pool_temperature, dim=-1)
                pick_probs = F.softmax(scores / pool_temperature, dim=-1)


                # End Test

                item_result[m] = pool[int(torch.multinomial(pick_probs, 1).item())]
        return item_result

    def _sample_flat_unrestricted(
        self,
        h_item: torch.Tensor,
        a_item: torch.Tensor,
        item_temperature: float,
    ) -> torch.Tensor:
        """
        Iter-1 baseline sampling: action-conditioned K-way softmax over the
        full vocab, no category gating, no pool restriction.
        """
        cond = h_item + self.item_head.action_cond(a_item)
        logits = self.item_head.fc(cond) / max(item_temperature, 1e-6)
        # SKU index 0 is PAD — never a valid emission.
        logits[:, 0] = float("-inf")
        probs = F.softmax(logits.float(), dim=-1)
        return torch.multinomial(probs, 1).squeeze(1)

    def _sample_flat_pool(
        self,
        h_item: torch.Tensor,
        a_item: torch.Tensor,
        c_idxs: torch.Tensor,
        c_list: list,
        pools,
        item_temperature: float,
    ) -> torch.Tensor:
        """
        Flat-vocab pool scoring: score only the pool-SKU slice of the item-head
        weight matrix, then softmax over the pool and multinomial.
        """
        M_ = h_item.size(0)
        item_result = torch.zeros(M_, dtype=torch.long, device=h_item.device)
        for m, ci in enumerate(c_list):
            pool = pools[ci] if pools is not None and ci < len(pools) else None
            assert pool is not None and len(pool) > 0, (
                f"empty_cat_mask failed: category {ci} has no SKU pool"
            )
            h_m  = h_item[m:m+1]
            a_m  = a_item[m:m+1]
            c_m  = c_idxs[m:m+1]
            w_pool = self.item_head.fc.weight.index_select(0, pool)
            b_pool = self.item_head.fc.bias.index_select(0, pool)
            cond   = (
                h_m
                + self.item_head.action_cond(a_m)
                + self.item_head.category_cond(c_m)
            )
            logits = (cond @ w_pool.t() + b_pool) / item_temperature
            probs  = F.softmax(logits.float(), dim=-1)
            item_result[m] = pool[int(torch.multinomial(probs, 1).item())]
        return item_result

    def _sample_temporal(
        self,
        h_t: torch.Tensor,
        a_idxs: torch.Tensor,
        i_idxs: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """Temporal head → softmax → multinomial. Returns [B] delta-bin indices."""
        i_emb_t  = self.item_emb(i_idxs)
        d_logits = self.temporal_head(h_t, a_idxs, i_emb_t) / temperature
        d_probs  = F.softmax(d_logits.float(), dim=-1)
        return torch.multinomial(d_probs, 1).squeeze(1)

    # Top-K recommendation baskets (read the item distribution, don't roll out)

    @torch.no_grad()
    def score_basket_batch(
        self,
        batch_inputs: List[tuple],   # [(client_id, sku, start_dt, history), ...]
        k: int = 20,
        rec_action_idx: Optional[int] = None,
        top_c: int = 100,
        item_temperature: float = 1.0,
    ) -> List[List[dict]]:
        """
        Return the top-K most probable SKUs per user as a ranked basket.

        Unlike infer_batch (which samples a behavioral trajectory and yields
        ~1-2 item-bearing events), this reads the model's item marginal
        log p(sku) at step 0, given the user's history and conditioned on the
        caller-provided recommendation action, then takes the K highest-scoring
        items.

        Dispatch on item_head_mode:
            flat  = global top-K over the fc logits (no category).
            hier  = joint log p(cat) + log p(sku|cat), top-K across top-C cats.
            svdpq = same joint, factored per-SKU score.

        Returns one list per input: [{client_id, sku, score, rank}, ...].
        """
        self.eval()
        device  = next(self.parameters()).device
        B       = len(batch_inputs)
        idx2sku = getattr(self, "_idx2sku", {})
        if rec_action_idx is None:
            raise ValueError("rec_action_idx is required for top-K basket scoring")

        autocast_ctx = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else torch.autocast("cpu", enabled=False)
        )

        with autocast_ctx:
            memory = self._encode_batch_memories(batch_inputs, device)
            tok_e, tok_i, tok_d, tok_c, client_ids, _ = self._seed_tokens(batch_inputs, device)
            pos = torch.arange(1, device=device).unsqueeze(0)            # [1, 1]
            x   = self._build_input(tok_e, tok_i, tok_d, tok_c, pos)     # [B, 1, d]
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(1, device=device)
            h   = self.decoder(x, memory, tgt_mask=tgt_mask, tgt_is_causal=True)
            h0  = h[:, -1, :]                                            # [B, d_model]

            a_idxs = torch.full((B,), rec_action_idx, dtype=torch.long, device=device)
            if self.item_head_mode == "flat":
                top_idx, top_score = self._score_flat(h0, a_idxs, k, item_temperature)
            else:
                top_idx, top_score = self._score_pool_joint(h0, a_idxs, k, top_c, item_temperature)

        idx_list   = top_idx.cpu().tolist()
        score_list = top_score.float().cpu().tolist()
        out: List[List[dict]] = [[] for _ in range(B)]
        for b in range(B):
            for rank, (i_idx, sc) in enumerate(zip(idx_list[b], score_list[b])):
                if i_idx <= 0:
                    continue
                sku_out = idx2sku.get(i_idx, i_idx - 1) if idx2sku else i_idx - 1
                out[b].append({
                    "client_id": client_ids[b],
                    "sku":       sku_out,
                    "score":     sc,
                    "rank":      rank,
                })
        return out

    def _score_flat(
        self,
        h0: torch.Tensor,
        a_idxs: torch.Tensor,
        k: int,
        item_temperature: float,
    ):
        """flat mode: global top-K over the action-conditioned fc logits."""
        cond   = h0 + self.item_head.action_cond(a_idxs)
        logits = self.item_head.fc(cond) / max(item_temperature, 1e-6)   # [B, V]
        logits[:, 0] = float("-inf")                                     # PAD
        logp   = F.log_softmax(logits.float(), dim=-1)
        kk     = min(k, logp.size(1))
        top_score, top_idx = logp.topk(kk, dim=-1)
        return top_idx, top_score

    def _score_pool_joint(
        self,
        h0: torch.Tensor,
        a_idxs: torch.Tensor,
        k: int,
        top_c: int,
        item_temperature: float,
    ):
        """
        hier / svdpq: rank categories by log p(cat) (empty pools masked), score
        SKUs in the top-C pools as log p(cat) + log p(sku|cat), then take the
        global top-K per user across those pools.
        """
        B      = h0.size(0)
        device = h0.device
        inv_T  = 1.0 / max(item_temperature, 1e-6)

        cat_logits = self.category_head(h0, a_idxs).float()              # [B, n_cats]
        if hasattr(self, "empty_cat_mask"):
            cat_logits = cat_logits.masked_fill(self.empty_cat_mask.unsqueeze(0), float("-inf"))
        cat_logp = F.log_softmax(cat_logits, dim=-1)                     # [B, n_cats]

        c_take = min(top_c, cat_logp.size(1))
        top_cat_logp, top_cat_idx = cat_logp.topk(c_take, dim=-1)        # [B, C]

        pools    = self._cat_sku_pools
        is_svdpq = isinstance(self.item_head, SVDPQItemHead)
        cat_list = top_cat_idx.cpu().tolist()

        out_idx   = torch.zeros(B, k, dtype=torch.long, device=device)
        out_score = torch.full((B, k), float("-inf"), device=device)

        for b in range(B):
            sku_buf, score_buf = [], []
            hb = h0[b:b + 1]
            ab = a_idxs[b:b + 1]
            for j, ci in enumerate(cat_list[b]):
                pool = pools[ci] if pools is not None and ci < len(pools) else None
                if pool is None or pool.numel() == 0:
                    continue
                c_t = torch.tensor([ci], device=device)
                if is_svdpq:
                    cond   = self.item_head._cond(hb, ab, c_t)
                    logits = self.item_head.fc(cond).view(-1, self.item_head.t, self.item_head.v)
                    logp   = F.log_softmax(logits.float() * inv_T, dim=-1)   # [1, t, v]
                    pool_tokens = self.item_head.sku_tokens[pool]            # [P, t]
                    sku_logp = logp[0].gather(1, pool_tokens.T).sum(dim=0) / self.item_head.t
                else:
                    cond   = hb + self.item_head.action_cond(ab) + self.item_head.category_cond(c_t)
                    w_pool = self.item_head.fc.weight.index_select(0, pool)
                    b_pool = self.item_head.fc.bias.index_select(0, pool)
                    logits = (cond @ w_pool.t() + b_pool) * inv_T            # [1, P]
                    sku_logp = F.log_softmax(logits.float(), dim=-1)[0]      # [P]
                sku_buf.append(pool)
                score_buf.append(top_cat_logp[b, j] + sku_logp)
            if not sku_buf:
                continue
            all_sku   = torch.cat(sku_buf)
            all_score = torch.cat(score_buf)
            kk = min(k, all_sku.numel())
            top_s, top_i = all_score.topk(kk)
            out_idx[b, :kk]   = all_sku[top_i]
            out_score[b, :kk] = top_s
        return out_idx, out_score

    # Checkpoint I/O

    def set_cat_sku_pools(self, cat_sku_pools: list, device) -> None:
        """
        Register per-category SKU index pools for hierarchical inference.
        cat_sku_pools: list[np.ndarray] indexed by dense_cat_idx (from get_cat_vocab).
        Stored as a list of tensors on the given device.

        Also registers `empty_cat_mask` ([n_categories] bool, True = unsamplable)
        so the category sampler cannot pick a category with no SKU pool (would
        otherwise emit a PAD event and get the whole session dropped).
        """
        self._cat_sku_pools = [
            torch.from_numpy(p).long().to(device) if len(p) > 0
            else torch.tensor([], dtype=torch.long, device=device)
            for p in cat_sku_pools
        ]
        n_cats = self.n_categories
        mask = torch.zeros(n_cats, dtype=torch.bool, device=device)
        for i in range(n_cats):
            if i >= len(self._cat_sku_pools) or self._cat_sku_pools[i].numel() == 0:
                mask[i] = True
        mask[0] = True                  # PAD (0) is never samplable
        self.register_buffer("empty_cat_mask", mask, persistent=False)

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
                "item_head_mode":  self.item_head_mode,
                "svdpq_enabled":   self.svdpq_enabled,
                "svdpq_t":         self.item_head.t if self.svdpq_enabled else 0,
                "svdpq_v":         self.item_head.v if self.svdpq_enabled else 0,
                "pool_temperature": self.pool_temperature,
            },
        }, path)
        print(f"  SessionTransformer saved -> {path}")

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

        # Resolve item-head mode with back-compat fallback to legacy `svdpq_enabled`.
        item_head_mode = cfg.get(
            "item_head_mode",
            "svdpq" if cfg.get("svdpq_enabled", False) else "hier",
        )

        # Rebuild SVD-PQ token table (not in state_dict — persistent=False buffer).
        sku_tokens_tbl = None
        if item_head_mode == "svdpq":
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
            pool_temperature  = cfg.get("pool_temperature", 1.0),
            item_head_mode    = item_head_mode,
        )

        model._cat_sku_pools = None
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        model.to(device)
        model.eval()

        # Auto-load per-category SKU pools so hierarchical inference actually fires
        # (infer_batch falls back to global top-k when self._cat_sku_pools is None).
        # Iter-1 flat mode has no category structure -> skip pool loading.
        if item_head_mode != "flat" and cfg.get("n_categories", 0) > 0:
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
