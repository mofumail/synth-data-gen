"""
CrossSessionHistory

Attention-pool encoder that compresses up to `window_H` pre-embedded past
events into a single `[B, 1, d_model]` context vector used as cross-attention
memory by the session decoder. Split out of session_transformer.py; no
behavior changes.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


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
