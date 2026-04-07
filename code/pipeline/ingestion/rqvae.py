"""
ingestion/rqvae.py — RQ-VAE Item Tokeniser

Offline step: encodes every SKU into a 3-tuple of discrete codes (c1, c2, c3)
from a codebook of size 128.  Run once before training:

    PYTHONPATH=. uv run python ingestion/rqvae.py

Produces:
    output/item_rqvae.pt        trained model weights
    output/sku2codes.joblib     {sku: (c1,c2,c3)} and {(c1,c2,c3): sku}

Architecture
------------
ItemFeatureEncoder
    cat_emb(6912→32) ⊕ price/99(1) ⊕ name_bytes/255(16)  →  Linear(49,128)→ReLU→Linear(128,64)

RQVAEItemTokenizer  (3 residual levels, codebook K=128, dim=64)
    z = encoder(features)
    for each level:
        c_k = nearest_neighbour(z_residual, codebook_k)
        z_residual -= codebook_k[c_k]          # subtract quantised vector
    decoder: sum(quantised vectors) → Linear(64,128)→ReLU→Linear(128,49) → reconstruct features

Loss = MSE(recon, features_norm) + 0.25 * sum(commitment_losses)

References
----------
- van den Oord et al., Neural Discrete Representation Learning, NeurIPS 2017
- Lee et al., Autoregressive Image Generation using Residual Quantization, CVPR 2022
- Rajput et al., Recommender Systems with Generative Retrieval, NeurIPS 2023
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from config import (
    DATA_DIR, MODEL_DIR, OUTPUT_DIR,
    RQVAE_BATCH_SIZE, RQVAE_CAT_EMB_DIM, RQVAE_CODEBOOK_SIZE,
    RQVAE_EPOCHS, RQVAE_LATENT_DIM, RQVAE_LR, RQVAE_N_LEVELS,
    RQVAE_MODEL_PATH, SKU2CODES_PATH,
)

COMMITMENT_BETA = 0.25   # standard VQ-VAE commitment loss weight
CAT_VOCAB_SIZE  = 6913   # 6912 categories + 1 padding (index 0)
NAME_DIM        = 16     # quantised LLM name embedding length
PRICE_DIM       = 1
FEATURE_DIM     = RQVAE_CAT_EMB_DIM + PRICE_DIM + NAME_DIM   # 32+1+16 = 49


# ── Feature encoder ──────────────────────────────────────────────────────────

class ItemFeatureEncoder(nn.Module):
    """
    Encodes per-item tabular features into a latent vector.

    Inputs (all [N]):
        cat        : LongTensor  category ID  (0 = padding/unknown)
        price_norm : FloatTensor price quantile bin / 99  → [0, 1]
        name_bytes : FloatTensor [N, 16]  quantised LLM name embedding / 255 → [0, 1]

    Output: [N, latent_dim]
    """

    def __init__(self, cat_emb_dim: int = RQVAE_CAT_EMB_DIM, latent_dim: int = RQVAE_LATENT_DIM):
        super().__init__()
        self.cat_emb_dim = cat_emb_dim
        self.cat_emb = nn.Embedding(CAT_VOCAB_SIZE, cat_emb_dim, padding_idx=0)
        in_dim = cat_emb_dim + PRICE_DIM + NAME_DIM
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

    def forward(
        self,
        cat: torch.Tensor,         # [N]
        price_norm: torch.Tensor,  # [N]
        name_bytes: torch.Tensor,  # [N, 16]
    ) -> torch.Tensor:             # [N, latent_dim]
        c = self.cat_emb(cat)                              # [N, cat_emb_dim]
        p = price_norm.unsqueeze(1)                        # [N, 1]
        x = torch.cat([c, p, name_bytes], dim=-1)          # [N, 49]
        return self.mlp(x)                                 # [N, latent_dim]


# ── Single VQ level ───────────────────────────────────────────────────────────

class ResidualVectorQuantizer(nn.Module):
    """
    One residual quantisation level.

    forward(z) returns:
        z_q_st  : straight-through quantised vector [N, dim]
        codes   : nearest codebook indices           [N]
        residual: z - codebook[codes] (no grad)      [N, dim]
        commit  : scalar commitment loss
    """

    def __init__(self, codebook_size: int = RQVAE_CODEBOOK_SIZE, dim: int = RQVAE_LATENT_DIM):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook = nn.Embedding(codebook_size, dim)
        # Initialise codebook entries with small uniform noise
        nn.init.uniform_(self.codebook.weight, -0.1, 0.1)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # z: [N, dim]
        # nearest neighbour in codebook
        with torch.no_grad():
            dists = torch.cdist(z, self.codebook.weight)   # [N, K]
            codes = dists.argmin(dim=-1)                   # [N]

        z_q = self.codebook(codes)                         # [N, dim]

        # Straight-through estimator: gradients pass through as if z_q == z
        z_q_st = z + (z_q - z).detach()

        residual = (z - z_q).detach()                      # used as input to next level

        # Commitment loss: encoder output toward codebook (β) + codebook toward encoder
        commit = (
            COMMITMENT_BETA * F.mse_loss(z, z_q.detach())   # encoder commitment
            + F.mse_loss(z.detach(), z_q)                   # codebook EMA-equivalent
        )
        return z_q_st, codes, residual, commit


# ── Full RQ-VAE ───────────────────────────────────────────────────────────────

class RQVAEItemTokenizer(nn.Module):
    """
    3-level Residual Quantisation VAE for item tokenisation.

    Training: forward() returns reconstruction + commitment losses.
    Inference: encode() returns code tuples for all items.
    """

    def __init__(
        self,
        n_levels:      int = RQVAE_N_LEVELS,
        codebook_size: int = RQVAE_CODEBOOK_SIZE,
        latent_dim:    int = RQVAE_LATENT_DIM,
        cat_emb_dim:   int = RQVAE_CAT_EMB_DIM,
    ):
        super().__init__()
        self.n_levels      = n_levels
        self.codebook_size = codebook_size
        self.latent_dim    = latent_dim
        self.cat_emb_dim   = cat_emb_dim

        self.encoder    = ItemFeatureEncoder(cat_emb_dim=cat_emb_dim, latent_dim=latent_dim)
        self.quantizers = nn.ModuleList([
            ResidualVectorQuantizer(codebook_size, latent_dim)
            for _ in range(n_levels)
        ])
        # Decoder: reconstruct normalised feature vector from sum of quantised vectors
        feat_dim = cat_emb_dim + PRICE_DIM + NAME_DIM
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feat_dim),
        )

    def forward(
        self,
        cat:        torch.Tensor,  # [N]
        price_norm: torch.Tensor,  # [N]
        name_bytes: torch.Tensor,  # [N, 16]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (recon [N, 49], total_loss scalar).
        total_loss = MSE(recon, features_norm) + sum(commitment_losses).
        """
        features_norm = torch.cat([
            self.encoder.cat_emb(cat),             # [N, cat_emb_dim] — same normalisation target
            price_norm.unsqueeze(1),               # [N, 1]
            name_bytes,                            # [N, 16]
        ], dim=-1)                                 # [N, 49]  — reconstruction target

        z = self.encoder(cat, price_norm, name_bytes)   # [N, latent_dim]

        z_q_sum      = torch.zeros_like(z)
        total_commit = torch.tensor(0.0, device=z.device)
        z_residual   = z

        for quantizer in self.quantizers:
            z_q_st, _, residual, commit = quantizer(z_residual)
            z_q_sum      = z_q_sum + z_q_st
            total_commit = total_commit + commit
            z_residual   = residual

        recon      = self.decoder(z_q_sum)                          # [N, 49]
        recon_loss = F.mse_loss(recon, features_norm.detach())
        total_loss = recon_loss + total_commit

        return recon, total_loss

    @torch.no_grad()
    def encode(
        self,
        cat:        torch.Tensor,
        price_norm: torch.Tensor,
        name_bytes: torch.Tensor,
    ) -> torch.Tensor:
        """Encode items to code tensor [N, n_levels]. No gradients."""
        z          = self.encoder(cat, price_norm, name_bytes)
        codes_list = []
        z_residual = z

        for quantizer in self.quantizers:
            dists   = torch.cdist(z_residual, quantizer.codebook.weight)  # [N, K]
            codes   = dists.argmin(dim=-1)                                 # [N]
            z_q     = quantizer.codebook(codes)
            codes_list.append(codes)
            z_residual = z_residual - z_q

        return torch.stack(codes_list, dim=1)   # [N, n_levels]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict":    self.state_dict(),
            "config": {
                "n_levels":      self.n_levels,
                "codebook_size": self.codebook_size,
                "latent_dim":    self.latent_dim,
                "cat_emb_dim":   self.cat_emb_dim,
            },
        }, path)
        print(f"  RQVAEItemTokenizer saved → {path}")

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "RQVAEItemTokenizer":
        ckpt   = torch.load(path, map_location=device, weights_only=False)
        cfg    = ckpt["config"]
        model  = cls(
            n_levels      = cfg["n_levels"],
            codebook_size = cfg["codebook_size"],
            latent_dim    = cfg["latent_dim"],
            cat_emb_dim   = cfg.get("cat_emb_dim", RQVAE_CAT_EMB_DIM),
        )
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        model.eval()
        print(f"  RQVAEItemTokenizer loaded ← {path}")
        return model


# ── Data preparation ──────────────────────────────────────────────────────────

def _parse_name_bytes(s: str) -> list[int]:
    """Parse the 16-byte quantised name embedding stored as '[x x x ...]'."""
    nums = re.findall(r'\d+', s)
    return [int(x) for x in nums[:16]]


def load_item_features(device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Loads product_properties.parquet and returns tensors ready for training.

    Returns:
        cat        [N] LongTensor
        price_norm [N] FloatTensor
        name_bytes [N, 16] FloatTensor  (values in [0,1])
        skus       [N] numpy int64 array
    """
    path = DATA_DIR / "product_properties.parquet"
    print(f"  Loading product_properties from {path} ...")
    pp = pl.read_parquet(path)
    N  = pp.height
    print(f"  {N:,} items loaded.")

    # Category: shift by 1 so 0 is padding; raw IDs start from 0
    cats = pp["category"].to_numpy().astype(np.int64) + 1   # [N]  → range [1, 6913]

    # Price: already 0-99 integer
    prices = pp["price"].to_numpy().astype(np.float32) / 99.0   # [N] → [0, 1]

    # Name bytes: parse and normalise
    name_raw = [_parse_name_bytes(s) for s in pp["name"].to_list()]
    # Pad/truncate to exactly 16 values (all should already be 16)
    name_arr = np.array(
        [nb[:16] + [0] * max(0, 16 - len(nb)) for nb in name_raw],
        dtype=np.float32,
    ) / 255.0   # [N, 16] → [0, 1]

    skus = pp["sku"].to_numpy().astype(np.int64)   # [N]

    return (
        torch.from_numpy(cats).to(device),
        torch.from_numpy(prices).to(device),
        torch.from_numpy(name_arr).to(device),
        skus,
    )


# ── Training ──────────────────────────────────────────────────────────────────

def train_rqvae(device: str = "cuda" if torch.cuda.is_available() else "cpu") -> RQVAEItemTokenizer:
    """Train RQVAEItemTokenizer on product features and save checkpoint."""
    print(f"\n=== RQ-VAE Item Tokeniser Training  (device={device}) ===")

    cat_t, price_t, name_t, skus = load_item_features(device="cpu")

    dataset    = TensorDataset(cat_t, price_t, name_t)
    loader     = DataLoader(
        dataset,
        batch_size  = RQVAE_BATCH_SIZE,
        shuffle     = True,
        num_workers = 4,
        pin_memory  = (device == "cuda"),
        persistent_workers = True,
    )

    model     = RQVAEItemTokenizer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=RQVAE_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=RQVAE_EPOCHS, eta_min=1e-5)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    best_loss = float("inf")
    t0        = time.time()

    for epoch in range(1, RQVAE_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for cat_b, price_b, name_b in loader:
            cat_b   = cat_b.to(device)
            price_b = price_b.to(device)
            name_b  = name_b.to(device)

            _, loss = model(cat_b, price_b, name_b)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg = epoch_loss / n_batches

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:>3}/{RQVAE_EPOCHS}  loss={avg:.6f}  {elapsed:.1f}s")

        if avg < best_loss:
            best_loss = avg
            model.save(RQVAE_MODEL_PATH)

    print(f"\n  Training complete. Best loss: {best_loss:.6f}")
    return RQVAEItemTokenizer.load(RQVAE_MODEL_PATH, device=device)


# ── sku2codes export ──────────────────────────────────────────────────────────

def build_sku2codes(
    model: RQVAEItemTokenizer,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[Dict[int, Tuple[int, int, int]], Dict[Tuple[int, int, int], int]]:
    """
    Encode all 1.5M items and build:
        sku2codes : {sku_int: (c1, c2, c3)}
        codes2sku : {(c1, c2, c3): sku_int}   — most-frequent training SKU per triple

    Collision resolution: when multiple SKUs share a code triple, codes2sku keeps
    the SKU with the highest training frequency (from events_clean.parquet).
    SKUs absent from training events default to frequency 0.
    """
    from config import CLEAN_PARQUET, TRAIN_CUTOFF
    print("\n  Loading training SKU frequencies for collision resolution ...")
    freq_df = (
        pl.scan_parquet(str(CLEAN_PARQUET))
        .filter(pl.col("timestamp") < pl.lit(TRAIN_CUTOFF).str.to_datetime())
        .filter(pl.col("sku").is_not_null())
        .group_by("sku")
        .agg(pl.len().alias("freq"))
        .collect()
    )
    sku_freq: Dict[int, int] = {
        int(row["sku"]): int(row["freq"])
        for row in freq_df.iter_rows(named=True)
    }

    print("\n  Encoding all items → code triples ...")
    cat_t, price_t, name_t, skus = load_item_features(device="cpu")

    # Encode in batches to avoid OOM
    batch_size = RQVAE_BATCH_SIZE * 2
    all_codes  = []
    model.eval()

    for start in tqdm(range(0, len(skus), batch_size), desc="  Encoding"):
        end    = min(start + batch_size, len(skus))
        cat_b  = cat_t[start:end].to(device)
        pr_b   = price_t[start:end].to(device)
        nm_b   = name_t[start:end].to(device)
        codes  = model.encode(cat_b, pr_b, nm_b).cpu().numpy()   # [B, 3]
        all_codes.append(codes)

    codes_arr = np.concatenate(all_codes, axis=0)   # [N, 3]

    sku2codes: Dict[int, Tuple[int, int, int]] = {}
    codes2sku: Dict[Tuple[int, int, int], int] = {}

    # Build sku2codes (no collision) and codes2sku (keep highest-frequency SKU per triple)
    for i in range(len(skus)):
        sku    = int(skus[i])
        triple = (int(codes_arr[i, 0]), int(codes_arr[i, 1]), int(codes_arr[i, 2]))
        sku2codes[sku] = triple
        # Keep the more popular SKU on collision
        existing = codes2sku.get(triple)
        if existing is None or sku_freq.get(sku, 0) > sku_freq.get(existing, 0):
            codes2sku[triple] = sku

    n_triples  = len(codes2sku)
    n_possible = RQVAE_CODEBOOK_SIZE ** RQVAE_N_LEVELS
    print(f"  sku2codes: {len(sku2codes):,} entries")
    print(f"  codes2sku: {n_triples:,} unique triples  ({n_triples/n_possible*100:.1f}% of {n_possible:,} possible)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump((sku2codes, codes2sku), SKU2CODES_PATH)
    # Sidecar: tiny file so config.py can read catalog size without loading the full dict
    (OUTPUT_DIR / "catalog_size.txt").write_text(str(len(sku2codes)))
    print(f"  Saved → {SKU2CODES_PATH}")

    return sku2codes, codes2sku


def load_sku2codes() -> Tuple[Dict, Dict]:
    """Load cached sku2codes and codes2sku from disk."""
    if not SKU2CODES_PATH.exists():
        raise FileNotFoundError(
            f"sku2codes not found at {SKU2CODES_PATH}. "
            "Run: PYTHONPATH=. uv run python ingestion/rqvae.py"
        )
    return joblib.load(SKU2CODES_PATH)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = train_rqvae(device=device)
    build_sku2codes(model, device=device)
    print("\nDone. Ready for train.py.")
