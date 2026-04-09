"""
ingestion/rqvae.py — RQ-VAE Item Tokeniser (Plan C: item2vec encoder)

Offline step: encodes every SKU into a 3-tuple of discrete codes (c1, c2, c3)
from a codebook of size 128.  Run once before training:

    PYTHONPATH=. uv run python ingestion/rqvae.py

Produces:
    output/item_rqvae.pt        trained model weights
    output/sku2codes.joblib     {sku: (c1,c2,c3)} and {(c1,c2,c3): sku}

Architecture (Plan C)
---------------------
Item2VecEncoder
    item2vec_emb(64)  →  Linear(64,128) → ReLU → Linear(128,64)

RQVAEItemTokenizer  (3 residual levels, codebook K=128, dim=64)
    z = encoder(item2vec_emb)
    for each level:
        c_k = nearest_neighbour(z_residual, codebook_k)
        z_residual -= codebook_k[c_k]          # subtract quantised vector
    decoder: sum(quantised vectors) → Linear(64,128) → ReLU → Linear(128,64) → reconstruct emb

Loss = MSE(recon, item2vec_emb) + 0.25 * sum(commitment_losses)

References
----------
- van den Oord et al., Neural Discrete Representation Learning, NeurIPS 2017
- Lee et al., Autoregressive Image Generation using Residual Quantization, CVPR 2022
- Rajput et al., Recommender Systems with Generative Retrieval, NeurIPS 2023
- Barkan & Koenigstein, Item2Vec: Neural Item Embedding for CF, 2016
"""

from __future__ import annotations

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
    MODEL_DIR, OUTPUT_DIR,
    RQVAE_BATCH_SIZE, RQVAE_CODEBOOK_SIZE,
    RQVAE_EPOCHS, RQVAE_LATENT_DIM, RQVAE_LR, RQVAE_N_LEVELS,
    RQVAE_MODEL_PATH, SKU2CODES_PATH,
    ITEM2VEC_DIM,
)

COMMITMENT_BETA = 0.25   # standard VQ-VAE commitment loss weight


# ── Item2Vec encoder ──────────────────────────────────────────────────────────

class Item2VecEncoder(nn.Module):
    """
    Projects item2vec embeddings into RQ-VAE latent space.

    Input:  emb [N, emb_dim]   (item2vec embedding, zero vector for OOV items)
    Output: z   [N, latent_dim]
    """

    def __init__(self, emb_dim: int = ITEM2VEC_DIM, latent_dim: int = RQVAE_LATENT_DIM):
        super().__init__()
        self.emb_dim = emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:   # [N, emb_dim] → [N, latent_dim]
        return self.mlp(emb)


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
        nn.init.uniform_(self.codebook.weight, -0.1, 0.1)
        # Usage counter reset each epoch; used for dead-entry restart
        self.register_buffer("_usage", torch.zeros(codebook_size, dtype=torch.long))

    def kmeans_init(self, z: torch.Tensor, n_iters: int = 10) -> None:
        """
        Initialise codebook entries with k-means centroids computed from z [N, dim].
        Prevents codebook collapse by ensuring each entry starts near real data.
        Based on TIGER (Rajput et al., NeurIPS 2023).
        """
        K   = self.codebook_size
        idx = torch.randperm(len(z), device=z.device)[:K]
        centroids = z[idx].clone().float()

        for _ in range(n_iters):
            dists       = torch.cdist(z.float(), centroids)    # [N, K]
            assignments = dists.argmin(dim=1)                   # [N]
            new_c = torch.zeros_like(centroids)
            counts = torch.zeros(K, device=z.device)
            new_c.scatter_add_(0, assignments.unsqueeze(1).expand(-1, z.shape[1]), z.float())
            counts.scatter_add_(0, assignments, torch.ones(len(z), device=z.device))
            mask          = counts > 0
            new_c[mask]  /= counts[mask].unsqueeze(1)
            new_c[~mask]  = centroids[~mask]   # keep old centroid for empty clusters
            centroids     = new_c

        with torch.no_grad():
            self.codebook.weight.copy_(centroids.to(self.codebook.weight.dtype))

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # z: [N, dim]
        with torch.no_grad():
            dists = torch.cdist(z, self.codebook.weight)   # [N, K]
            codes = dists.argmin(dim=-1)                   # [N]
            # Track which entries were used this epoch
            self._usage.scatter_add_(0, codes, torch.ones_like(codes, dtype=torch.long))

        z_q = self.codebook(codes)                         # [N, dim]
        z_q_st   = z + (z_q - z).detach()
        residual = (z - z_q).detach()

        commit = (
            COMMITMENT_BETA * F.mse_loss(z, z_q.detach())
            + F.mse_loss(z.detach(), z_q)
        )
        return z_q_st, codes, residual, commit

    @torch.no_grad()
    def restart_dead(self, z_sample: torch.Tensor) -> int:
        """
        Reinitialise unused codebook entries from random encoder outputs.
        Call once per epoch to prevent codebook collapse.
        Resets the usage counter for the next epoch.
        Returns number of entries restarted.
        """
        dead_mask = self._usage == 0
        n_dead    = int(dead_mask.sum().item())
        if n_dead > 0:
            rand_idx = torch.randint(len(z_sample), (n_dead,), device=z_sample.device)
            self.codebook.weight[dead_mask] = z_sample[rand_idx].to(self.codebook.weight.dtype)
        self._usage.zero_()
        return n_dead


# ── Full RQ-VAE ───────────────────────────────────────────────────────────────

class RQVAEItemTokenizer(nn.Module):
    """
    3-level Residual Quantisation VAE for item tokenisation.

    Input: item2vec embeddings [N, emb_dim].
    Training: forward() returns reconstruction + commitment losses.
    Inference: encode() returns code tuples for all items.
    """

    def __init__(
        self,
        n_levels:      int = RQVAE_N_LEVELS,
        codebook_size: int = RQVAE_CODEBOOK_SIZE,
        latent_dim:    int = RQVAE_LATENT_DIM,
        emb_dim:       int = ITEM2VEC_DIM,
    ):
        super().__init__()
        self.n_levels      = n_levels
        self.codebook_size = codebook_size
        self.latent_dim    = latent_dim
        self.emb_dim       = emb_dim

        self.encoder    = Item2VecEncoder(emb_dim=emb_dim, latent_dim=latent_dim)
        self.quantizers = nn.ModuleList([
            ResidualVectorQuantizer(codebook_size, latent_dim)
            for _ in range(n_levels)
        ])
        # Decoder: reconstruct item2vec embedding from sum of quantised vectors
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, emb_dim),
        )

    def forward(
        self,
        emb: torch.Tensor,   # [N, emb_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (recon [N, emb_dim], total_loss scalar).
        total_loss = MSE(recon, emb) + sum(commitment_losses).
        """
        z = self.encoder(emb)   # [N, latent_dim]

        z_q_sum      = torch.zeros_like(z)
        total_commit = torch.tensor(0.0, device=z.device)
        z_residual   = z

        for quantizer in self.quantizers:
            z_q_st, _, residual, commit = quantizer(z_residual)
            z_q_sum      = z_q_sum + z_q_st
            total_commit = total_commit + commit
            z_residual   = residual

        recon      = self.decoder(z_q_sum)                  # [N, emb_dim]
        recon_loss = F.mse_loss(recon, emb.detach())
        total_loss = recon_loss + total_commit

        return recon, total_loss

    @torch.no_grad()
    def encode(self, emb: torch.Tensor) -> torch.Tensor:
        """Encode items to code tensor [N, n_levels]. No gradients."""
        z          = self.encoder(emb)
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
            "state_dict": self.state_dict(),
            "config": {
                "n_levels":      self.n_levels,
                "codebook_size": self.codebook_size,
                "latent_dim":    self.latent_dim,
                "emb_dim":       self.emb_dim,
            },
        }, path)
        print(f"  RQVAEItemTokenizer saved → {path}")

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "RQVAEItemTokenizer":
        ckpt  = torch.load(path, map_location=device, weights_only=False)
        cfg   = ckpt["config"]
        model = cls(
            n_levels      = cfg["n_levels"],
            codebook_size = cfg["codebook_size"],
            latent_dim    = cfg["latent_dim"],
            emb_dim       = cfg.get("emb_dim", ITEM2VEC_DIM),
        )
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        model.eval()
        print(f"  RQVAEItemTokenizer loaded ← {path}")
        return model


# ── Data loading ──────────────────────────────────────────────────────────────

def load_item_embeddings(device: str = "cpu") -> Tuple[torch.Tensor, np.ndarray]:
    """
    Load item2vec embeddings for all catalog SKUs.

    Returns:
        emb  [N, item2vec_dim] FloatTensor
        skus [N] int64 numpy array
    """
    from ingestion.item2vec import load_item2vec
    emb_arr, skus = load_item2vec()
    emb_t = torch.from_numpy(emb_arr.astype(np.float32)).to(device)
    return emb_t, skus


# ── Training ──────────────────────────────────────────────────────────────────

def train_rqvae(device: str = "cuda" if torch.cuda.is_available() else "cpu") -> RQVAEItemTokenizer:
    """Train RQVAEItemTokenizer on item2vec embeddings and save checkpoint."""
    from ingestion.item2vec import ensure_item2vec
    ensure_item2vec()

    print(f"\n=== RQ-VAE Item Tokeniser Training  (device={device}) ===")

    emb_t, skus = load_item_embeddings(device="cpu")
    print(f"  {len(skus):,} items loaded  (emb_dim={emb_t.shape[1]})")

    dataset = TensorDataset(emb_t)
    loader  = DataLoader(
        dataset,
        batch_size         = RQVAE_BATCH_SIZE,
        shuffle            = True,
        num_workers        = 4,
        pin_memory         = (device == "cuda"),
        persistent_workers = True,
    )

    model     = RQVAEItemTokenizer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=RQVAE_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # K-means codebook initialisation (TIGER-style) — prevents codebook collapse
    print("  Initialising codebooks with k-means ...")
    model.eval()
    with torch.no_grad():
        n_init = min(len(emb_t), 50_000)
        idx    = torch.randperm(len(emb_t))[:n_init]
        z_init = model.encoder(emb_t[idx].to(device))
        z_res  = z_init
        for quantizer in model.quantizers:
            quantizer.kmeans_init(z_res)
            dists  = torch.cdist(z_res, quantizer.codebook.weight)
            codes  = dists.argmin(dim=1)
            z_q    = quantizer.codebook(codes)
            z_res  = z_res - z_q
    model.train()
    print("  K-means init done.")

    best_loss  = float("inf")
    no_improve = 0
    patience   = 15
    t0         = time.time()

    epoch_bar = tqdm(range(1, RQVAE_EPOCHS + 1), desc="RQ-VAE", unit="ep")
    for epoch in epoch_bar:
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        last_emb_b = None
        for (emb_b,) in loader:
            emb_b = emb_b.to(device)

            _, loss = model(emb_b)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1
            last_emb_b  = emb_b

        avg = epoch_loss / n_batches
        scheduler.step(avg)
        current_lr = optimizer.param_groups[0]["lr"]

        # Dead-entry restart: reinitialise unused codebook entries from last batch
        with torch.no_grad():
            z_sample   = model.encoder(last_emb_b)
            z_res      = z_sample
            total_dead = 0
            for quantizer in model.quantizers:
                n_dead     = quantizer.restart_dead(z_res)
                total_dead += n_dead
                dists      = torch.cdist(z_res, quantizer.codebook.weight)
                codes      = dists.argmin(dim=1)
                z_res      = z_res - quantizer.codebook(codes)

        epoch_bar.set_postfix(loss=f"{avg:.4f}", lr=f"{current_lr:.1e}", dead=total_dead)

        if avg < best_loss:
            best_loss  = avg
            no_improve = 0
            model.save(RQVAE_MODEL_PATH)
            elapsed = time.time() - t0
            tqdm.write(f"  [epoch {epoch:>3}] New best: loss={best_loss:.6f}  dead={total_dead}  {elapsed:.0f}s")
        else:
            no_improve += 1
            if no_improve >= patience:
                tqdm.write(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
                break

    print(f"\n  Training complete. Best loss: {best_loss:.6f}")
    return RQVAEItemTokenizer.load(RQVAE_MODEL_PATH, device=device)


# ── sku2codes export ──────────────────────────────────────────────────────────

def build_sku2codes(
    model: RQVAEItemTokenizer,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[Dict[int, Tuple[int, int, int]], Dict[Tuple[int, int, int], Tuple[np.ndarray, np.ndarray]]]:
    """
    Encode all catalog items and build:
        sku2codes : {sku_int: (c1, c2, c3)}
        codes2sku : {(c1, c2, c3): (skus: np.int64[N], probs: np.float32[N])}

    Stochastic bucket (Plan C2): each triple maps to ALL SKUs that share it,
    with sampling probabilities proportional to training frequency. At inference
    time the generator samples one SKU from this bucket, unlocking all SKUs that
    collide on a triple instead of only the top-1 most frequent. This removes
    the ~84% coverage ceiling from the old "keep max-freq only" scheme.

    SKUs absent from training events get frequency 0; if every SKU in a triple
    is OOV, the bucket falls back to uniform sampling.
    """
    from collections import defaultdict
    from config import CLEAN_PARQUET, TRAIN_CUTOFF

    print("\n  Loading training SKU frequencies for stochastic bucket weights ...")
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
    emb_t, skus = load_item_embeddings(device="cpu")

    batch_size = RQVAE_BATCH_SIZE * 2
    all_codes  = []
    model.eval()

    for start in tqdm(range(0, len(skus), batch_size), desc="  Encoding"):
        end    = min(start + batch_size, len(skus))
        emb_b  = emb_t[start:end].to(device)
        codes  = model.encode(emb_b).cpu().numpy()   # [B, 3]
        all_codes.append(codes)

    codes_arr = np.concatenate(all_codes, axis=0)   # [N, 3]

    sku2codes: Dict[int, Tuple[int, int, int]] = {}
    buckets: Dict[Tuple[int, int, int], list] = defaultdict(list)

    for i in range(len(skus)):
        sku    = int(skus[i])
        triple = (int(codes_arr[i, 0]), int(codes_arr[i, 1]), int(codes_arr[i, 2]))
        sku2codes[sku] = triple
        buckets[triple].append((sku, sku_freq.get(sku, 0)))

    # Compress each bucket into (sku_arr, prob_arr)
    codes2sku: Dict[Tuple[int, int, int], Tuple[np.ndarray, np.ndarray]] = {}
    bucket_sizes = []
    oov_buckets  = 0
    for triple, sku_freqs in buckets.items():
        sku_arr  = np.array([s for s, _ in sku_freqs], dtype=np.int64)
        freq_arr = np.array([f for _, f in sku_freqs], dtype=np.float64)
        total    = freq_arr.sum()
        if total == 0:
            probs = np.full(len(freq_arr), 1.0 / len(freq_arr), dtype=np.float32)
            oov_buckets += 1
        else:
            probs = (freq_arr / total).astype(np.float32)
        codes2sku[triple] = (sku_arr, probs)
        bucket_sizes.append(len(sku_arr))

    n_triples  = len(codes2sku)
    n_possible = RQVAE_CODEBOOK_SIZE ** RQVAE_N_LEVELS
    sizes = np.array(bucket_sizes)
    print(f"  sku2codes: {len(sku2codes):,} entries")
    print(f"  codes2sku: {n_triples:,} unique triples  ({n_triples/n_possible*100:.1f}% of {n_possible:,} possible)")
    print(f"  bucket sizes: mean={sizes.mean():.2f}  median={int(np.median(sizes))}  "
          f"max={sizes.max()}  p95={int(np.percentile(sizes, 95))}")
    print(f"  all-OOV buckets (uniform fallback): {oov_buckets:,}")
    print(f"  reachable SKUs (sum of bucket sizes): {int(sizes.sum()):,}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump((sku2codes, codes2sku), SKU2CODES_PATH)
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
