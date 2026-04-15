"""
ingestion/svdpq.py — SVD Product-Quantization Item Tokeniser

Offline step: encodes every in-vocab SKU into a t-tuple of discrete tokens
(per-dim vocab size v) derived from truncated SVD of the user-item interaction
matrix.  Needs to run before train.py & after preprocessing changes.

PYTHONPATH=. uv run python ingestion/svdpq.py

Output is:
    output/sku_tokens.joblib   [V, t] int16 array, values in [0, v-1]

1. Build weighted sparse (user, sku) matrix A from events_clean.parquet using
   SVDPQ_EVENT_WEIGHTS (buy=10, cart=3, page_visit=1, ...). /// Ablation for balances
2. Frequency damp (BM25-style): divide each nonzero by
   log1p(user_rowsum) * log1p(item_colsum) to stop power users / popular items
   from dominating SVD.
3. Per-user L2 normalization
4. Truncated SVD to get item factors E of shape [V, t].
5. Per-dim min-max scale to [0, 1], abnd add small Gaussian noise.
6. Quantize each dim into v bins (quantile or uniform?).
7. Cold-start fallback for SKUs with < SVDPQ_MIN_INTERACTIONS training interactions inherit the quantized
   category-centroid tuple.

Petrov & Macdonald, "Generative Sequential Recommendation with GPTRec",
Gen-IR@SIGIR 2023, §4.2 (SVD Tokenisation)
"""

from __future__ import annotations

import time
from typing import Dict

import joblib
import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD

from config import (
    CLEAN_PARQUET, OUTPUT_DIR, TRAIN_CUTOFF, VOCAB_K,
    SVDPQ_BINNING, SVDPQ_EVENT_WEIGHTS, SVDPQ_MIN_INTERACTIONS,
    SVDPQ_NOISE_STD, SVDPQ_PATH, SVDPQ_T, SVDPQ_V,
)
from ingestion.dataset import get_sku2idx, get_cat_vocab


def _build_interaction_matrix(sku2idx: Dict[int, int]) -> tuple[sp.csr_matrix, np.ndarray]:
    """
    Builds a weighted (user, sku_idx) CSR matrix from training events and return
    (A, interaction_counts) where interaction_counts[sku_idx] is the number of
    distinct (user, event) pairs touching that SKU. The latter drives the
    cold-start fallback.
    """
    print(f"Loading training events from {CLEAN_PARQUET} ...")
    df = (
        pl.scan_parquet(str(CLEAN_PARQUET))
        .filter(pl.col("timestamp") < pl.lit(TRAIN_CUTOFF).str.to_datetime())
        .filter(pl.col("sku").is_not_null())
        .select(["client_id", "sku", "event_type"])
        .collect()
    )
    print(f"{df.height:,} item-bearing training events loaded.")

    # Map SKUs to indices; drop OOV rows (sku not in top-VOCAB_K)
    sku_map = np.full(int(df["sku"].max()) + 1, 0, dtype=np.int64)
    for sku, idx in sku2idx.items():
        sku_map[int(sku)] = idx
    sku_arr = sku_map[df["sku"].cast(pl.Int64).to_numpy()]
    mask = sku_arr > 0
    sku_arr = sku_arr[mask]

    event_arr  = df["event_type"].to_numpy()[mask]
    client_arr = df["client_id"].to_numpy()[mask]

    # Map client_id -> dense row index
    unique_clients, client_row = np.unique(client_arr, return_inverse=True)
    n_users = unique_clients.size
    print(f"{n_users:,} unique users.")

    # Map event_type -> weight
    weight_map = {k: float(v) for k, v in SVDPQ_EVENT_WEIGHTS.items()}
    weights = np.array(
        [weight_map.get(str(e), 0.0) for e in event_arr],
        dtype=np.float32,
    )
    keep = weights > 0
    client_row = client_row[keep]
    sku_arr    = sku_arr[keep]
    weights    = weights[keep]
    print(f"{weights.size:,} weighted interactions after dropping zero-weight events.")

    A = sp.coo_matrix(
        (weights, (client_row, sku_arr)),
        shape=(n_users, VOCAB_K),
        dtype=np.float32,
    ).tocsr()
    # Sum duplicate (user, sku) entries (coo_matrix -> csr sums duplicates)
    A.sum_duplicates()

    # Per-sku interaction count (for cold-start detection): count of nonzero rows
    item_nnz = np.asarray((A > 0).sum(axis=0)).ravel()  # [V]
    return A, item_nnz


def _bm25_damp(A: sp.csr_matrix) -> sp.csr_matrix:
    """Divide A[u, i] by log1p(rowsum[u]) * log1p(colsum[i])."""
    row_sum = np.asarray(A.sum(axis=1)).ravel()
    col_sum = np.asarray(A.sum(axis=0)).ravel()
    row_scale = 1.0 / np.log1p(row_sum + 1e-9)
    col_scale = 1.0 / np.log1p(col_sum + 1e-9)
    # A' = diag(row_scale) @ A @ diag(col_scale)
    A_damp = sp.diags(row_scale) @ A @ sp.diags(col_scale)
    return A_damp.tocsr()


def _quantize(E: np.ndarray, v: int, binning: str) -> np.ndarray:
    """Per-dim quantize E [N, t] into int16 tokens in [0, v-1]."""
    N, t = E.shape
    tokens = np.zeros((N, t), dtype=np.int16)

    # Scale each dim to [0, 1]
    E_min = E.min(axis=0, keepdims=True)
    E_max = E.max(axis=0, keepdims=True)
    E_scaled = (E - E_min) / np.clip(E_max - E_min, 1e-9, None)

    rng = np.random.default_rng(42)
    E_scaled = E_scaled + rng.normal(0.0, SVDPQ_NOISE_STD, size=E_scaled.shape).astype(np.float32)

    if binning == "quantile":
        # Quantile edges (v+1 edges -> v bins); drop first and last to get interior edges
        qs = np.linspace(0.0, 1.0, v + 1)[1:-1]
        for k in range(t):
            edges = np.quantile(E_scaled[:, k], qs)
            tokens[:, k] = np.clip(np.digitize(E_scaled[:, k], edges), 0, v - 1).astype(np.int16)
    elif binning == "uniform":
        for k in range(t):
            tokens[:, k] = np.clip((E_scaled[:, k] * v).astype(np.int32), 0, v - 1).astype(np.int16)
    else:
        raise ValueError(f"Unknown SVDPQ_BINNING: {binning!r} (expected 'quantile' or 'uniform')")
    return tokens


def _cold_start_fallback(
    tokens: np.ndarray,         # [V, t] int16
    E: np.ndarray,              # [V, t] pre-quantization float
    item_nnz: np.ndarray,       # [V] int
    min_interactions: int,
    v: int,
    binning: str,
) -> int:
    """
    In-place overwrite of cold-start SKU tokens with the quantized category
    centroid. Uses the SVD float embeddings (E) rather than the tokens for
    centroid arithmetic, then re-quantizes using the same global bin edges as
    the main quantization step.
    Returns the number of SKUs replaced.
    """
    sku2idx = get_sku2idx()
    _, cat_pools = get_cat_vocab(sku2idx)
    V, t = tokens.shape

    cold = item_nnz < min_interactions
    cold[0] = False  # index 0 is PAD, leave untouched
    n_cold = int(cold.sum())
    if n_cold == 0:
        return 0

    # Pre-compute bin edges per dim from the warm set, so cold centroids land
    # in the same discretization as warm items.
    warm = ~cold
    warm_E = E[warm]
    E_min = warm_E.min(axis=0, keepdims=True)
    E_max = warm_E.max(axis=0, keepdims=True)
    warm_scaled = (warm_E - E_min) / np.clip(E_max - E_min, 1e-9, None)

    edges_per_dim: list[np.ndarray] = []
    if binning == "quantile":
        qs = np.linspace(0.0, 1.0, v + 1)[1:-1]
        for k in range(t):
            edges_per_dim.append(np.quantile(warm_scaled[:, k], qs))

    # For each category with at least one warm SKU, compute centroid SVD vector.
    global_centroid = warm_E.mean(axis=0)  # [t]
    replaced = 0
    warm_mask_global = warm  # [V] bool over original indices

    # Build per-SKU category map from cat_pools (sku_idx -> cat_idx)
    sku_cat = np.zeros(V, dtype=np.int64)
    for ci, pool in enumerate(cat_pools):
        if len(pool) > 0:
            sku_cat[np.asarray(pool, dtype=np.int64)] = ci

    # Precompute per-category centroid (warm SKUs only), falling back to global.
    n_cats = len(cat_pools)
    cat_centroid = np.tile(global_centroid, (n_cats, 1))  # [n_cats, t]
    for ci, pool in enumerate(cat_pools):
        if len(pool) == 0:
            continue
        pool = np.asarray(pool, dtype=np.int64)
        pool_warm = pool[warm_mask_global[pool]]
        if pool_warm.size > 0:
            cat_centroid[ci] = E[pool_warm].mean(axis=0)

    # Quantize cold centroids using the warm bin edges
    cold_idx = np.nonzero(cold)[0]
    cold_cats = sku_cat[cold_idx]
    cold_E = cat_centroid[cold_cats]  # [n_cold, t]
    cold_scaled = (cold_E - E_min) / np.clip(E_max - E_min, 1e-9, None)

    for k in range(t):
        if binning == "quantile":
            tok = np.digitize(cold_scaled[:, k], edges_per_dim[k])
        else:  # uniform
            tok = (cold_scaled[:, k] * v).astype(np.int32)
        tokens[cold_idx, k] = np.clip(tok, 0, v - 1).astype(np.int16)
        replaced = n_cold
    return replaced


def build_svdpq_tokens() -> np.ndarray:
    """Run full SVD-PQ tokenization. Returns tokens [V, t] int16."""
    print(f"\nSVD-PQ Item Tokeniser (t={SVDPQ_T}, v={SVDPQ_V}, binning={SVDPQ_BINNING})")
    t0 = time.time()

    sku2idx = get_sku2idx()
    assert len(sku2idx) + 1 <= VOCAB_K, "sku2idx larger than VOCAB_K; rebuild vocab_stats."

    A, item_nnz = _build_interaction_matrix(sku2idx)
    print(f"Interaction matrix nnz={A.nnz:,}  shape={A.shape}")

    print("BM25-style frequency damping ...")
    A = _bm25_damp(A)

    print("Per-user L2 normalizing ...")
    A = normalize(A, norm="l2", axis=1, copy=False)

    print(f"Truncated SVD (n_components={SVDPQ_T}) ...")
    svd = TruncatedSVD(n_components=SVDPQ_T, algorithm="arpack", random_state=42)
    svd.fit(A)
    # components_ is [t, V]; transpose to get [V, t] item embeddings
    E = svd.components_.T.astype(np.float32)   # [V, t]
    print(f"SVD done. Explained variance: {svd.explained_variance_ratio_.sum():.4f}")

    print(f"Quantizing into {SVDPQ_V} bins per dim ...")
    tokens = _quantize(E, SVDPQ_V, SVDPQ_BINNING)

    # Per-dim histogram sanity check
    for k in [0, SVDPQ_T // 2, SVDPQ_T - 1]:
        counts = np.bincount(tokens[:, k], minlength=SVDPQ_V)
        print(
            f"dim {k:>2}: bin counts min={counts.min():>6}  "
            f"max={counts.max():>6}  mean={counts.mean():.1f}"
        )

    print(f"Cold-start fallback (threshold={SVDPQ_MIN_INTERACTIONS}) ...")
    n_replaced = _cold_start_fallback(
        tokens, E, item_nnz, SVDPQ_MIN_INTERACTIONS, SVDPQ_V, SVDPQ_BINNING,
    )
    print(f"{n_replaced:,}/{VOCAB_K:,} SKUs ({n_replaced/VOCAB_K*100:.1f}%) replaced with category centroid.")

    # Zero out PAD row (index 0) — tokens there should never be trained against
    tokens[0] = 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(tokens, SVDPQ_PATH)
    elapsed = time.time() - t0
    print(f"\n  Saved sku_tokens -> {SVDPQ_PATH}  "
          f"shape={tokens.shape}  dtype={tokens.dtype}  ({elapsed:.1f}s)")
    return tokens


def load_sku_tokens() -> np.ndarray:
    """Load cached tokens [V, t] int16 from SVDPQ_PATH."""
    if not SVDPQ_PATH.exists():
        raise FileNotFoundError(
            f"sku_tokens not found at {SVDPQ_PATH}. "
            "Run: PYTHONPATH=. uv run python ingestion/svdpq.py"
        )
    return joblib.load(SVDPQ_PATH)


if __name__ == "__main__":
    build_svdpq_tokens()
    print("\nSVDPQ done, ready for train.py.")
