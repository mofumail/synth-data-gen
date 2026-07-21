"""
ingestion/item2vec.py — Train Word2Vec skip-gram on real training sessions.

Each session is treated as a sentence of SKU ID strings. The resulting
64-dim embeddings capture co-purchase / co-view co-occurrence structure,
giving RQ-VAE a continuous geometry it can meaningfully partition across
all 3 residual levels (unlike the flat 49-dim tabular feature space).

Usage:
    PYTHONPATH=. uv run python ingestion/item2vec.py

Output:
    output/item2vec_embeddings.joblib  →  (embeddings [N, 64], skus [N])
        embeddings: np.float32 [N_catalog_skus, item2vec_dim]
                    row i corresponds to skus[i]; zero vector for OOV SKUs
        skus:       np.int64  [N_catalog_skus]

References:
    Barkan & Koenigstein, Item2Vec: Neural Item Embedding for CF, 2016
    Rajput et al., Recommender Systems with Generative Retrieval, NeurIPS 2023
"""

from __future__ import annotations

from typing import Tuple

import joblib
import numpy as np
import polars as pl
from gensim.models import Word2Vec

from config import (
    CLEAN_PARQUET, DATA_DIR, OUTPUT_DIR,
    ITEM2VEC_PATH, ITEM2VEC_DIM, ITEM2VEC_EPOCHS, ITEM2VEC_WINDOW, ITEM2VEC_MIN_COUNT,
    TRAIN_CUTOFF,
)


def train_item2vec() -> None:
    """
    Train Word2Vec skip-gram on training sessions and save per-SKU embeddings.

    SKUs not seen in training sessions (long-tail / new items) receive a zero
    embedding vector and will cluster together in the RQ-VAE codebook — acceptable
    since the transformer never seeds generation from unseen items anyway.
    """
    print("\n=== item2vec Training ===")
    print(f"  Loading sessions from {CLEAN_PARQUET} ...")

    df = (
        pl.scan_parquet(str(CLEAN_PARQUET))
        .filter(pl.col("timestamp") < pl.lit(TRAIN_CUTOFF).str.to_datetime())
        .filter(pl.col("sku").is_not_null())
        .select(["session_id", "sku", "timestamp"])
        .with_columns(pl.col("sku").cast(pl.Int64))   # float64→int64 (nulls already filtered)
        .sort(["session_id", "timestamp"])
        .collect()
    )
    print(f"  {len(df):,} item-bearing events across {df['session_id'].n_unique():,} sessions")

    # Build sentences: each session is a list of str(sku)
    sentences = (
        df.group_by("session_id", maintain_order=True)
        .agg(pl.col("sku").cast(pl.Utf8).alias("skus"))   # int64→str, e.g. "12345"
        ["skus"]
        .to_list()
    )
    print(f"  {len(sentences):,} sentences built")

    print(f"  Training Word2Vec (dim={ITEM2VEC_DIM}, window={ITEM2VEC_WINDOW}, "
          f"min_count={ITEM2VEC_MIN_COUNT}, epochs={ITEM2VEC_EPOCHS}) ...")
    model = Word2Vec(
        sentences    = sentences,
        vector_size  = ITEM2VEC_DIM,
        window       = ITEM2VEC_WINDOW,
        min_count    = ITEM2VEC_MIN_COUNT,
        sg           = 1,          # skip-gram
        epochs       = ITEM2VEC_EPOCHS,
        workers      = 4,
        seed         = 42,
    )
    wv_vocab = set(model.wv.key_to_index.keys())
    print(f"  Word2Vec vocab: {len(wv_vocab):,} unique SKUs learned")

    # Build dense embedding matrix aligned to full product catalog
    catalog_path = DATA_DIR / "product_properties.parquet"
    pp = pl.read_parquet(catalog_path, columns=["sku"])
    catalog_skus = pp["sku"].to_numpy().astype(np.int64)
    N = len(catalog_skus)

    embeddings = np.zeros((N, ITEM2VEC_DIM), dtype=np.float32)
    n_found = 0
    for i, sku in enumerate(catalog_skus):
        key = str(sku)
        if key in wv_vocab:
            embeddings[i] = model.wv[key]
            n_found += 1

    coverage = n_found / N * 100
    print(f"  Embedding coverage: {n_found:,} / {N:,} catalog SKUs ({coverage:.1f}%)")
    print(f"  ({N - n_found:,} OOV SKUs will use zero embedding)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump((embeddings, catalog_skus), ITEM2VEC_PATH)
    print(f"  Saved → {ITEM2VEC_PATH}")


def load_item2vec() -> Tuple[np.ndarray, np.ndarray]:
    """Load saved item2vec embeddings. Returns (embeddings [N, dim], skus [N])."""
    if not ITEM2VEC_PATH.exists():
        raise FileNotFoundError(
            f"item2vec embeddings not found at {ITEM2VEC_PATH}. "
            "Run: PYTHONPATH=. uv run python ingestion/item2vec.py"
        )
    return joblib.load(ITEM2VEC_PATH)


def ensure_item2vec() -> None:
    """Train item2vec if the cached embeddings don't exist yet."""
    if ITEM2VEC_PATH.exists():
        print(f"  item2vec embeddings found — skipping training.")
        return
    train_item2vec()


if __name__ == "__main__":
    train_item2vec()
