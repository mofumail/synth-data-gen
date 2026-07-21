"""
SimpleIdentitySampler

Seeds each new synthetic session with a (client_id, sku) pair sampled from
real training data, weighted by item frequency.

Replaces CTGAN IdentityFactory. No external utility dependencies.
The CTGAN approach generated synthetic user/item pairs; this approach
samples from observed pairs. Works for an easier start, but is lacking for actual generation

"""

import os
import numpy as np
import pandas as pd
import joblib

from config import CLEAN_PARQUET, TRAIN_CUTOFF, ITEM_BEARING_EVENTS, MODEL_DIR


class SimpleIdentitySampler:
    def __init__(self):
        self.client_ids = None   # array of client_ids
        self.skus       = None   # array of skus (parallel to client_ids)
        self.weights    = None   # sampling weights
        self.is_trained = False

    def fit(self, sample_size: int = 100_000) -> None:
        print(f"\nFitting SimpleIdentitySampler from {CLEAN_PARQUET}...")
        df = pd.read_parquet(
            CLEAN_PARQUET,
            columns=["client_id", "timestamp", "event_type", "sku"],
        )

        train_cutoff = pd.Timestamp(TRAIN_CUTOFF)
        df = df[
            (df["timestamp"] < train_cutoff) &
            (df["event_type"].isin(ITEM_BEARING_EVENTS)) &
            df["sku"].notna()
        ]

        pairs = df.groupby(["client_id", "sku"]).size().reset_index(name="count")
        if len(pairs) > sample_size:
            pairs = pairs.sample(n=sample_size, weights="count", replace=True, random_state=42).drop_duplicates()

        self.client_ids = pairs["client_id"].to_numpy()
        self.skus       = pairs["sku"].to_numpy().astype(int)
        raw_weights     = pairs["count"].to_numpy().astype(float)
        self.weights    = raw_weights / raw_weights.sum()
        self.is_trained = True

        print(f"  Pool size: {len(self.client_ids):,} (client_id, sku) pairs")

    def generate_identity(self) -> dict:
        if not self.is_trained:
            raise RuntimeError("SimpleIdentitySampler not fitted.")
        idx = np.random.choice(len(self.client_ids), p=self.weights)
        return {"client_id": int(self.client_ids[idx]), "sku": int(self.skus[idx])}

    def save(self, path) -> None:
        joblib.dump({
            "client_ids": self.client_ids,
            "skus":       self.skus,
            "weights":    self.weights,
            "is_trained": self.is_trained,
        }, path)
        print(f"  Sampler saved to {path}")

    def load(self, path) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"No sampler file found at {path}")
        data = joblib.load(path)
        self.client_ids = data["client_ids"]
        self.skus       = data["skus"]
        self.weights    = data["weights"]
        self.is_trained = data["is_trained"]
        print(f"  Sampler loaded from {path}")


if __name__ == "__main__":
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    sampler = SimpleIdentitySampler()
    sampler.fit()
    sampler.save(MODEL_DIR / "identity_sampler.pkl")
    print("\nTest generation:")
    for _ in range(3):
        print(" ", sampler.generate_identity())
