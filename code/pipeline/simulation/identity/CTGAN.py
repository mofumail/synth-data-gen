"""
IdentityFactory

Generates (client_id, sku) seed pairs for synthetic session generation using
CTGANSynthesizer. Replaces SimpleIdentitySampler.

The CTGAN learns the joint distribution of (client_id, sku) from real training
interactions. Sampling from it produces more diverse seeds than
frequency-weighted sampling, reducing the popularity halo effect.

Usage:
    PYTHONPATH=. uv run python simulation/identity/CTGAN.py
"""

import os

import joblib
import pandas as pd
from ctgan import CTGAN as CTGANSynthesizer

from config import CLEAN_PARQUET, TRAIN_CUTOFF, ITEM_BEARING_EVENTS, MODEL_DIR


class IdentityFactory:

    def __init__(
        self,
        model_storage=None,
        epochs: int = 30,
        embedding_dim: int = 128,
        generator_dim: tuple = (256, 256),
        discriminator_dim: tuple = (256, 256),
    ):
        self.model_storage     = str(model_storage or MODEL_DIR / "identity_sampler.pkl")
        self.epochs            = epochs
        self.embedding_dim     = embedding_dim
        self.generator_dim     = generator_dim
        self.discriminator_dim = discriminator_dim
        self._model            = None
        self.is_trained        = False

    def fit(self, sample_size: int = 100_000, top_n_items: int = 2_500) -> None:
        print(f"\nFitting IdentityFactory (CTGAN) from {CLEAN_PARQUET}...")
        df = pd.read_parquet(
            CLEAN_PARQUET,
            columns=["client_id", "timestamp", "event_type", "sku"],
        )

        train_cutoff = pd.Timestamp(TRAIN_CUTOFF)
        df = df[
            (df["timestamp"] < train_cutoff)
            & (df["event_type"].isin(ITEM_BEARING_EVENTS))
            & df["sku"].notna()
        ]

        top_items = df["sku"].value_counts().nlargest(top_n_items).index
        df = df[df["sku"].isin(top_items)][["client_id", "sku"]]
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)

        df = df.reset_index(drop=True)
        df["client_id"] = df["client_id"].astype(float)
        df["sku"]       = df["sku"].astype(int).astype(str)   # clean integer strings for discrete encoding

        print(f"  Training on {len(df):,} pairs ({df['sku'].nunique():,} unique items)...")
        self._model = CTGANSynthesizer(
            embedding_dim      = self.embedding_dim,
            generator_dim      = self.generator_dim,
            discriminator_dim  = self.discriminator_dim,
            epochs             = self.epochs,
            verbose            = True,
        )
        self._model.fit(df, discrete_columns=["sku"])
        self.is_trained = True
        self.save(self.model_storage)
        print(f"  IdentityFactory saved to {self.model_storage}")

    def save(self, path=None) -> None:
        joblib.dump(self, path or self.model_storage)

    def load(self, path=None) -> bool:
        path = path or self.model_storage
        if not os.path.exists(path):
            return False
        loaded = joblib.load(path)
        self._model     = loaded._model
        self.is_trained = loaded.is_trained
        print(f"  IdentityFactory loaded from {path}")
        return True

    def generate_identity(self) -> dict:
        if not self.is_trained:
            raise RuntimeError("IdentityFactory not fitted. Run fit() first.")
        row = self._model.sample(1).iloc[0]
        return {
            "client_id": int(round(float(row["client_id"]))),
            "sku":        int(row["sku"]),
        }


if __name__ == "__main__":
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    factory = IdentityFactory()
    if not factory.load():
        factory.fit()
    print("\nTest generation:")
    for _ in range(3):
        print(" ", factory.generate_identity())
