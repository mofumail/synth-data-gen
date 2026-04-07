"""
ReferenceProfiler + ReferenceStore

ReferenceProfiler runs once before the seed loop and populates a ReferenceStore.
ReferenceStore is the single home for everything precomputed from the real training
split. All evaluators read from it; nothing writes to it during the seed loop.

Also provides RealData (train/val/test session container) and RealDataLoader.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import polars as pl

from config import (
    CLEAN_PARQUET, TEST_PARQUET, ITEM_BEARING_EVENTS, TRAIN_CUTOFF, VAL_CUTOFF, VOCAB_K,
)


# Data containers

@dataclass
class DistributionProfile:
    action_frequencies: Dict[str, float]    # action_type -> fraction
    session_length_counts: Dict[int, int]   # length -> count
    temporal_deltas_sample: List[float] = field(default_factory=list)  # up to 50k inter-event deltas (seconds)
    conversion_rate: float = 0.0            # fraction of sessions with ≥1 product_buy
    cart_abandonment_rate: float = 0.0      # fraction with add_to_cart but no product_buy


@dataclass
class BiasProfile:
    item_coverage: float
    popularity_distribution: Dict[int, float]   # sku -> fraction (top-1000)


@dataclass
class RealData:
    """
    Container for pre-loaded session splits.

    train_split  : sessions with start < TRAIN_CUTOFF, used to train the generator
                   and the reference profiler.
    val_split    : sessions TRAIN_CUTOFF <= start < VAL_CUTOFF, used for all
                   ablation comparisons and development TSTR/TRTR runs.
    test_split   : sessions with start >= VAL_CUTOFF. LOCKED. Only used in the
                   final evaluation pass (evaluate.py --final). Never use this
                   during model selection or ablation comparisons.
    """
    train_split: List[List[dict]]
    val_split:   List[List[dict]]
    test_split:  List[List[dict]]


@dataclass
class ReferenceStore:
    """
    Single artifact for all precomputed reference values.
    Populated by ReferenceProfiler.profile(); read by all evaluators.
    diversity_baseline: mean pairwise Jaccard on real train split (item-bearing only).
    """
    action_bigrams:     Dict        # str(tuple) -> count  (JSON-safe keys)
    item_bigrams:       Dict        # str(tuple) -> count  (item-bearing only)
    diversity_baseline: float
    distributions:      DistributionProfile
    bias:               BiasProfile
    legal_bigrams:      List        # list of [str, str] pairs

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        print(f"  ReferenceStore saved -> {path}")

    @staticmethod
    def load(path) -> "ReferenceStore":
        rs = joblib.load(path)
        print(f"  ReferenceStore loaded ← {path}")
        return rs

    def legal_bigrams_set(self) -> set:
        """Return legal_bigrams as a set of (str, str) tuples."""
        return {(a, b) for a, b in self.legal_bigrams}


# RealDataLoader

class RealDataLoader:
    """
    Loads sessions from events_clean.parquet into a RealData object.
    Sessions are lists of event dicts with keys: client_id, timestamp,
    event_type, sku (int or None).
    """

    @staticmethod
    def load(
        parquet_path=None,
        test_parquet_path=None,
        max_train_sessions: Optional[int] = None,
        max_val_sessions:   Optional[int] = None,
        max_test_sessions:  Optional[int] = None,
        min_length: int = 2,
        include_test: bool = False,
    ) -> RealData:
        """
        Load train and val splits from events_clean.parquet.

        The test split (events_test.parquet) is LOCKED and only loaded when
        include_test=True is explicitly passed. Do not set this flag until all
        model and architecture decisions are final.
        """
        # Train + val come from events_clean.parquet (no test data)
        path = str(parquet_path or CLEAN_PARQUET)
        df = pl.read_parquet(
            path,
            columns=["client_id", "timestamp", "event_type", "session_id", "sku"],
        ).sort(["session_id", "timestamp"])

        train_cut = pd.Timestamp(TRAIN_CUTOFF)
        val_cut   = pd.Timestamp(VAL_CUTOFF)

        # Session-start timestamps (one per session) - stays in Rust
        session_starts = (
            df.group_by("session_id")
            .agg(pl.col("timestamp").min().alias("session_start"))
        )
        df = df.join(session_starts, on="session_id")

        def build_sessions(source_df, filter_expr, max_n: Optional[int]) -> List[List[dict]]:
            sessions_df = (
                source_df.filter(filter_expr)
                .group_by("session_id", maintain_order=True)
                .agg(
                    pl.struct(["client_id", "event_type", "sku", "timestamp"])
                    .alias("events"),
                    pl.len().alias("n"),
                )
                .filter(pl.col("n") >= min_length)
            )
            if max_n is not None:
                sessions_df = sessions_df.sample(n=min(max_n, len(sessions_df)), seed=42)

            out = []
            for row in sessions_df["events"].to_list():
                session = []
                for e in row:
                    sku = e["sku"]
                    session.append({
                        "client_id":  int(e["client_id"]),
                        "event_type": e["event_type"],
                        "sku":        int(sku) if sku is not None else None,
                        "timestamp":  pd.Timestamp(e["timestamp"]),
                    })
                out.append(session)
            return out

        print("  Building train sessions ...")
        train = build_sessions(df, pl.col("session_start") < train_cut, max_train_sessions)
        print(f"    {len(train):,} train sessions")

        print("  Building val sessions ...")
        val = build_sessions(
            df,
            (pl.col("session_start") >= train_cut) & (pl.col("session_start") < val_cut),
            max_val_sessions,
        )
        print(f"    {len(val):,} val sessions")

        # Test split - only loaded when explicitly requested
        if include_test:
            print("  Building test sessions (LOCKED - final evaluation only) ...")
            test_path = str(test_parquet_path or TEST_PARQUET)
            df_test = pl.read_parquet(
                test_path,
                columns=["client_id", "timestamp", "event_type", "session_id", "sku"],
            ).sort(["session_id", "timestamp"])
            df_test = df_test.join(
                df_test.group_by("session_id").agg(pl.col("timestamp").min().alias("session_start")),
                on="session_id",
            )
            test = build_sessions(df_test, pl.lit(True), max_test_sessions)
            print(f"    {len(test):,} test sessions")
        else:
            print("  Test split not loaded (pass include_test=True for final evaluation only).")
            test = []

        return RealData(train_split=train, val_split=val, test_split=test)


# ReferenceProfiler

class ReferenceProfiler:
    """
    Profiles the real training split to produce a ReferenceStore.
    profile() is the public entry point.
    """

    DIVERSITY_SAMPLES = 500   # max sessions to sample for Jaccard computation

    def profile(self, sessions: List[List[dict]]) -> ReferenceStore:
        """Run all profiling steps and return a populated ReferenceStore."""
        print(f"  Profiling {len(sessions):,} training sessions ...")
        return ReferenceStore(
            action_bigrams     = self.compute_action_bigrams(sessions),
            item_bigrams       = self.compute_item_bigrams(sessions),
            diversity_baseline = self.compute_sample_diversity(sessions),
            distributions      = self.compute_distributions(sessions),
            bias               = self.compute_bias(sessions),
            legal_bigrams      = self._compute_legal_bigrams(sessions),
        )

    def compute_action_bigrams(self, sessions: List[List[dict]]) -> Dict:
        counts: Dict = defaultdict(int)
        for session in sessions:
            for a, b in zip(session, session[1:]):
                key = str((a["event_type"], b["event_type"]))
                counts[key] += 1
        return dict(counts)

    def compute_item_bigrams(self, sessions: List[List[dict]]) -> Dict:
        """Item-bearing events only."""
        counts: Dict = defaultdict(int)
        for session in sessions:
            item_evs = [e for e in session if e.get("sku") is not None]
            for a, b in zip(item_evs, item_evs[1:]):
                key = str((a["sku"], b["sku"]))
                counts[key] += 1
        return dict(counts)

    def compute_sample_diversity(
        self,
        sessions: List[List[dict]],
        max_samples: int = None,
    ) -> float:
        """Mean pairwise Jaccard distance on item sets (item-bearing only)."""
        max_s = max_samples or self.DIVERSITY_SAMPLES
        item_sets = []
        for session in sessions:
            s = {e["sku"] for e in session if e.get("sku") is not None}
            if s:
                item_sets.append(s)

        if len(item_sets) < 2:
            return 0.0

        if len(item_sets) > max_s:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(item_sets), max_s, replace=False)
            item_sets = [item_sets[i] for i in idx]

        total, count = 0.0, 0
        n = len(item_sets)
        for i in range(n):
            for j in range(i + 1, n):
                inter = len(item_sets[i] & item_sets[j])
                union = len(item_sets[i] | item_sets[j])
                if union > 0:
                    total += 1.0 - inter / union
                    count += 1

        return total / count if count > 0 else 0.0

    def compute_distributions(self, sessions: List[List[dict]]) -> DistributionProfile:
        action_counts: Dict = defaultdict(int)
        length_counts: Dict = defaultdict(int)
        total_events = 0
        all_deltas: List[float] = []
        n_convert = 0
        n_abandon = 0

        for session in sessions:
            length_counts[len(session)] += 1
            event_types = set()
            for ev in session:
                action_counts[ev["event_type"]] += 1
                total_events += 1
                event_types.add(ev["event_type"])
            for a, b in zip(session, session[1:]):
                delta = (b["timestamp"] - a["timestamp"]).total_seconds()
                if delta >= 0:
                    all_deltas.append(delta)
            has_buy  = "product_buy"  in event_types
            has_cart = "add_to_cart"  in event_types
            if has_buy:
                n_convert += 1
            elif has_cart:
                n_abandon += 1

        if len(all_deltas) > 50_000:
            rng = np.random.default_rng(42)
            all_deltas = list(rng.choice(all_deltas, 50_000, replace=False))

        n_sessions = len(sessions)
        action_freq = {
            k: v / total_events for k, v in action_counts.items()
        } if total_events > 0 else {}

        return DistributionProfile(
            action_frequencies=action_freq,
            session_length_counts=dict(length_counts),
            temporal_deltas_sample=all_deltas,
            conversion_rate=n_convert / n_sessions if n_sessions else 0.0,
            cart_abandonment_rate=n_abandon / n_sessions if n_sessions else 0.0,
        )

    def compute_bias(self, sessions: List[List[dict]]) -> BiasProfile:
        sku_counts: Dict = defaultdict(int)
        for session in sessions:
            for ev in session:
                sku = ev.get("sku")
                if sku is not None:
                    sku_counts[sku] += 1

        n_unique = len(sku_counts)
        coverage = n_unique / VOCAB_K if VOCAB_K > 0 else 0.0

        total = sum(sku_counts.values())
        # Store only top-1000 for popularity distribution (JSON/memory efficiency)
        top_1000 = sorted(sku_counts.items(), key=lambda x: -x[1])[:1000]
        pop_dist  = {int(k): v / total for k, v in top_1000} if total > 0 else {}

        return BiasProfile(item_coverage=coverage, popularity_distribution=pop_dist)

    def _compute_legal_bigrams(self, sessions: List[List[dict]]) -> List:
        seen = set()
        for session in sessions:
            for a, b in zip(session, session[1:]):
                seen.add((a["event_type"], b["event_type"]))
        return [list(pair) for pair in seen]
