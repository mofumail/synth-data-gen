"""
ReferenceProfiler + ReferenceStore

ReferenceProfiler runs once before the seed loop and populates a ReferenceStore.
ReferenceStore is the single home for everything precomputed from the real training
split. All evaluators read from it; nothing writes to it during the seed loop.

Also provides RealData (train/val/test session container) and RealDataLoader.
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import polars as pl

from config import (
    CLEAN_PARQUET, TEST_PARQUET, TRAIN_CUTOFF, VAL_CUTOFF, VOCAB_K,
    OUTPUT_DIR,
)


# Module-level worker for multiprocessing (must be top-level, not nested, to be picklable)
def _build_sessions_chunk(args: tuple) -> List[List[dict]]:
    # ts values are already datetime.datetime from Polars to_list() — no pd.Timestamp() needed.
    # datetime arithmetic (b - a).total_seconds() works identically.
    cid_ints, ev_types, skus_col, tss_col = args
    out = []
    for cid_int, evts, skus, tss in zip(cid_ints, ev_types, skus_col, tss_col):
        out.append([
            {
                "client_id":  cid_int,
                "event_type": et,
                "sku":        int(sk) if sk is not None else None,
                "timestamp":  ts,
            }
            for et, sk, ts in zip(evts, skus, tss)
        ])
    return out


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

        The slow steps (full parquet read + Polars group_by agg) are cached as
        compact aggregated parquets (~100MB) keyed on parquet mtime + parameters.
        On cache hit, only the parallel Python dict construction runs (~1-2 min).
        On cache miss the full build runs and writes the cache (~10-15 min, once only).

        Avoids pickling the full List[List[dict]] (50GB+ RAM → OOM).

        The test split is LOCKED; only loaded when include_test=True.
        """
        path      = str(parquet_path or CLEAN_PARQUET)
        mtime     = int(os.path.getmtime(path) * 1000)
        cache_tag = (
            f"{mtime}_ml{min_length}"
            f"_tr{max_train_sessions or 'all'}"
            f"_vl{max_val_sessions or 'all'}"
            f"_test{int(include_test)}"
        )
        cache_dir = OUTPUT_DIR / f"real_sessions_agg_{cache_tag}"

        def agg_to_sessions(agg_df: "pl.DataFrame") -> List[List[dict]]:
            """Convert aggregated Polars DataFrame to List[List[dict]].
            Single-threaded — multiprocessing was removed because pickling 5M dicts
            through IPC pipes doubled peak RAM and caused OOM. Without pd.Timestamp()
            the plain Python loop is fast enough (~15-20s for 5M sessions).
            """
            cids     = agg_df["client_id"].to_list()
            ev_types = agg_df["event_types"].to_list()
            skus_col = agg_df["skus"].to_list()
            tss_col  = agg_df["timestamps"].to_list()
            return _build_sessions_chunk((
                [int(c) for c in cids], ev_types, skus_col, tss_col
            ))

        if cache_dir.exists():
            print(f"  Loading aggregated session cache ({cache_dir.name}) ...")
            train_agg = pl.read_parquet(cache_dir / "train.parquet")
            val_agg   = pl.read_parquet(cache_dir / "val.parquet")
            print("  Building train sessions ...")
            train = agg_to_sessions(train_agg)
            print(f"    {len(train):,} train sessions")
            print("  Building val sessions ...")
            val = agg_to_sessions(val_agg)
            print(f"    {len(val):,} val sessions")
            test = []
            if include_test and (cache_dir / "test.parquet").exists():
                test_agg = pl.read_parquet(cache_dir / "test.parquet")
                print("  Building test sessions ...")
                test = agg_to_sessions(test_agg)
                print(f"    {len(test):,} test sessions")
            return RealData(train_split=train, val_split=val, test_split=test)

        # --- Full build from raw parquet ---
        # Single-pass lazy query: no global sort, no session_starts join.
        # Sort within each group via sort_by() — O(k log k) per session (~7 events)
        # vs global O(N log N) on 199M rows. Drop maintain_order (hash groupby is faster).
        train_cut = pd.Timestamp(TRAIN_CUTOFF)
        val_cut   = pd.Timestamp(VAL_CUTOFF)

        print("  Aggregating sessions from parquet (single pass) ...")
        full_agg = (
            pl.scan_parquet(path)
            .select(["client_id", "timestamp", "event_type", "session_id", "sku"])
            .group_by("session_id")
            .agg(
                pl.col("client_id").first(),
                pl.col("event_type").sort_by("timestamp").alias("event_types"),
                pl.col("sku").sort_by("timestamp").alias("skus"),
                pl.col("timestamp").sort().alias("timestamps"),
                pl.col("timestamp").min().alias("session_start"),
                pl.len().alias("n"),
            )
            .filter(pl.col("n") >= min_length)
            .collect()
        )

        def make_agg(filter_expr, max_n: Optional[int]) -> "pl.DataFrame":
            agg_df = full_agg.filter(filter_expr)
            if max_n is not None:
                agg_df = agg_df.sample(n=min(max_n, len(agg_df)), seed=42)
            return agg_df.drop("session_start", "n")

        print("  Building train sessions ...")
        train_agg = make_agg(pl.col("session_start") < train_cut, max_train_sessions)
        train = agg_to_sessions(train_agg)
        print(f"    {len(train):,} train sessions")

        print("  Building val sessions ...")
        val_agg = make_agg(
            (pl.col("session_start") >= train_cut) & (pl.col("session_start") < val_cut),
            max_val_sessions,
        )
        val = agg_to_sessions(val_agg)
        print(f"    {len(val):,} val sessions")

        test, test_agg = [], None
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
            test_agg = make_agg(df_test, pl.lit(True), max_test_sessions)
            test = agg_to_sessions(test_agg)
            print(f"    {len(test):,} test sessions")
        else:
            print("  Test split not loaded (pass include_test=True for final evaluation only).")

        # Save aggregated parquets (compact, no OOM risk)
        print(f"  Saving aggregated session cache -> {cache_dir.name}/ ...")
        cache_dir.mkdir(parents=True, exist_ok=True)
        train_agg.write_parquet(cache_dir / "train.parquet")
        val_agg.write_parquet(cache_dir / "val.parquet")
        if test_agg is not None:
            test_agg.write_parquet(cache_dir / "test.parquet")

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
