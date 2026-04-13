"""
MarkovSessionGenerator

Bigram Markov chain baseline. Implements GeneratorInterface.

Used as the TSTR-M condition in the evaluation:
    evaluate(synth_M, real_test, seed)  ->  TSTR-M

Transition matrix built from observed action bigrams in the training split.
Item selection is uniform over the observed item pool per action type.
Temporal deltas are sampled from the observed bin distribution.

"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

from config import CLEAN_PARQUET, ITEM_BEARING_EVENTS, TRAIN_CUTOFF
from simulation.generator.session_transformer import (
    ACTION2IDX,
    IDX2ACTION,
    N_ACTIONS,
    EOS_IDX,
    BIN_EDGES,
    N_TEMPORAL_BINS,
    TEMPORAL_MIN_S,
    TEMPORAL_MAX_S,
    bin_to_seconds,
)
from simulation.validity import ValidityLayer


class MarkovSessionGenerator:
    """
    Bigram Markov chain baseline. Implements GeneratorInterface.

    Args:
        action_bigrams: (action_str_a, action_str_b) -> count
        item_pool: action_str -> list[int sku]
        delta_distribution: flat list of observed temporal bin indices
        legal_bigrams: set of (str, str) tuples for ValidityLayer;
            if None, apply_constraints is a no-op
        max_length: max events per session before forced stop
    """

    def __init__(
        self,
        action_bigrams: Dict,
        item_pool: Dict[str, List[int]],
        delta_distribution: List[int],
        legal_bigrams: Optional[Set] = None,
        max_length: int = 50,
    ):
        self.max_length        = max_length
        self.delta_distribution = np.array(delta_distribution, dtype=np.int32)
        self.legal_bigrams     = legal_bigrams
        # Pre-convert item pools to numpy arrays so rng.choice() avoids
        # repeated list->array conversion (critical for large SKU pools)
        self.item_pool: Dict[str, np.ndarray] = {
            k: np.array(v, dtype=np.int64) for k, v in item_pool.items()
        }

        if legal_bigrams is not None:
            self._validity = ValidityLayer(legal_bigrams)
        else:
            self._validity = None

        # --- Build normalized transition matrix [N_ACTIONS, N_ACTIONS] ---
        # Rows = source action, cols = next action (including EOS)
        counts = np.zeros((N_ACTIONS, N_ACTIONS), dtype=np.float64)
        for (a, b), cnt in action_bigrams.items():
            ai = ACTION2IDX.get(a)
            bi = ACTION2IDX.get(b)
            if ai is not None and bi is not None:
                counts[ai, bi] += cnt

        # Allow EOS from any state (fallback row normalization)
        row_sums = counts.sum(axis=1, keepdims=True)
        zero_rows = (row_sums == 0).flatten()
        counts[zero_rows, EOS_IDX] = 1.0
        row_sums = counts.sum(axis=1, keepdims=True)
        self._trans = counts / row_sums   # [N_ACTIONS, N_ACTIONS]

        # Start-action distribution (marginal over source actions)
        start_counts = np.zeros(N_ACTIONS - 1, dtype=np.float64)  # exclude EOS
        for (a, _), cnt in action_bigrams.items():
            ai = ACTION2IDX.get(a)
            if ai is not None and ai != EOS_IDX:
                start_counts[ai] += cnt
        if start_counts.sum() == 0:
            start_counts[:] = 1.0
        self._start_probs = start_counts / start_counts.sum()

        # Precompute delta bin distribution for bulk sampling
        arr = self.delta_distribution
        if len(arr) > 0:
            unique, counts_u = np.unique(arr, return_counts=True)
            self._delta_bins  = unique
            self._delta_probs = counts_u / counts_u.sum()
        else:
            self._delta_bins  = np.arange(N_TEMPORAL_BINS, dtype=np.int32)
            self._delta_probs = np.ones(N_TEMPORAL_BINS) / N_TEMPORAL_BINS
        # Lookup table: bin index -> midpoint seconds
        self._bin_seconds = np.array(
            [bin_to_seconds(i) for i in range(N_TEMPORAL_BINS)], dtype=np.float64
        )

    # GeneratorInterface

    def generate(
        self,
        n_sessions: int,
        seed: int,
        apply_constraints: bool = True,
    ) -> List[List[dict]]:
        """
        Generate n_sessions sessions using the Markov chain.

        Vectorized across all B sessions per step:
          - Bulk delta-bin sampling (one rng call for all sessions per step)
          - Vectorized transition sampling via cumsum trick
          - Per-session cart logic handled in a tight Python inner loop
          - Timestamps accumulated as float offsets; converted to pd.Timestamp
            at emit time using the nanosecond constructor (fast path)

        apply_constraints=True  -> sessions pass through ValidityLayer
                                   (requires legal_bigrams to have been set)
        apply_constraints=False -> raw sessions (for pre-constraint rate)
        """
        rng  = np.random.default_rng(seed)
        B    = n_sessions
        ML   = self.max_length

        # Pre-sample all random numbers upfront to minimise rng call overhead
        # Budget: ML steps × B sessions × 1 main delta + ML × B potential auto-adds
        delta_budget  = 3 * B * (ML + 1)
        flat_deltas   = rng.choice(
            self._delta_bins, size=delta_budget, p=self._delta_probs,
        ).astype(np.int32)
        delta_ptr     = 0

        # [ML, B] uniform draws for transition sampling (cumsum trick)
        trans_u = rng.random((ML, B))

        # Start actions [B]
        action_idxs = rng.choice(
            N_ACTIONS - 1, size=B, p=self._start_probs,
        ).astype(np.int32)

        # Mutable state
        done:     np.ndarray         = np.zeros(B, dtype=bool)
        carts:    List[set]          = [set() for _ in range(B)]
        sessions: List[List[dict]]   = [[]    for _ in range(B)]

        # Float timestamp accumulators; convert via nanosecond constructor
        ts_s       = np.zeros(B, dtype=np.float64)
        base_ns    = pd.Timestamp("2022-10-01").value   # int64 ns since epoch
        NS_PER_S   = 1_000_000_000

        def _ts(b: int) -> pd.Timestamp:
            return pd.Timestamp(int(base_ns + ts_s[b] * NS_PER_S))

        def _sample_sku(action_str: str) -> Optional[int]:
            pool = self.item_pool.get(action_str)
            return int(rng.choice(pool)) if pool is not None and len(pool) else None

        for step in tqdm(range(ML), desc="  Markov gen", unit="step", leave=False, file=__import__("sys").stdout):
            if done.all():
                break

            active = ~done

            # --- Vectorized next-action sampling ---
            trans_rows = self._trans[action_idxs]               # [B, N_ACTIONS]
            cumprobs   = np.cumsum(trans_rows, axis=1)           # [B, N_ACTIONS]
            u          = trans_u[step, :, np.newaxis]            # [B, 1]
            next_idxs  = (u > cumprobs).sum(axis=1)             # [B]
            next_idxs  = np.clip(next_idxs, 0, N_ACTIONS - 1).astype(np.int32)

            # --- Bulk delta consumption for this step ---
            end = delta_ptr + B
            if end > delta_budget:
                # Refill if we somehow exhaust the buffer (extremely unlikely)
                extra = rng.choice(
                    self._delta_bins, size=delta_budget, p=self._delta_probs,
                ).astype(np.int32)
                flat_deltas = np.concatenate([flat_deltas[delta_ptr:], extra])
                delta_ptr   = 0
                end         = B
            batch_secs  = self._bin_seconds[flat_deltas[delta_ptr:end]]  # [B]
            delta_ptr   = end
            ts_s       += np.where(active, batch_secs, 0.0)

            # --- Per-session cart / item logic ---
            for b in range(B):
                if done[b]:
                    continue
                a_idx = int(action_idxs[b])
                a_str = IDX2ACTION[a_idx]

                if a_str == "product_buy":
                    if not carts[b]:
                        atc_sku = _sample_sku("add_to_cart") or _sample_sku(a_str)
                        if atc_sku is not None:
                            # Extra event gets its own delta
                            if delta_ptr >= delta_budget:
                                flat_deltas = rng.choice(
                                    self._delta_bins, size=delta_budget,
                                    p=self._delta_probs,
                                ).astype(np.int32)
                                delta_ptr = 0
                            ts_s[b] += self._bin_seconds[int(flat_deltas[delta_ptr])]
                            delta_ptr += 1
                            sessions[b].append({
                                "client_id":  0,
                                "event_type": "add_to_cart",
                                "sku":        atc_sku,
                                "timestamp":  _ts(b),
                            })
                            carts[b].add(atc_sku)
                    sku = int(rng.choice(list(carts[b]))) if carts[b] else None
                elif a_str == "add_to_cart":
                    sku = _sample_sku(a_str)
                    if sku is not None:
                        carts[b].add(sku)
                elif a_str == "remove_from_cart":
                    if carts[b]:
                        sku = int(rng.choice(list(carts[b])))
                        carts[b].discard(sku)
                    else:
                        sku = _sample_sku(a_str)
                else:
                    sku = None  # page_visit, search_query

                sessions[b].append({
                    "client_id":  0,
                    "event_type": a_str,
                    "sku":        sku,
                    "timestamp":  _ts(b),
                })

            # Update state - mark done where EOS was sampled
            newly_done  = (next_idxs == EOS_IDX) & active
            done       |= newly_done
            action_idxs = np.where(active & ~newly_done, next_idxs, action_idxs)

        if apply_constraints and self._validity is not None:
            sessions = [
                s if (s and self._validity.validate(s)) else []
                for s in sessions
            ]

        return sessions

    # Factory

    @classmethod
    def from_parquet(
        cls,
        parquet_path=None,
        max_length: int = 50,
    ) -> "MarkovSessionGenerator":
        """
        Build directly from events_clean.parquet training split.
        Computes action bigrams, item pools, delta distribution, legal bigrams.
        """
        path = str(parquet_path or CLEAN_PARQUET)
        train_cut = pd.Timestamp(TRAIN_CUTOFF)

        df = (
            pl.read_parquet(
                path,
                columns=["client_id", "timestamp", "event_type", "session_id", "sku"],
            )
            .sort(["session_id", "timestamp"])
        )

        # Training split only - filter in Rust
        session_starts = df.group_by("session_id").agg(
            pl.col("timestamp").min().alias("session_start")
        )
        df = (
            df.join(session_starts, on="session_id")
            .filter(pl.col("session_start") < train_cut)
        )

        # --- Vectorised bigram + item pool computation ---

        # Action bigrams: shift event_type within session, count pairs
        df_bigrams = df.with_columns(
            pl.col("event_type").shift(-1).over("session_id").alias("next_event")
        ).filter(pl.col("next_event").is_not_null())

        bigram_counts = (
            df_bigrams
            .group_by(["event_type", "next_event"])
            .agg(pl.len().alias("cnt"))
        )
        action_bigrams: Dict = {
            (r["event_type"], r["next_event"]): r["cnt"]
            for r in bigram_counts.iter_rows(named=True)
        }
        legal_bigrams: Set = set(action_bigrams.keys())

        # Item pool: unique SKUs per item-bearing event type
        item_pool_df = (
            df.filter(
                pl.col("event_type").is_in(list(ITEM_BEARING_EVENTS)) &
                pl.col("sku").is_not_null()
            )
            .group_by("event_type")
            .agg(pl.col("sku").unique().alias("skus"))
        )
        item_pool: Dict[str, list] = {
            r["event_type"]: [int(s) for s in r["skus"]]
            for r in item_pool_df.iter_rows(named=True)
        }

        # Delta distribution: inter-event gaps in seconds -> bin indices
        delta_s_np = (
            df.with_columns(
                pl.col("timestamp").diff().over("session_id")
                .dt.total_microseconds()
                .truediv(1_000_000)
                .alias("delta_s")
            )
            .filter(pl.col("delta_s").is_not_null() & (pl.col("delta_s") > 0))
            ["delta_s"]
            .to_numpy()
        )
        delta_clipped = np.clip(delta_s_np, TEMPORAL_MIN_S, TEMPORAL_MAX_S)
        delta_distribution: List[int] = np.minimum(
            np.searchsorted(BIN_EDGES[1:], delta_clipped),
            N_TEMPORAL_BINS - 1,
        ).astype(np.int32).tolist()

        return cls(
            action_bigrams=dict(action_bigrams),
            item_pool=item_pool,
            delta_distribution=delta_distribution,
            legal_bigrams=legal_bigrams,
            max_length=max_length,
        )

    # Private

    def _generate_one(self, rng: np.random.Generator) -> List[dict]:
        """
        Generate one session with the given RNG.

        Cart state is tracked so that product_buy is always preceded by
        add_to_cart for the same SKU within the session. If product_buy is
        sampled with an empty cart, an add_to_cart event is prepended for a
        freshly sampled SKU before the buy is emitted.
        """
        # Sample start action (excluding EOS)
        action_idx = int(rng.choice(N_ACTIONS - 1, p=self._start_probs))
        events: List[dict] = []
        cart:   set = set()

        start_dt   = pd.Timestamp("2022-10-01")   # placeholder
        current_dt = start_dt

        def _next_dt() -> pd.Timestamp:
            nonlocal current_dt
            if len(self.delta_distribution) > 0:
                bin_idx = int(rng.choice(self.delta_distribution))
            else:
                bin_idx = int(rng.integers(0, 64))
            current_dt = current_dt + pd.Timedelta(seconds=bin_to_seconds(bin_idx))
            return current_dt

        def _sample_sku(action_str: str) -> int | None:
            pool = self.item_pool.get(action_str)
            return int(rng.choice(pool)) if pool is not None and len(pool) else None

        for _ in range(self.max_length):
            action_str = IDX2ACTION[action_idx]

            if action_str == "product_buy":
                # Ensure add_to_cart precedes buy
                if not cart:
                    # Emit an add_to_cart first
                    atc_sku = _sample_sku("add_to_cart")
                    if atc_sku is None:
                        atc_sku = _sample_sku(action_str)
                    if atc_sku is not None:
                        events.append({
                            "client_id":  0,
                            "event_type": "add_to_cart",
                            "sku":        atc_sku,
                            "timestamp":  _next_dt(),
                        })
                        cart.add(atc_sku)
                # Buy from cart
                sku = int(rng.choice(list(cart))) if cart else None
            elif action_str == "add_to_cart":
                sku = _sample_sku(action_str)
                if sku is not None:
                    cart.add(sku)
            elif action_str == "remove_from_cart":
                if cart:
                    sku = int(rng.choice(list(cart)))
                    cart.discard(sku)
                else:
                    sku = _sample_sku(action_str)
            else:
                sku = None   # page_visit, search_query

            events.append({
                "client_id":  0,
                "event_type": action_str,
                "sku":        sku,
                "timestamp":  _next_dt(),
            })

            # Sample next action
            next_probs = self._trans[action_idx]
            next_idx   = int(rng.choice(N_ACTIONS, p=next_probs))

            if next_idx == EOS_IDX:
                break
            action_idx = next_idx

        return events
