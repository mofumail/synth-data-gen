"""
FidelityEvaluator

Eight metrics comparing generated sessions against the real training distribution.
All reference values come from ReferenceStore (precomputed once, not per seed).

Metrics:
    1. JSD(action distribution)    - Jensen-Shannon divergence on action frequencies
    2. KS(session length)          - Kolmogorov-Smirnov two-sample statistic
    3. L1(action bigram matrix)    - L1 on normalized action transition matrix
    4. L1(item bigram matrix)      - L1 on item transitions (item-bearing only)
    5. sample_diversity            - relative diversity: synth Jaccard / real Jaccard (1.0 = matches real)
    6. KS(temporal delta)          - KS on inter-event delta distributions (seconds)
    7. conversion_rate_delta       - abs difference in fraction of sessions with a product_buy
    8. cart_abandonment_delta      - abs difference in fraction of sessions with add_to_cart but no buy

Plus bias metrics (informational): item_coverage, popularity_jsd.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp

from config import VOCAB_K
from evaluation.data_classes import BiasResult, FidelityResult
from evaluation.reference import ReferenceProfiler, ReferenceStore


def _gini(counts) -> float:
    """Gini coefficient of a frequency array. 0 = uniform, 1 = monopoly."""
    x = np.sort(np.array(list(counts), dtype=np.float64))
    if x.sum() == 0 or len(x) == 0:
        return 0.0
    n   = len(x)
    idx = np.arange(1, n + 1)
    return float((2.0 * (idx * x).sum()) / (n * x.sum()) - (n + 1) / n)


def _l1(ref: dict, synth: dict, restrict_to_ref_keys: bool = False) -> float:
    """L1 distance between two normalized count dicts."""
    keys = sorted(ref.keys() if restrict_to_ref_keys else set(ref) | set(synth))
    p = np.array([ref.get(k, 0)   for k in keys], dtype=np.float64)
    q = np.array([synth.get(k, 0) for k in keys], dtype=np.float64)
    p /= p.sum() + 1e-12
    q /= q.sum() + 1e-12
    return float(np.abs(p - q).sum())


class FidelityEvaluator:

    DIVERSITY_SAMPLES = 500

    def compute_jsd(self, real_dist: dict, synth_dist: dict) -> float:
        """
        Jensen-Shannon divergence between two frequency/count dicts.
        Returns a value in [0, 1] - 0 means identical distributions.
        Note: scipy.jensenshannon returns the square root (distance), so we square it.
        """
        keys = sorted(set(real_dist) | set(synth_dist))
        p = np.array([real_dist.get(k, 0)  for k in keys], dtype=np.float64)
        q = np.array([synth_dist.get(k, 0) for k in keys], dtype=np.float64)
        p /= p.sum() + 1e-12
        q /= q.sum() + 1e-12
        return float(jensenshannon(p, q) ** 2)

    def compute_ks(self, real_lengths: list, synth_lengths: list) -> float:
        """Two-sample KS statistic on session lengths. 0 = identical distributions."""
        if not real_lengths or not synth_lengths:
            return 1.0
        return float(ks_2samp(real_lengths, synth_lengths).statistic)

    def compute_l1_action_bigrams(self, ref: dict, synth: dict) -> float:
        """L1 on normalized action bigram distributions."""
        return _l1(ref, synth)

    def compute_l1_item_bigrams(self, ref: dict, synth: dict) -> float:
        """L1 on item bigrams, restricted to keys observed in ref (avoids UNK explosion)."""
        if not ref:
            return 0.0
        return _l1(ref, synth, restrict_to_ref_keys=True)

    def compute_sample_diversity(self, sessions: list) -> float:
        """Mean pairwise Jaccard distance on item sets (item-bearing only)."""
        return ReferenceProfiler().compute_sample_diversity(
            sessions, max_samples=self.DIVERSITY_SAMPLES
        )

    def compute_ks_temporal_delta(self, ref_deltas: list, synth_sessions: list) -> float:
        """KS statistic on inter-event delta (seconds) distributions."""
        synth_deltas = []
        for session in synth_sessions:
            for a, b in zip(session, session[1:]):
                delta = (b["timestamp"] - a["timestamp"]).total_seconds()
                if delta >= 0:
                    synth_deltas.append(delta)
        if not ref_deltas or not synth_deltas:
            return 1.0
        return float(ks_2samp(ref_deltas, synth_deltas).statistic)

    def compute_conversion_rate(self, synth_sessions: list) -> float:
        """Fraction of sessions containing ≥1 product_buy."""
        if not synth_sessions:
            return 0.0
        n = sum(1 for s in synth_sessions if any(e["event_type"] == "product_buy" for e in s))
        return n / len(synth_sessions)

    def compute_cart_abandonment_rate(self, synth_sessions: list) -> float:
        """Fraction of sessions with add_to_cart but no product_buy."""
        if not synth_sessions:
            return 0.0
        n = sum(
            1 for s in synth_sessions
            if any(e["event_type"] == "add_to_cart" for e in s)
            and not any(e["event_type"] == "product_buy" for e in s)
        )
        return n / len(synth_sessions)

    def compute_bias_metrics(
        self,
        sessions: list,
        ref_store: ReferenceStore,
        matched_real_sample: list | None = None,
    ) -> BiasResult:
        """
        When `matched_real_sample` is given (real sessions subsampled to the same
        N as `sessions`), coverage and Gini use that matched sample as the
        reference — otherwise small-N synth is unfairly penalised on coverage
        (unique_SKUs / VOCAB_K) and Gini (synth distribution full of zeros vs a
        5M-session reference).

        Matched-N semantics:
            item_coverage = unique_synth / unique_real_matched  (1.0 = parity)
            gini_delta    = |gini(synth full pop) - gini(real matched full pop)|
        """
        sku_counts: dict = defaultdict(int)
        for session in sessions:
            for ev in session:
                sku = ev.get("sku")
                if sku is not None:
                    sku_counts[sku] += 1
        synth_unique = len(sku_counts)

        if matched_real_sample is not None:
            real_sku_counts: dict = defaultdict(int)
            for session in matched_real_sample:
                for ev in session:
                    sku = ev.get("sku")
                    if sku is not None:
                        real_sku_counts[sku] += 1
            real_unique = len(real_sku_counts)
            coverage    = (synth_unique / real_unique) if real_unique > 0 else 0.0

            real_total  = sum(real_sku_counts.values()) or 1
            real_gini   = _gini(v / real_total for v in real_sku_counts.values())

            synth_total_all = sum(sku_counts.values()) or 1
            synth_gini      = _gini(v / synth_total_all for v in sku_counts.values())
        else:
            coverage  = synth_unique / VOCAB_K if VOCAB_K > 0 else 0.0
            real_unique = VOCAB_K                 # denominator for the unmatched branch
            ref_pop   = ref_store.bias.popularity_distribution
            real_gini = _gini(ref_pop.values())
            synth_freq  = np.array([sku_counts.get(sku, 0) for sku in ref_pop], dtype=np.float64)
            synth_total = synth_freq.sum() or 1.0
            synth_gini  = _gini(synth_freq / synth_total)

        gini_delta = abs(synth_gini - real_gini)

        # JSD between synth and real popularity distributions (over top-1000 real items).
        # Shape comparison — size-robust, uses full ref_store even when matched.
        ref_pop = ref_store.bias.popularity_distribution
        total     = sum(sku_counts.values()) or 1
        synth_pop = {sku: sku_counts.get(sku, 0) / total for sku in ref_pop}
        pop_jsd   = self.compute_jsd(ref_pop, synth_pop)

        return BiasResult(
            item_coverage            = coverage,
            popularity_jsd           = pop_jsd,
            gini_coefficient_delta   = gini_delta,
            unique_synth_skus        = float(synth_unique),
            unique_real_matched_skus = float(real_unique),
        )

    def evaluate(
        self,
        real_data,
        synth_sessions: list,
        ref_store: ReferenceStore,
        match_n: bool = True,
        matched_source: list | None = None,
    ) -> FidelityResult:
        """
        Compute all eight fidelity metrics + bias against ref_store.

        match_n: if True, coverage and Gini are computed against a random
            subsample of `matched_source` (defaults to real_data.train_split)
            with size = len(synth_sessions). Keeps bias metrics comparable
            when synth N << reference N.
        matched_source: list of real sessions to subsample from. Must match
            the split that `ref_store` was profiled from (pass val_split for a
            val_ref_store, train_split for a train ref_store).
        """
        profiler = ReferenceProfiler()

        matched = None
        if match_n and synth_sessions:
            pool = matched_source if matched_source is not None else real_data.train_split
            k    = min(len(synth_sessions), len(pool))
            rng  = np.random.default_rng(0)
            idx  = rng.choice(len(pool), size=k, replace=False)
            matched = [pool[i] for i in idx]

        # 1. JSD - action distribution
        synth_action_counts = defaultdict(int)
        for session in synth_sessions:
            for ev in session:
                synth_action_counts[ev["event_type"]] += 1
        total = sum(synth_action_counts.values()) or 1
        synth_action_dist = {k: v / total for k, v in synth_action_counts.items()}
        jsd_action = self.compute_jsd(ref_store.distributions.action_frequencies, synth_action_dist)

        # 2. KS - session length
        real_lengths = []
        for length, cnt in ref_store.distributions.session_length_counts.items():
            real_lengths.extend([int(length)] * cnt)
        synth_lengths = [len(s) for s in synth_sessions]
        ks_session_length = self.compute_ks(real_lengths, synth_lengths)

        # 3. L1 - action bigrams
        l1_action = self.compute_l1_action_bigrams(
            ref_store.action_bigrams, profiler.compute_action_bigrams(synth_sessions)
        )

        # 4. L1 - item bigrams
        l1_item = self.compute_l1_item_bigrams(
            ref_store.item_bigrams, profiler.compute_item_bigrams(synth_sessions)
        )

        # 5. Sample diversity - relative to real baseline (synth / real)
        # Raw Jaccard is near 1.0 for any sparse catalog; ratio is meaningful.
        synth_diversity = self.compute_sample_diversity(synth_sessions)
        baseline = ref_store.diversity_baseline or 1.0
        diversity = synth_diversity / baseline

        # 6. KS - temporal delta
        ks_temporal = self.compute_ks_temporal_delta(
            ref_store.distributions.temporal_deltas_sample, synth_sessions
        )

        # 7. Conversion rate delta
        synth_conv = self.compute_conversion_rate(synth_sessions)
        conv_delta = abs(synth_conv - ref_store.distributions.conversion_rate)

        # 8. Cart abandonment delta
        synth_aban = self.compute_cart_abandonment_rate(synth_sessions)
        aban_delta = abs(synth_aban - ref_store.distributions.cart_abandonment_rate)

        return FidelityResult(
            jsd_action             = jsd_action,
            ks_session_length      = ks_session_length,
            l1_action_bigrams      = l1_action,
            l1_item_bigrams        = l1_item,
            sample_diversity       = diversity,
            bias                   = self.compute_bias_metrics(synth_sessions, ref_store, matched_real_sample=matched),
            ks_temporal_delta      = ks_temporal,
            conversion_rate_delta  = conv_delta,
            cart_abandonment_delta = aban_delta,
        )
