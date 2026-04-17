"""
EvaluationOrchestrator

Coordinates the full evaluation run across num_seeds seeds.
Aggregates mean ± std across seeds and computes Pearson correlation between
fidelity metrics and downstream utility as a secondary analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

from config import OUTPUT_DIR
from evaluation.data_classes import (
    AggregatedResult, BiasResult, CorrelationResult,
    FidelityResult, SeedResult, UtilityResult, ValidityResult,
)
from evaluation.fidelity import FidelityEvaluator
from evaluation.validity_checker import ValidityChecker
from evaluation.downstream import DownstreamEvaluator
from evaluation.reference import ReferenceStore

SYNTH_DIR = OUTPUT_DIR / "synthetic"


class EvaluationOrchestrator:

    def __init__(
        self,
        num_seeds: int = 5,
        base_seed: int = 42,
        k: int = 10,
        gru4rec_save_dir: str = None,
        model_tag: str = "",
    ):
        self.num_seeds        = num_seeds
        self.base_seed        = base_seed
        self.k                = k
        self.gru4rec_save_dir = gru4rec_save_dir
        self.model_tag        = model_tag

    # Public entry point

    def evaluate(
        self,
        primary_gen,                    # GeneratorInterface
        baseline_gen,                   # GeneratorInterface
        real_data,                      # RealData
        ref_store:     ReferenceStore,  # profiled from train_split
        val_ref_store: ReferenceStore,  # profiled from val_split (generalization check)
        n_sessions: int = 1000,
        use_test_split: bool = False,
    ) -> AggregatedResult:
        """
        Run evaluation across num_seeds seeds and return aggregated results.

        Args:
            primary_gen: GeneratorInterface - transformer-based generator (TSTR-T)
            baseline_gen: GeneratorInterface - Markov baseline (TSTR-M)
            real_data: RealData
            ref_store: ReferenceStore - precomputed from train_split
            val_ref_store: ReferenceStore - precomputed from val_split; used to compute
                fidelity_val, which tests whether the generator generalises
                beyond the training period (generalization / anti-memorization check)
            n_sessions: number of synthetic sessions to generate per seed
            use_test_split: if True, evaluate downstream against real_data.test_split
                (final run only). Default uses real_data.val_split.
        """
        downstream_split = real_data.test_split if use_test_split else real_data.val_split
        split_label      = "TEST (final)" if use_test_split else "VAL (development)"
        print(f"\n=== EvaluationOrchestrator: {self.num_seeds} seeds, {n_sessions} sessions each ===")
        print(f"    Downstream split: {split_label} ({len(downstream_split):,} sessions)")
        seed_results: List[SeedResult] = []

        for i in range(self.num_seeds):
            seed = self.base_seed + i
            print(f"\n--- Seed {seed} ({i+1}/{self.num_seeds}) ---")
            # Model reloads lazily on first generate() call (no action needed here)

            # Reset per-user synthetic history so each seed starts fresh
            if hasattr(primary_gen, "reset_registry"):
                primary_gen.reset_registry()
            result = self._run_seed(
                primary_gen, baseline_gen, real_data,
                ref_store, val_ref_store,
                seed, n_sessions, downstream_split,
            )
            seed_results.append(result)

        correlations = self.compute_fidelity_utility_correlation(seed_results)
        aggregated   = self._aggregate(seed_results, correlations)
        return aggregated

    # Helpers

    @staticmethod
    def _save_sessions(sessions: list, path: Path) -> None:
        """Flatten List[List[dict]] -> parquet with a session_id column."""
        rows = []
        for sid, session in enumerate(sessions):
            for ev in session:
                rows.append({**ev, "session_id": sid})
        if rows:
            pd.DataFrame(rows).to_parquet(path, index=False)
            non_empty = sum(1 for s in sessions if s)
            print(f"  Saved {non_empty:,} sessions ({len(rows):,} events) -> {path}", flush=True)

    # Per-seed evaluation

    def _run_seed(
        self,
        primary_gen,
        baseline_gen,
        real_data,
        ref_store:     ReferenceStore,
        val_ref_store: ReferenceStore,
        seed: int,
        n_sessions: int,
        downstream_split: list,
    ) -> SeedResult:
        """
        Full evaluation for one seed:
            1. Generate synthetic sessions (transformer + markov)
            2. Fidelity vs train + fidelity vs val (on transformer sessions)
            3. Validity pre/post (on transformer sessions)
            4. TSTR-T, TSTR-M, TRTR utility
        """
        fid_ev    = FidelityEvaluator()
        val_chk   = ValidityChecker(legal_bigrams=ref_store.legal_bigrams_set())
        down_ev   = DownstreamEvaluator(k=self.k, save_dir=self.gru4rec_save_dir, model_tag=self.model_tag)

        # --- Generate sessions ---
        print(f"  Generating {n_sessions} transformer sessions ...")
        synth_T_raw_all  = primary_gen.generate(n_sessions=n_sessions, seed=seed, apply_constraints=False)
        if hasattr(primary_gen, "reset_registry"):
            primary_gen.reset_registry()   # clear invalid sessions before constrained pass
        synth_T_post_all = primary_gen.generate(n_sessions=n_sessions, seed=seed, apply_constraints=True)

        print(f"  Generating {n_sessions} Markov sessions ...")
        synth_M_all = baseline_gen.generate(n_sessions=n_sessions, seed=seed, apply_constraints=True)

        # Filter empty-session placeholders before fidelity / TSTR / save.
        # ValidityLayer leaves [] placeholders when a session fails constraints;
        # feeding those downstream shrinks the effective training set and
        # desyncs the matched-N coverage reference from the actual synth count.
        # Validity metrics themselves still receive the raw (empties-included)
        # lists — ValidityChecker filters internally to report per-session rates.
        synth_T_post = [s for s in synth_T_post_all if s]
        synth_M      = [s for s in synth_M_all      if s]
        print(f"  synth_T_post: {len(synth_T_post):,}/{n_sessions:,} non-empty "
              f"({100*len(synth_T_post)/n_sessions:.1f}%)")
        print(f"  synth_M     : {len(synth_M):,}/{n_sessions:,} non-empty "
              f"({100*len(synth_M)/n_sessions:.1f}%)")

        # --- Save synthetic datasets for offline exploration ---
        SYNTH_DIR.mkdir(parents=True, exist_ok=True)
        self._save_sessions(synth_T_post, SYNTH_DIR / f"{self.model_tag}-seed{seed}-{n_sessions}.parquet")
        self._save_sessions(synth_M,      SYNTH_DIR / f"markov-seed{seed}-{n_sessions}.parquet")

        # --- Delete generator model from GPU before TSTR training ---
        # GRU4Rec needs GPU memory; free it after generation is done.
        if hasattr(primary_gen, "_model") and primary_gen._model is not None:
            if hasattr(primary_gen._model, "cpu"):
                primary_gen._model.cpu()
            del primary_gen._model
            primary_gen._model = None
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        print(f"  GPU after transformer cleanup: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated", flush=True)

        # --- Fidelity (primary generator, constrained) ---
        # matched_source must match the split ref_store was profiled from,
        # otherwise matched-N coverage / Gini compares against the wrong pool.
        print("  Computing fidelity vs train ...")
        fidelity_train = fid_ev.evaluate(
            real_data, synth_T_post, ref_store,
            matched_source=real_data.train_split,
        )
        print("  Computing fidelity vs val (generalization check) ...")
        fidelity_val = fid_ev.evaluate(
            real_data, synth_T_post, val_ref_store,
            matched_source=real_data.val_split,
        )

        # --- Fidelity (markov baseline) ---
        print("  Computing Markov fidelity vs train ...")
        fidelity_markov_train = fid_ev.evaluate(
            real_data, synth_M, ref_store,
            matched_source=real_data.train_split,
        )
        print("  Computing Markov fidelity vs val ...")
        fidelity_markov_val = fid_ev.evaluate(
            real_data, synth_M, val_ref_store,
            matched_source=real_data.val_split,
        )

        # --- Validity ---
        # Raw (empties-included) lists: ValidityChecker.evaluate filters
        # internally and reports per-session rates, which is what we want.
        print("  Computing validity ...")
        validity_pre  = val_chk.evaluate(synth_T_raw_all)
        validity_post = val_chk.evaluate(synth_T_post_all)

        # --- Downstream utility ---
        # Sample exactly n_sessions real sessions for TRTR so training set size is
        # identical across TSTR-T, TSTR-M, and TRTR - removing the sample-size confound.
        n_real = min(n_sessions, len(real_data.train_split))
        if n_real < n_sessions:
            print(f"  WARNING: train_split has only {len(real_data.train_split):,} sessions; "
                  f"TRTR will train on {n_real:,} (< n_sessions={n_sessions:,}). "
                  f"Remove --max-train or increase it to avoid this.")
        rng_sample = np.random.default_rng(seed)
        idxs = rng_sample.choice(len(real_data.train_split), size=n_real, replace=False)
        real_train_sample = [real_data.train_split[i] for i in idxs]
        self._save_sessions(real_train_sample, SYNTH_DIR / f"real_train-seed{seed}-{n_real}.parquet")

        print(f"  Computing downstream utility (n_sessions={n_sessions}, TRTR sample={n_real}) ...")
        # Shared-vocab OOV filter uses the *intersection* of all three conditions'
        # train vocabs, not TRTR's. Using TRTR as reference systematically dings
        # TSTR when a gt item sits in TRTR ∩ real_test but not in TSTR's vocab —
        # that pair auto-misses for TSTR but counts as a valid hit for TRTR.
        util_T, util_M, util_RR = down_ev.evaluate_all_shared_vocab(
            conditions=[
                ("TSTR-T", synth_T_post),
                ("TSTR-M", synth_M),
                ("TRTR",   real_train_sample),
            ],
            real_test      = downstream_split,
            seed           = seed,
            reference_cond = None,
        )

        # --- Fidelity of TRTR sample vs reference distributions ---
        print("  Computing TRTR fidelity vs train ...")
        fidelity_trtr_train = fid_ev.evaluate(
            real_data, real_train_sample, ref_store,
            matched_source=real_data.train_split,
        )
        print("  Computing TRTR fidelity vs val ...")
        fidelity_trtr_val = fid_ev.evaluate(
            real_data, real_train_sample, val_ref_store,
            matched_source=real_data.val_split,
        )

        result = SeedResult(
            seed                  = seed,
            fidelity_train        = fidelity_train,
            fidelity_val          = fidelity_val,
            fidelity_markov_train = fidelity_markov_train,
            fidelity_markov_val   = fidelity_markov_val,
            fidelity_trtr_train   = fidelity_trtr_train,
            fidelity_trtr_val     = fidelity_trtr_val,
            validity_pre          = validity_pre,
            validity_post         = validity_post,
            utility               = [util_T, util_M, util_RR],
        )

        print(f"  Results | fid_train={fidelity_train.jsd_action + fidelity_train.ks_session_length + fidelity_train.l1_action_bigrams:.4f}"
              f"  fid_val={fidelity_val.jsd_action + fidelity_val.ks_session_length + fidelity_val.l1_action_bigrams:.4f}"
              f"  TSTR-T HR@{self.k}={util_T.hr_at_k:.4f}  TSTR-M HR@{self.k}={util_M.hr_at_k:.4f}  TRTR HR@{self.k}={util_RR.hr_at_k:.4f}")

        return result

    # Aggregation

    def _aggregate(
        self,
        seed_results: List[SeedResult],
        correlations: List[CorrelationResult],
    ) -> AggregatedResult:
        """Compute mean ± std across seeds for all metrics."""

        def _agg_fidelity(split_attr: str, fn) -> FidelityResult:
            f = lambda attr: float(fn([getattr(getattr(r, split_attr), attr) for r in seed_results]))
            return FidelityResult(
                jsd_action             = f("jsd_action"),
                ks_session_length      = f("ks_session_length"),
                l1_action_bigrams      = f("l1_action_bigrams"),
                l1_item_bigrams        = f("l1_item_bigrams"),
                sample_diversity       = f("sample_diversity"),
                bias=BiasResult(
                    item_coverage            = float(fn([getattr(r, split_attr).bias.item_coverage            for r in seed_results])),
                    popularity_jsd           = float(fn([getattr(r, split_attr).bias.popularity_jsd           for r in seed_results])),
                    gini_coefficient_delta   = float(fn([getattr(r, split_attr).bias.gini_coefficient_delta   for r in seed_results])),
                    unique_synth_skus        = float(fn([getattr(r, split_attr).bias.unique_synth_skus        for r in seed_results])),
                    unique_real_matched_skus = float(fn([getattr(r, split_attr).bias.unique_real_matched_skus for r in seed_results])),
                ),
                ks_temporal_delta      = f("ks_temporal_delta"),
                conversion_rate_delta  = f("conversion_rate_delta"),
                cart_abandonment_delta = f("cart_abandonment_delta"),
            )

        def _agg_validity(attr: str, fn) -> ValidityResult:
            v = lambda field: float(fn([getattr(getattr(r, attr), field) for r in seed_results]))
            return ValidityResult(
                illegal_transition_rate     = v("illegal_transition_rate"),
                monotonicity_violation_rate = v("monotonicity_violation_rate"),
                purchase_exposure_rate      = v("purchase_exposure_rate"),
            )

        # Utility per condition - indexed by position (TSTR-T, TSTR-M, TRTR)
        conditions = [u.condition for u in seed_results[0].utility]
        utility_mean, utility_std = [], []
        for i, cond in enumerate(conditions):
            hrs  = [r.utility[i].hr_at_k   for r in seed_results]
            ndcgs = [r.utility[i].ndcg_at_k for r in seed_results]
            oovs  = [r.utility[i].oov_rate  for r in seed_results]
            utility_mean.append(UtilityResult(condition=cond, hr_at_k=float(np.mean(hrs)),  ndcg_at_k=float(np.mean(ndcgs)), oov_rate=float(np.mean(oovs))))
            utility_std.append( UtilityResult(condition=cond, hr_at_k=float(np.std(hrs)),   ndcg_at_k=float(np.std(ndcgs)),  oov_rate=float(np.std(oovs))))

        return AggregatedResult(
            fidelity_train_mean        = _agg_fidelity("fidelity_train",        np.mean),
            fidelity_train_std         = _agg_fidelity("fidelity_train",        np.std),
            fidelity_val_mean          = _agg_fidelity("fidelity_val",          np.mean),
            fidelity_val_std           = _agg_fidelity("fidelity_val",          np.std),
            fidelity_markov_train_mean = _agg_fidelity("fidelity_markov_train", np.mean),
            fidelity_markov_train_std  = _agg_fidelity("fidelity_markov_train", np.std),
            fidelity_markov_val_mean   = _agg_fidelity("fidelity_markov_val",   np.mean),
            fidelity_markov_val_std    = _agg_fidelity("fidelity_markov_val",   np.std),
            fidelity_trtr_train_mean   = _agg_fidelity("fidelity_trtr_train",   np.mean),
            fidelity_trtr_train_std    = _agg_fidelity("fidelity_trtr_train",   np.std),
            fidelity_trtr_val_mean     = _agg_fidelity("fidelity_trtr_val",     np.mean),
            fidelity_trtr_val_std      = _agg_fidelity("fidelity_trtr_val",     np.std),
            validity_pre_mean  = _agg_validity("validity_pre",  np.mean),
            validity_pre_std   = _agg_validity("validity_pre",  np.std),
            validity_post_mean = _agg_validity("validity_post", np.mean),
            validity_post_std  = _agg_validity("validity_post", np.std),
            utility_mean       = utility_mean,
            utility_std        = utility_std,
            correlations       = correlations,
            seed_results       = seed_results,
        )

    # Fidelity-utility correlation

    def compute_fidelity_utility_correlation(
        self, seed_results: List[SeedResult]
    ) -> List[CorrelationResult]:
        """
        Pearson r between each fidelity metric and TSTR-T HR@K across seeds.

        Both outcomes are reportable:
            r is significant -> fidelity is a proxy for utility
            r is not         -> direct evidence fidelity alone is insufficient
        """
        if len(seed_results) < 3:
            # Pearson r is meaningless with < 3 points
            return []

        # Use TSTR-T HR@K as the utility proxy
        tstr_t_hr = []
        for r in seed_results:
            for u in r.utility:
                if u.condition == "TSTR-T":
                    tstr_t_hr.append(u.hr_at_k)
                    break

        fidelity_metrics = {
            "jsd_action":            [r.fidelity_train.jsd_action            for r in seed_results],
            "ks_session_length":     [r.fidelity_train.ks_session_length     for r in seed_results],
            "ks_temporal_delta":     [r.fidelity_train.ks_temporal_delta     for r in seed_results],
            "l1_action_bigrams":     [r.fidelity_train.l1_action_bigrams     for r in seed_results],
            "l1_item_bigrams":       [r.fidelity_train.l1_item_bigrams       for r in seed_results],
            "sample_diversity":      [r.fidelity_train.sample_diversity      for r in seed_results],
            "conversion_rate_delta": [r.fidelity_train.conversion_rate_delta for r in seed_results],
        }

        results = []
        for metric_name, scores in fidelity_metrics.items():
            if len(set(scores)) < 2 or len(set(tstr_t_hr)) < 2:
                # Zero variance - correlation undefined
                results.append(CorrelationResult(
                    fidelity_metric=metric_name, correlation=float("nan"), p_value=float("nan")
                ))
                continue
            r_val, p_val = pearsonr(scores, tstr_t_hr)
            results.append(CorrelationResult(
                fidelity_metric=metric_name,
                correlation=float(r_val),
                p_value=float(p_val),
            ))

        return results
