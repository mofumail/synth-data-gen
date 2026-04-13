"""
Evaluation data classes.

SeedResult is the load-bearing data structure for the evaluation pipeline.
It must carry FidelityResult + ValidityResult (pre+post) + UtilityResult per
condition so that compute_fidelity_utility_correlation() has everything it
needs without reaching back into orchestrator state.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List


@dataclass
class BiasResult:
    item_coverage: float          # fraction of vocab seen in generated sessions
    popularity_jsd: float         # JSD between real and generated item popularity
    gini_coefficient_delta: float = 0.0   # |gini(synth) - gini(real)| over top-1000 items


@dataclass
class FidelityResult:
    jsd_action: float             # Jensen-Shannon divergence on action frequencies
    ks_session_length: float      # KS statistic on session length distribution
    l1_action_bigrams: float      # L1 distance on action bigram transition matrix
    l1_item_bigrams: float        # L1 distance on item bigram matrix (item-bearing only)
    sample_diversity: float       # mean pairwise Jaccard (item-bearing only)
    bias: BiasResult
    ks_temporal_delta: float = 0.0        # KS on inter-event delta (seconds) distribution
    conversion_rate_delta: float = 0.0    # |synth - real| fraction of sessions with product_buy
    cart_abandonment_delta: float = 0.0   # |synth - real| fraction with add_to_cart but no buy


@dataclass
class ValidityResult:
    illegal_transition_rate: float      # fraction of sessions with illegal bigrams
    monotonicity_violation_rate: float  # fraction with any delta_t < 0
    purchase_exposure_rate: float       # fraction of buys without prior add_to_cart


@dataclass
class UtilityResult:
    condition: str      # "TSTR-T", "TSTR-M", or "TRTR"
    hr_at_k: float
    ndcg_at_k: float
    oov_rate: float = 0.0   # fraction of test pairs filtered due to OOV ground truth


@dataclass
class SeedResult:
    seed: int
    fidelity_train: FidelityResult        # transformer synthetic vs training distribution
    fidelity_val:   FidelityResult        # transformer synthetic vs val distribution
    fidelity_markov_train: FidelityResult # markov baseline vs training distribution
    fidelity_markov_val:   FidelityResult # markov baseline vs val distribution
    fidelity_trtr_train: FidelityResult   # real train sample vs training distribution
    fidelity_trtr_val:   FidelityResult   # real train sample vs val distribution
    validity_pre: ValidityResult          # before ValidityLayer
    validity_post: ValidityResult         # after ValidityLayer
    utility: List[UtilityResult]          # one per condition: TSTR-T, TSTR-M, TRTR


@dataclass
class CorrelationResult:
    fidelity_metric: str    # which fidelity metric was used
    correlation: float      # Pearson r between fidelity scores and TSTR utility
    p_value: float


@dataclass
class AggregatedResult:
    fidelity_train_mean: FidelityResult        # transformer synthetic vs training distribution
    fidelity_train_std:  FidelityResult
    fidelity_val_mean:   FidelityResult        # transformer synthetic vs val distribution
    fidelity_val_std:    FidelityResult
    fidelity_markov_train_mean: FidelityResult # markov baseline vs training distribution
    fidelity_markov_train_std:  FidelityResult
    fidelity_markov_val_mean:   FidelityResult # markov baseline vs val distribution
    fidelity_markov_val_std:    FidelityResult
    fidelity_trtr_train_mean:   FidelityResult # real train sample vs training distribution
    fidelity_trtr_train_std:    FidelityResult
    fidelity_trtr_val_mean:     FidelityResult # real train sample vs val distribution
    fidelity_trtr_val_std:      FidelityResult
    validity_pre_mean: ValidityResult
    validity_pre_std: ValidityResult
    validity_post_mean: ValidityResult
    validity_post_std: ValidityResult
    utility_mean: List[UtilityResult]
    utility_std: List[UtilityResult]
    correlations: List[CorrelationResult]
    seed_results: List[SeedResult]      # kept for post-hoc analysis
