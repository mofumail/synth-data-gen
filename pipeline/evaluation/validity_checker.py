"""
ValidityChecker

Measures violation rates across a list of sessions (for reporting).

Distinct from ValidityLayer (simulation/validity.py):
  - ValidityLayer (simulation): filters/rejects sessions during generation
  - ValidityChecker (evaluation): measures what fraction are violated (for metrics)

Called twice per seed by EvaluationOrchestrator:
    1. On raw generated sessions (before constraints) -> validity_pre
    2. On constrained sessions (after constraints)    -> validity_post

The delta between pre and post shows how much the constraint layer contributes.
"""

from __future__ import annotations

from typing import List, Set, Tuple

from evaluation.data_classes import ValidityResult


class ValidityChecker:
    """
    Args:
        legal_bigrams: set of (action_str, action_str) tuples observed in training.
                       Pass as a set or leave None to skip illegal-transition check.
    """

    def __init__(self, legal_bigrams: Set[Tuple[str, str]] = None):
        self.legal_bigrams = legal_bigrams or set()

    def check_illegal_transitions(self, sessions: List[List[dict]]) -> float:
        """Fraction of sessions containing at least one illegal action bigram."""
        if not sessions or not self.legal_bigrams:
            return 0.0
        violations = 0
        for session in sessions:
            for a, b in zip(session, session[1:]):
                bigram = (a["event_type"], b["event_type"])
                if bigram not in self.legal_bigrams:
                    violations += 1
                    break
        return violations / len(sessions)

    def check_monotonicity(self, sessions: List[List[dict]]) -> float:
        """Fraction of sessions with any timestamp going backwards (delta_t < 0)."""
        if not sessions:
            return 0.0
        violations = 0
        for session in sessions:
            timestamps = [e["timestamp"] for e in session if e.get("timestamp") is not None]
            for t1, t2 in zip(timestamps, timestamps[1:]):
                if t2 <= t1:
                    violations += 1
                    break
        return violations / len(sessions)

    def check_purchase_exposure(self, sessions: List[List[dict]]) -> float:
        """Fraction of sessions where a product_buy has no prior add_to_cart for the same SKU."""
        if not sessions:
            return 0.0
        violations = 0
        for session in sessions:
            cart: set = set()
            has_violation = False
            for ev in session:
                etype = ev["event_type"]
                sku   = ev.get("sku")
                if etype == "add_to_cart" and sku is not None:
                    cart.add(sku)
                elif etype == "remove_from_cart" and sku is not None:
                    cart.discard(sku)
                elif etype == "product_buy":
                    if sku not in cart:
                        has_violation = True
                        break
            if has_violation:
                violations += 1
        return violations / len(sessions)

    def evaluate(self, sessions: List[List[dict]]) -> ValidityResult:
        """Compute all three violation rates and return a ValidityResult."""
        # Filter out empty sessions (validity failures that were dropped)
        non_empty = [s for s in sessions if s]
        if not non_empty:
            return ValidityResult(
                illegal_transition_rate    = 1.0,
                monotonicity_violation_rate= 1.0,
                purchase_exposure_rate     = 1.0,
            )
        return ValidityResult(
            illegal_transition_rate     = self.check_illegal_transitions(non_empty),
            monotonicity_violation_rate = self.check_monotonicity(non_empty),
            purchase_exposure_rate      = self.check_purchase_exposure(non_empty),
        )
