"""
ValidityLayer

Post-generation hard constraint checks applied to a completed session.

Two constraint layers exist in the architecture:
  1. TransitionConstraintMask (PRE-sampling, inside ActionHead) - prevents
     illegal next-action logits before sampling. Lives in the model.
  2. ValidityLayer (POST-generation, here) - checks the full completed
     session as a whole. Catches anything that slipped through.

Both violation rates must be reported: pre-constraint and post-constraint.
"""

from __future__ import annotations
from typing import List


class ValidityLayer:
    def __init__(self, legal_bigrams: set[tuple[str, str]]):
        """
        Args:
            legal_bigrams: set of (action_a, action_b) tuples observed in
                training data. Extracted by ReferenceProfiler.
        """
        self.legal_bigrams = legal_bigrams

    def check_monotonicity(self, session: List[dict]) -> bool:
        """All timestamps must be strictly increasing within the session."""
        timestamps = [e["timestamp"] for e in session]
        return all(t1 < t2 for t1, t2 in zip(timestamps, timestamps[1:]))

    def check_purchase_exposure(self, session: List[dict]) -> bool:
        """Every product_buy must be preceded by add_to_cart for the same sku."""
        cart: set = set()
        for event in session:
            etype = event["event_type"]
            sku   = event.get("sku")
            if etype == "add_to_cart" and sku is not None:
                cart.add(sku)
            elif etype == "remove_from_cart" and sku is not None:
                cart.discard(sku)
            elif etype == "product_buy":
                if sku not in cart:
                    return False
        return True

    def check_illegal_transitions(self, session: List[dict]) -> bool:
        """No (action_a, action_b) bigram outside the legal_bigrams set."""
        for a, b in zip(session, session[1:]):
            bigram = (a["event_type"], b["event_type"])
            if bigram not in self.legal_bigrams:
                return False
        return True

    def validate(self, session: List[dict]) -> bool:
        """Return True only if all three checks pass."""
        if not session:
            return False
        return (
            self.check_monotonicity(session)
            and self.check_purchase_exposure(session)
            and self.check_illegal_transitions(session)
        )
