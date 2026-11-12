"""Cost-minimised deferral rule with asymmetric cost matrix.

Replaces the fixed-threshold reject option with an expected-cost minimiser.
This is operationally critical in naval contexts where errors have asymmetric
costs: wrongly calling an enemy ship friendly costs far more than a false alarm.

Cost matrix convention
-----------------------
For a classification problem with K classes, the cost matrix C is K×K where:
- ``C[i, j]`` = cost of predicting class i when the true class is j
- ``C[i, i] = 0`` for all i (correct prediction costs nothing)

For the ship classification problem, we use a cost matrix that penalises
"friendly-to-enemy confusion" (false alarm) much less than
"enemy-to-friendly confusion" (catastrophic miss).

Deferral rule
-------------
The expected cost of making a prediction is:
    E[cost | x] = Σ_i p(y=i | x) · C[predicted_class, i]

The expected cost of deferring to a human is a fixed cost C_defer.

The model defers when:
    E[cost | predict] > C_defer

The threshold that minimises expected cost is the one where these two are equal.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ───────────────────────────────────────────────────────────
# Cost matrix
# ───────────────────────────────────────────────────────────

@dataclass
class CostMatrix:
    """Asymmetric cost matrix for ship classification.

    Convention: C[predicted, true] = cost of predicting `predicted` when
    the true class is `true`. Diagonal entries are 0 (correct prediction
    costs nothing).

    For naval ship classification, the asymmetric costs are:
    - Predicting "friendly" when true class is "enemy" → catastrophic (10×)
    - Predicting "enemy" when true class is "friendly" → false alarm (3×)
    - Predicting any class when true is "unknown/OOD" → moderate (5×)
    - Correct prediction → 0
    """
    n_classes: int
    costs: np.ndarray                 # (K, K) cost matrix
    defer_cost: float = 1.0           # cost of deferring to a human

    @classmethod
    def default_naval(cls, n_classes: int = 6) -> "CostMatrix":
        """Create the default naval cost matrix.

        Class indices follow SHIP_CLASSES:
            0=cargo, 1=tanker, 2=passenger, 3=fishing, 4=naval, 5=small_craft

        Treating "naval" (4) as the high-stakes class:
        - Predicting naval when it's actually a civilian cargo/tanker → 3 (false alarm)
        - Predicting civilian when it's actually naval → 10 (catastrophic miss)
        - All other confusions → 1 (standard misclassification cost)
        """
        costs = np.ones((n_classes, n_classes), dtype=np.float64)
        np.fill_diagonal(costs, 0.0)
        # Asymmetric naval misclassification
        if n_classes > 4:
            # False alarm: predict naval (4) when actually cargo (0) or tanker (1)
            costs[4, 0] = 3.0
            costs[4, 1] = 3.0
            # Catastrophic miss: predict cargo (0) or tanker (1) when actually naval (4)
            costs[0, 4] = 10.0
            costs[1, 4] = 10.0
        return cls(n_classes=n_classes, costs=costs, defer_cost=1.0)

    @classmethod
    def custom(cls, costs: np.ndarray, defer_cost: float = 1.0) -> "CostMatrix":
        """Create a custom cost matrix."""
        n = costs.shape[0]
        assert costs.shape == (n, n), "Cost matrix must be square"
        return cls(n_classes=n, costs=costs.copy(), defer_cost=defer_cost)

    def cost_of(self, predicted: int, true: int) -> float:
        """Get the cost of predicting `predicted` when true class is `true`."""
        return float(self.costs[predicted, true])

    def expected_cost(self, probs: np.ndarray, predicted_class: int) -> float:
        """Expected cost of predicting `predicted_class` given class probabilities.

        E[cost | predict=predicted_class] = Σ_i p(y=i) · C[predicted_class, i]
        """
        return float(np.sum(probs * self.costs[predicted_class]))


# ───────────────────────────────────────────────────────────
# Cost-minimised deferral rule
# ───────────────────────────────────────────────────────────

@dataclass
class DeferredDecision:
    """Result of evaluating the deferral rule on a single sample."""
    predicted_class: Optional[int]    # None if deferred
    confidence: float                  # max class probability
    expected_cost_predict: float       # expected cost of making a prediction
    expected_cost_defer: float         # cost of deferring to a human
    deferred: bool                     # True if deferred to human
    top_2_classes: tuple[int, int]     # top-2 predicted class indices
    top_2_probs: tuple[float, float]   # top-2 class probabilities


class CostMinimisingDeferral:
    """Deferral rule based on expected cost minimisation.

    The rule is: defer to a human supervisor when the expected cost of
    making a prediction exceeds the cost of deferral.

        E[cost | predict] = min_k Σ_i p(y=i | x) · C[k, i]
        defer if E[cost | predict] > C_defer

    This is operationally superior to a fixed threshold because it accounts
    for the asymmetric costs of different types of errors.
    """

    def __init__(self, cost_matrix: CostMatrix):
        self.cost_matrix = cost_matrix

    def evaluate(self, probs: np.ndarray) -> DeferredDecision:
        """Evaluate the deferral rule for a single sample.

        Parameters
        ----------
        probs : np.ndarray
            Class probability vector (K,).
        """
        probs = np.asarray(probs, dtype=np.float64).ravel()
        n_classes = len(probs)

        # Compute expected cost for each possible prediction
        expected_costs = np.array([
            self.cost_matrix.expected_cost(probs, k)
            for k in range(n_classes)
        ])

        # Best prediction = lowest expected cost
        best_class = int(np.argmin(expected_costs))
        best_cost = float(expected_costs[best_class])

        # Top-2 classes by probability
        top2_idx = np.argsort(-probs)[:2]
        top_2_classes = (int(top2_idx[0]), int(top2_idx[1]))
        top_2_probs = (float(probs[top2_idx[0]]), float(probs[top2_idx[1]]))

        # Defer if expected cost of predicting > defer cost
        defer = best_cost > self.cost_matrix.defer_cost

        return DeferredDecision(
            predicted_class=None if defer else best_class,
            confidence=float(probs.max()),
            expected_cost_predict=best_cost,
            expected_cost_defer=self.cost_matrix.defer_cost,
            deferred=defer,
            top_2_classes=top_2_classes,
            top_2_probs=top_2_probs,
        )

    def evaluate_batch(self, probs_batch: np.ndarray) -> list[DeferredDecision]:
        """Evaluate the deferral rule for a batch of samples."""
        return [self.evaluate(p) for p in probs_batch]

    def optimal_threshold(self, probs_samples: np.ndarray) -> float:
        """Find the uncertainty threshold that minimises expected total cost.

        Sweeps over uncertainty thresholds and computes the total expected cost
        (deferral cost + prediction cost) for each. Returns the optimal threshold.

        This is the threshold a DRDO operator should use for the cost matrix.
        """
        probs_samples = np.asarray(probs_samples)
        # Use max probability as the "uncertainty" score (1 - max_prob)
        uncertainties = 1.0 - probs_samples.max(axis=-1)

        # Sweep thresholds
        candidates = np.linspace(0.0, 1.0, 50)
        best_total_cost = float("inf")
        best_threshold = 0.5

        for threshold in candidates:
            retained_mask = uncertainties <= threshold
            if not retained_mask.any():
                continue
            n_deferred = int((~retained_mask).sum())
            n_retained = int(retained_mask.sum())

            # Cost of deferrals
            defer_total = n_deferred * self.cost_matrix.defer_cost

            # Cost of predictions on retained samples
            predict_total = 0.0
            for i, probs in enumerate(probs_samples):
                if not retained_mask[i]:
                    continue
                costs = np.array([
                    self.cost_matrix.expected_cost(probs, k)
                    for k in range(len(probs))
                ])
                predict_total += float(costs.min())

            total = defer_total + predict_total
            if total < best_total_cost:
                best_total_cost = total
                best_threshold = float(threshold)

        return best_threshold
