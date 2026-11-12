"""Extended calibration and robustness metrics.

Adds the metrics required for rigorous UQ evaluation that were missing from
the base metrics module:

- **FPR95** — false positive rate at 95% true positive rate (OOD detection)
- **AUROC(uncertainty vs. errors)** — does uncertainty predict mistakes?
- **Risk-coverage curve** — accuracy on retained cases vs. deferral rate
- **Reliability diagram** — binned confidence vs. accuracy for visualisation
- **Brier score decomposition** — reliability, resolution, uncertainty
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ───────────────────────────────────────────────────────────
# FPR95 — false positive rate at 95% TPR (OOD detection)
# ───────────────────────────────────────────────────────────

def fpr95(
    in_dist_uncertainty: np.ndarray,
    out_dist_uncertainty: np.ndarray,
) -> float:
    """False positive rate at 95% true positive rate for OOD detection.

    A lower FPR95 is better — it means that 95% of OOD inputs are correctly
    flagged while only a small fraction of in-distribution inputs are wrongly
    rejected.

    Parameters
    ----------
    in_dist_uncertainty : np.ndarray
        Uncertainty scores for in-distribution samples (should be low).
    out_dist_uncertainty : np.ndarray
        Uncertainty scores for OOD samples (should be high).
    """
    in_dist = np.asarray(in_dist_uncertainty).ravel()
    out_dist = np.asarray(out_dist_uncertainty).ravel()

    if len(in_dist) == 0 or len(out_dist) == 0:
        return float("nan")

    # Combine and label
    labels = np.concatenate([np.zeros(len(in_dist)), np.ones(len(out_dist))])
    scores = np.concatenate([in_dist, out_dist])

    # Sort by descending score — high score = more likely OOD
    order = np.argsort(-scores)
    sorted_labels = labels[order]

    # Find threshold at TPR = 0.95
    n_pos = len(out_dist)
    n_neg = len(in_dist)
    tpr_target = 0.95
    cum_pos = np.cumsum(sorted_labels)
    cum_neg = np.arange(1, len(sorted_labels) + 1) - cum_pos

    # TPR = cum_pos / n_pos, FPR = cum_neg / n_neg
    tprs = cum_pos / n_pos
    fprs = cum_neg / n_neg

    # Find the FPR at the point where TPR first reaches 0.95
    idx = np.searchsorted(tprs, tpr_target)
    if idx >= len(fprs):
        return float(fprs[-1])
    return float(fprs[idx])


# ───────────────────────────────────────────────────────────
# AUROC(uncertainty vs. errors) — does uncertainty predict mistakes?
# ───────────────────────────────────────────────────────────

def uncertainty_error_auroc(
    uncertainties: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
) -> float:
    """AUROC for using uncertainty to predict classification errors.

    A high AUROC (close to 1.0) means that uncertainty is a good predictor
    of which samples the model will get wrong. This is the key metric for
    "does doubt predict mistakes?".

    Parameters
    ----------
    uncertainties : np.ndarray
        Per-sample uncertainty scores (epistemic or total).
    predictions : np.ndarray
        Per-sample predicted class indices.
    labels : np.ndarray
        Per-sample true class indices.
    """
    uncertainties = np.asarray(uncertainties).ravel()
    predictions = np.asarray(predictions).ravel()
    labels = np.asarray(labels).ravel()

    if len(uncertainties) != len(predictions) or len(uncertainties) != len(labels):
        raise ValueError("uncertainties, predictions, labels must have the same length")

    # Binary target: 1 = error, 0 = correct
    errors = (predictions != labels).astype(int)
    n_pos = errors.sum()
    n_neg = len(errors) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5  # undefined — return random

    # Rank-based AUROC
    # High uncertainty should correlate with high error rate
    order = np.argsort(uncertainties)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(uncertainties) + 1)
    sum_ranks_pos = ranks[errors == 1].sum()
    auroc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auroc)


# ───────────────────────────────────────────────────────────
# Risk-coverage curve
# ───────────────────────────────────────────────────────────

@dataclass
class RiskCoveragePoint:
    """A single point on the risk-coverage curve."""
    coverage: float          # fraction of cases retained (not deferred)
    accuracy: float           # accuracy on retained cases
    threshold: float          # uncertainty threshold at this point
    n_retained: int


@dataclass
class RiskCoverageCurve:
    """The risk-coverage curve — accuracy vs. deferral rate."""
    points: list[RiskCoveragePoint] = field(default_factory=list)

    def add_point(self, coverage: float, accuracy: float, threshold: float, n_retained: int):
        self.points.append(RiskCoveragePoint(coverage, accuracy, threshold, n_retained))

    @property
    def n_points(self) -> int:
        return len(self.points)

    def best_coverage_at_accuracy(self, target_accuracy: float) -> Optional[RiskCoveragePoint]:
        """Find the highest coverage that achieves the target accuracy."""
        valid = [p for p in self.points if p.accuracy >= target_accuracy]
        if not valid:
            return None
        return max(valid, key=lambda p: p.coverage)

    def to_dict(self) -> list[dict]:
        return [
            {"coverage": p.coverage, "accuracy": p.accuracy, "threshold": p.threshold, "n_retained": p.n_retained}
            for p in self.points
        ]


def compute_risk_coverage_curve(
    uncertainties: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    n_thresholds: int = 20,
) -> RiskCoverageCurve:
    """Compute the risk-coverage curve.

    Sorts samples by uncertainty and sweeps a threshold from highest to lowest.
    At each threshold, the model "defers" the most uncertain samples and we
    measure accuracy on the retained cases.

    A good deferral rule shows accuracy increasing as coverage decreases
    (more samples deferred → higher accuracy on the kept cases).
    """
    uncertainties = np.asarray(uncertainties).ravel()
    predictions = np.asarray(predictions).ravel()
    labels = np.asarray(labels).ravel()

    n_total = len(uncertainties)
    curve = RiskCoverageCurve()

    # Sweep thresholds from high uncertainty (defer everything) to low (keep everything)
    sorted_indices = np.argsort(uncertainties)
    thresholds = np.linspace(
        uncertainties.min(),
        uncertainties.max(),
        n_thresholds,
    )

    for threshold in thresholds:
        # Retain samples with uncertainty <= threshold
        retained_mask = uncertainties <= threshold
        n_retained = int(retained_mask.sum())
        if n_retained == 0:
            continue
        retained_preds = predictions[retained_mask]
        retained_labels = labels[retained_mask]
        accuracy = float((retained_preds == retained_labels).mean())
        coverage = n_retained / n_total
        curve.add_point(coverage, accuracy, float(threshold), n_retained)

    # Always include the full-coverage point (no deferral)
    full_acc = float((predictions == labels).mean())
    curve.add_point(1.0, full_acc, float(uncertainties.max()), n_total)

    return curve


# ───────────────────────────────────────────────────────────
# Reliability diagram
# ───────────────────────────────────────────────────────────

@dataclass
class ReliabilityDiagram:
    """Reliability diagram data — binned confidence vs. accuracy."""
    bin_centers: np.ndarray       # center confidence of each bin
    bin_accuracies: np.ndarray    # empirical accuracy in each bin
    bin_confidences: np.ndarray   # mean predicted confidence in each bin
    bin_counts: np.ndarray        # number of samples in each bin
    n_bins: int

    @property
    def ece(self) -> float:
        """Expected Calibration Error computed from this diagram."""
        total = self.bin_counts.sum()
        if total == 0:
            return 0.0
        return float(np.sum(
            (self.bin_counts / total) * np.abs(self.bin_accuracies - self.bin_confidences)
        ))


def compute_reliability_diagram(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> ReliabilityDiagram:
    """Compute the reliability diagram for a classifier.

    The reliability diagram bins predictions by predicted confidence and
    compares the average confidence in each bin to the empirical accuracy
    in that bin. A perfectly calibrated classifier has confidence = accuracy
    in every bin (diagonal line on the plot).

    Parameters
    ----------
    probs : np.ndarray
        (N, C) array of predicted class probabilities.
    labels : np.ndarray
        (N,) array of true class indices.
    n_bins : int
        Number of confidence bins (default 15, following Guo et al. 2017).
    """
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim == 1:
        probs = probs.reshape(1, -1)

    confidences = probs.max(axis=-1)
    predictions = probs.argmax(axis=-1)
    if labels.ndim > 1:
        labels = labels.argmax(axis=-1)
    accuracies = (predictions == labels).astype(np.float64)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    bin_accuracies = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() > 0:
            bin_accuracies[i] = accuracies[mask].mean()
            bin_confidences[i] = confidences[mask].mean()
            bin_counts[i] = int(mask.sum())

    return ReliabilityDiagram(
        bin_centers=bin_centers,
        bin_accuracies=bin_accuracies,
        bin_confidences=bin_confidences,
        bin_counts=bin_counts,
        n_bins=n_bins,
    )


# ───────────────────────────────────────────────────────────
# Brier score decomposition
# ───────────────────────────────────────────────────────────

@dataclass
class BrierDecomposition:
    """Decomposition of the Brier score into reliability, resolution, uncertainty."""
    brier: float
    reliability: float       # lower is better (calibration component)
    resolution: float        # higher is better (discrimination component)
    uncertainty: float       # intrinsic difficulty of the problem


def brier_decomposition(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> BrierDecomposition:
    """Decompose the Brier score into reliability, resolution, and uncertainty.

    Following Murphy (1973):
        Brier = Reliability - Resolution + Uncertainty

    - **Reliability** — how well do predicted probabilities match empirical
      frequencies? Lower is better.
    - **Resolution** — how much do the forecasts differ from the climatological
      base rate? Higher is better.
    - **Uncertainty** — the inherent uncertainty of the event being forecast.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim == 1:
        probs = probs.reshape(1, -1)

    n_samples, n_classes = probs.shape

    # One-hot encode labels
    one_hot = np.zeros_like(probs)
    if labels.ndim == 1:
        one_hot[np.arange(n_samples), labels] = 1.0
    else:
        one_hot = labels.astype(np.float64)

    # Overall Brier score
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=-1)))

    # Bin by max confidence
    confidences = probs.max(axis=-1)
    predictions = probs.argmax(axis=-1)
    accuracies = (predictions == labels if labels.ndim == 1 else
                  predictions == labels.argmax(axis=-1)).astype(np.float64)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    reliability = 0.0
    resolution = 0.0
    base_rate = float(accuracies.mean())
    uncertainty = base_rate * (1 - base_rate)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        n_k = int(mask.sum())
        if n_k == 0:
            continue
        acc_k = float(accuracies[mask].mean())
        conf_k = float(confidences[mask].mean())
        weight = n_k / n_samples

        # Reliability — |confidence - accuracy|² in each bin
        reliability += weight * (conf_k - acc_k) ** 2
        # Resolution — |accuracy - base_rate|² in each bin
        resolution += weight * (acc_k - base_rate) ** 2

    return BrierDecomposition(
        brier=brier,
        reliability=reliability,
        resolution=resolution,
        uncertainty=uncertainty,
    )
