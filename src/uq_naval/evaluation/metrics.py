"""Calibration and operational metrics.

Each metric is implemented as a pure function operating on numpy arrays,
so they can be used both within the evaluation harness and standalone.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------

def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error (ECE).

    Measures the gap between predicted confidence and empirical accuracy,
    averaged over confidence bins.

    Parameters
    ----------
    probs:
        (N, C) array of predicted class probabilities.
    labels:
        (N,) array of true class indices (or (N, C) one-hot).
    n_bins:
        Number of confidence bins.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim == 1:
        probs = probs.reshape(1, -1)

    confidences = probs.max(axis=-1)
    predictions = probs.argmax(axis=-1)
    if labels.ndim == 1:
        accuracies = (predictions == labels).astype(np.float64)
    else:
        accuracies = (predictions == labels.argmax(axis=-1)).astype(np.float64)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        ece += abs(accuracies[mask].mean() - confidences[mask].mean()) * mask.sum() / n
    return float(ece)


def brier_score(
    probs: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Brier Score — mean squared error of probabilistic predictions.

    Lower is better; baseline 0.25 for uniform random binary prediction.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim == 1:
        probs = probs.reshape(1, -1)
    if labels.ndim == 1:
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(len(labels)), labels] = 1.0
    else:
        one_hot = labels
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=-1)))


def empirical_coverage(
    prediction_sets: list[list[int]],
    true_labels: np.ndarray,
) -> float:
    """Fraction of test instances whose true label is in the prediction set."""
    n = len(true_labels)
    if n == 0:
        return 0.0
    covered = sum(1 for s, y in zip(prediction_sets, true_labels) if int(y) in set(s))
    return float(covered / n)


def ood_auroc(
    in_dist_unc: np.ndarray,
    out_dist_unc: np.ndarray,
) -> float:
    """AUROC for OOD detection using uncertainty as the score.

    A perfect OOD detector assigns higher uncertainty to OOD inputs.
    Returns 0.5 when scores are all equal (random-tie convention).
    """
    in_dist_unc = np.asarray(in_dist_unc).ravel()
    out_dist_unc = np.asarray(out_dist_unc).ravel()

    # Random-tie convention: if all scores are equal, return 0.5
    if np.all(in_dist_unc == in_dist_unc[0]) and np.all(out_dist_unc == out_dist_unc[0]):
        if in_dist_unc[0] == out_dist_unc[0]:
            return 0.5

    labels = np.concatenate([np.zeros(len(in_dist_unc)), np.ones(len(out_dist_unc))])
    scores = np.concatenate([in_dist_unc, out_dist_unc])
    # Rank-based AUROC
    order = np.argsort(scores, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1
    n_pos = pos.sum()
    n_neg = (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return 0.5
    sum_ranks_pos = ranks[pos].sum()
    auroc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auroc)


# ---------------------------------------------------------------------------
# Operational metrics
# ---------------------------------------------------------------------------

def colregs_violation_rate(
    decisions: list[dict],
) -> float:
    """Fraction of decisions where COLREGs compliance < 1.0."""
    if not decisions:
        return 0.0
    n_violations = sum(
        1 for d in decisions
        if d.get("colregs_compliance", 1.0) < 1.0
    )
    return float(n_violations / len(decisions))


def rejection_rate(
    decisions: list[dict],
) -> float:
    """Fraction of decisions that triggered the reject option."""
    if not decisions:
        return 0.0
    n_reject = sum(1 for d in decisions if d.get("action") != "EXECUTE")
    return float(n_reject / len(decisions))


# ---------------------------------------------------------------------------
# Combined
# ---------------------------------------------------------------------------

def compute_all_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    prediction_sets: Optional[list[list[int]]] = None,
    decisions: Optional[list[dict]] = None,
) -> dict[str, float]:
    """Compute all calibration + operational metrics in one call."""
    metrics = {
        "ece": expected_calibration_error(probs, labels),
        "brier": brier_score(probs, labels),
    }
    if prediction_sets is not None:
        metrics["coverage"] = empirical_coverage(prediction_sets, labels)
        metrics["mean_set_size"] = float(np.mean([len(s) for s in prediction_sets]))
    if decisions is not None:
        metrics["colregs_violation_rate"] = colregs_violation_rate(decisions)
        metrics["rejection_rate"] = rejection_rate(decisions)
    return metrics
