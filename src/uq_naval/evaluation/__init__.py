"""Evaluation harness — metrics + scenarios + benchmark runner.

Computes all reported calibration metrics across UQ methods on a unified
benchmark, producing the comparison tables reported in the proposal.
"""
from .metrics import (
    expected_calibration_error,
    brier_score,
    empirical_coverage,
    ood_auroc,
    colregs_violation_rate,
    rejection_rate,
    compute_all_metrics,
)
from .harness import EvaluationHarness
from .scenarios import ScenarioLibrary

__all__ = [
    "expected_calibration_error",
    "brier_score",
    "empirical_coverage",
    "ood_auroc",
    "colregs_violation_rate",
    "rejection_rate",
    "compute_all_metrics",
    "EvaluationHarness",
    "ScenarioLibrary",
]
