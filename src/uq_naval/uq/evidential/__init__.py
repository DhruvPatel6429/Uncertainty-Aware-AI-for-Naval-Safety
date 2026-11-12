"""Evidential Deep Learning methods.

- :class:`EvidentialClassifier` (Sensoy et al., 2018) — Dirichlet prior over class probs.
- :class:`EvidentialRegressor` (Amini et al., 2020) — Normal-Inverse-Gamma prior.

Both produce uncertainty decomposed into aleatoric and epistemic in a single
forward pass — making them suitable for latency-critical real-time loops.
"""
from .classification import EvidentialClassifier
from .regression import EvidentialRegressor

__all__ = ["EvidentialClassifier", "EvidentialRegressor"]
