"""Mondrian (class-conditional) conformal prediction.

Produces per-class coverage guarantees — operationally significant when
different object classes (small craft vs. large vessel vs. buoy) carry
different misclassification costs.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..base import UQInterface, register_uq_method
from ...types import CalibrationResult, UQOutput
from .inductive import InductiveConformal


@register_uq_method("conformal_mondrian")
class MondrianConformal(UQInterface):
    """Class-conditional conformal prediction.

    Maintains a separate conformal quantile per class, so that the
    prediction set for an example of class c contains c with probability
    ≥ 1-α (per class, marginalised over examples of class c).

    Parameters
    ----------
    model:
        A callable producing class probabilities.
    config:
        - ``alpha`` (float, default 0.1)
        - ``reject_threshold`` (float, default 0.7)
    """

    def __init__(self, model: Any, config: Optional[dict] = None) -> None:
        super().__init__(model=model, config=config)
        self.alpha: float = float(self.config.get("alpha", 0.1))
        self.reject_threshold: float = float(self.config.get("reject_threshold", 0.7))
        self._per_class_quantiles: dict[int, float] = {}
        self._n_classes: int = 0

    # ------------------------------------------------------------------
    def calibrate(self, calibration_data: Any) -> CalibrationResult:
        # Collect (prob, label) per example
        all_probs, all_labels = [], []
        for batch in calibration_data:
            x, y = batch
            probs = self._call_model(x)
            all_probs.append(np.atleast_2d(probs))
            all_labels.append(np.atleast_1d(np.asarray(y)))
        probs = np.concatenate(all_probs, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        n = len(labels)
        self._n_classes = int(probs.shape[-1])

        # Per-class quantile of (1 - prob[c]) for examples of class c
        self._per_class_quantiles = {}
        for c in range(self._n_classes):
            mask = labels == c
            if mask.sum() == 0:
                self._per_class_quantiles[c] = float("inf")
                continue
            scores = 1 - probs[mask, c]
            q_level = np.ceil((mask.sum() + 1) * (1 - self.alpha)) / mask.sum()
            q_level = min(q_level, 1.0)
            self._per_class_quantiles[c] = float(np.quantile(scores, q_level, method="higher"))

        self._calibrated = True
        self._calibration_state = {
            "per_class_quantiles": self._per_class_quantiles,
            "n_classes": self._n_classes,
            "n_calibration": n,
        }

        # Empirical per-class coverage
        per_class_cov = {}
        for c in range(self._n_classes):
            mask = labels == c
            if mask.sum() == 0:
                continue
            # An example is "covered" if the class-c quantile includes its true class
            scores = 1 - probs[mask, c]
            per_class_cov[c] = float(np.mean(scores <= self._per_class_quantiles[c]))

        return CalibrationResult(
            method=self.method_name,
            n_calibration=n,
            metrics={
                "nominal_coverage": 1 - self.alpha,
                "mean_per_class_coverage": float(np.mean(list(per_class_cov.values()))) if per_class_cov else 0.0,
                **{f"coverage_class_{c}": v for c, v in per_class_cov.items()},
            },
            extra=self._calibration_state,
        )

    # ------------------------------------------------------------------
    def predict(self, x: Any) -> UQOutput:
        if not self._calibrated:
            raise RuntimeError("Mondrian conformal must be calibrated first.")

        probs = self._call_model(x)
        if probs.ndim == 1:
            probs = probs.reshape(1, -1)
        probs = probs[0]

        # Prediction set: class c is in set iff 1 - prob[c] <= q_c
        prediction_set = []
        for c in range(self._n_classes):
            if 1 - probs[c] <= self._per_class_quantiles.get(c, float("inf")):
                prediction_set.append(c)

        confidence = float(probs.max())
        prediction = int(probs.argmax())
        set_size = len(prediction_set)
        epistemic = float(set_size / self._n_classes)
        aleatoric = float(1 - confidence)
        reject = bool(set_size > self.reject_threshold * self._n_classes)

        return UQOutput(
            prediction=prediction,
            aleatoric_unc=aleatoric,
            epistemic_unc=epistemic,
            prediction_set=prediction_set,
            confidence=confidence,
            reject=reject,
            audit={
                "per_class_quantiles": self._per_class_quantiles,
                "set_size": set_size,
                "alpha": self.alpha,
            },
        )

    # ------------------------------------------------------------------
    def _call_model(self, x: Any) -> np.ndarray:
        try:
            out = self.model(x)
            if hasattr(out, "detach"):
                return out.detach().cpu().numpy()
            return np.asarray(out)
        except Exception:
            import torch
            if isinstance(x, np.ndarray):
                out = self.model(torch.from_numpy(x).float())
                return out.detach().cpu().numpy() if hasattr(out, "detach") else np.asarray(out)
            raise
