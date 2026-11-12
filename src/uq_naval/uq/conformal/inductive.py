"""Inductive (split) conformal prediction.

Distribution-free UQ: produces a prediction set guaranteed to contain the
true label with probability at least 1-α, provided the calibration and test
data are exchangeable.

This implementation wraps any pre-trained classifier. Given a calibration
set, it computes the (1-α)-quantile of the nonconformity score distribution
and uses it to construct prediction sets at inference time.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..base import UQInterface, register_uq_method
from ...types import CalibrationResult, UQOutput


@register_uq_method("conformal")
class InductiveConformal(UQInterface):
    """Inductive (split) conformal prediction wrapper.

    Parameters
    ----------
    model:
        A callable producing class probabilities (softmax outputs).
    config:
        - ``alpha`` (float, default 0.1) — miscoverage tolerance; coverage = 1-α.
        - ``score`` (str, default "thr") — nonconformity score: "thr" (1-prob),
          "aps" (accumulated probability score, Angelopoulos et al. 2021).
        - ``reject_threshold`` (float, default 0.7) — minimum set size to trigger reject.
    """

    def __init__(self, model: Any, config: Optional[dict] = None) -> None:
        super().__init__(model=model, config=config)
        self.alpha: float = float(self.config.get("alpha", 0.1))
        self.score_type: str = str(self.config.get("score", "thr"))
        self.reject_threshold: float = float(self.config.get("reject_threshold", 0.7))
        self._quantile: float = float("inf")
        self._n_classes: int = 0

    # ------------------------------------------------------------------
    def calibrate(self, calibration_data: Any) -> CalibrationResult:
        """Compute the conformal quantile on the calibration set.

        ``calibration_data`` is an iterable of (x, y) batches. We compute
        nonconformity scores for each calibration example and take the
        ceil((n+1)(1-α)/n)-th quantile.
        """
        scores, labels = [], []
        for batch in calibration_data:
            x, y = batch
            probs = self._predict_proba(x)
            scores.append(self._score_fn(probs))
            labels.append(np.atleast_1d(np.asarray(y)))

        all_scores = np.concatenate([s.ravel() for s in scores])
        all_labels = np.concatenate([l.ravel() for l in labels])
        n = len(all_scores)

        # Conformal quantile: ceil((n+1)(1-α)) / n  (finite-sample correction)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)
        self._quantile = float(np.quantile(all_scores, q_level, method="higher"))
        self._n_classes = int(max([s.shape[-1] if s.ndim > 0 else 1 for s in scores]))
        self._calibrated = True
        self._calibration_state = {
            "quantile": self._quantile,
            "n_classes": self._n_classes,
            "n_calibration": n,
        }

        # Empirical coverage on the calibration set itself (sanity check)
        covered = 0
        for s_batch, l_batch in zip(scores, labels):
            for i in range(len(l_batch)):
                if self.score_type == "thr":
                    set_c = np.where(s_batch[i] <= self._quantile)[0]
                else:
                    set_c = self._invert_aps_score(s_batch[i], l_batch[i])
                if l_batch[i] in set_c:
                    covered += 1
        empirical_coverage = covered / n if n > 0 else 0.0

        return CalibrationResult(
            method=self.method_name,
            n_calibration=n,
            metrics={
                "nominal_coverage": 1 - self.alpha,
                "empirical_coverage": empirical_coverage,
                "quantile": self._quantile,
                "mean_set_size": float(np.mean([
                    np.sum(s_row <= self._quantile) for s_row in all_scores.reshape(-1, 1)
                ])) if n > 0 else 0.0,
            },
            threshold=self._quantile,
            extra=self._calibration_state,
        )

    # ------------------------------------------------------------------
    def predict(self, x: Any) -> UQOutput:
        if not self._calibrated:
            raise RuntimeError("Conformal predictor must be calibrated before predict().")

        probs = self._predict_proba(x)
        if probs.ndim == 1:
            probs = probs.reshape(1, -1)
        # Use the first sample (single-input prediction)
        probs = probs[0]

        # Construct the prediction set
        if self.score_type == "thr":
            scores = 1 - probs  # threshold score
            prediction_set = list(np.where(scores <= self._quantile)[0])
        elif self.score_type == "aps":
            # Accumulated probability score: sort classes by descending prob,
            # include until accumulated prob >= 1 - quantile.
            sorted_idx = np.argsort(-probs)
            cum = np.cumsum(probs[sorted_idx])
            cutoff = np.searchsorted(cum, 1 - self._quantile) + 1
            prediction_set = list(sorted_idx[:cutoff])
        else:
            raise ValueError(f"Unknown score type: {self.score_type}")

        # Confidence: max class probability
        confidence = float(probs.max())
        prediction = int(probs.argmax())

        # Epistemic: high if prediction set is large (model is uncertain)
        set_size = len(prediction_set)
        epistemic = float(set_size / self._n_classes) if self._n_classes > 0 else 1.0
        aleatoric = float(1 - confidence)  # data-inherent uncertainty

        reject = bool(set_size > self.reject_threshold * self._n_classes)

        return UQOutput(
            prediction=prediction,
            aleatoric_unc=aleatoric,
            epistemic_unc=epistemic,
            prediction_set=prediction_set,
            confidence=confidence,
            reject=reject,
            audit={
                "alpha": self.alpha,
                "quantile": self._quantile,
                "set_size": set_size,
                "n_classes": self._n_classes,
                "score_type": self.score_type,
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _predict_proba(self, x: Any) -> np.ndarray:
        out = self._call_model(x)
        if out.ndim == 1:
            out = out.reshape(1, -1)
        # Assume model outputs either logits or probabilities; if logits, softmax
        if (out < 0).any() or (out.sum(axis=-1) - 1).abs().max() > 0.1 if hasattr(out, "abs") else True:
            out = _softmax(out, axis=-1)
        return out

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

    def _score_fn(self, probs: np.ndarray) -> np.ndarray:
        """Nonconformity score per (sample, class) pair."""
        if self.score_type == "thr":
            return 1 - probs  # threshold score: lower = more conforming
        elif self.score_type == "aps":
            # For APS we need the true label; return 1-prob here and invert later.
            return 1 - probs
        raise ValueError(f"Unknown score type: {self.score_type}")

    def _invert_aps_score(self, scores_row: np.ndarray, label: int) -> np.ndarray:
        probs_row = 1 - scores_row
        sorted_idx = np.argsort(-probs_row)
        cum = np.cumsum(probs_row[sorted_idx])
        cutoff = np.searchsorted(cum, probs_row[label]) + 1
        return sorted_idx[:cutoff]


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)
