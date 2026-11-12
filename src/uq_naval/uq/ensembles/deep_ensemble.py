"""Standard Deep Ensemble (Lakshminarayanan et al., 2017).

Trains M independent copies of the same architecture with different random
initialisations. The ensemble's predictive distribution is the mixture of
per-member distributions; epistemic uncertainty is the variance of per-member
means.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..base import UQInterface, register_uq_method
from ...types import CalibrationResult, UQOutput


@register_uq_method("ensemble")
class DeepEnsemble(UQInterface):
    """Standard deep ensemble.

    Parameters
    ----------
    model:
        A list of M trained models, or a single callable model that internally
        aggregates M members (e.g., returns (M, ...) outputs).
    config:
        - ``temperature`` (float, default 1.0) — post-hoc temperature scaling.
        - ``reject_threshold`` (float, default 0.5)
        - ``diversity_weight`` (float, default 0.0) — penalise member agreement
          (used only when training; ignored at inference).
    """

    def __init__(self, model: Any, config: Optional[dict] = None) -> None:
        super().__init__(model=model, config=config)
        self.temperature: float = float(self.config.get("temperature", 1.0))
        self.reject_threshold: float = float(self.config.get("reject_threshold", 0.5))
        self._temperature: float = self.temperature

    # ------------------------------------------------------------------
    def calibrate(self, calibration_data: Any) -> CalibrationResult:
        """Temperature-scale the ensemble on a calibration set."""
        all_probs, all_labels = [], []
        for batch in calibration_data:
            x, y = batch
            probs = self._ensemble_predict_proba(x)  # (M, B, C)
            mean_probs = probs.mean(axis=0)  # (B, C)
            all_probs.append(np.atleast_2d(mean_probs))
            all_labels.append(np.atleast_1d(np.asarray(y)))

        probs = np.concatenate(all_probs, axis=0)
        labels = np.concatenate(all_labels, axis=0)

        # Grid-search temperature
        best_T, best_nll = 1.0, float("inf")
        logits = np.log(probs + 1e-12)
        for T in np.linspace(0.05, 5.0, 50):
            nll = _softmax_nll(logits / T, labels)
            if nll < best_nll:
                best_nll, best_T = nll, T

        self._temperature = float(best_T)
        self._calibrated = True
        self._calibration_state = {"temperature": self._temperature}

        ece = _expected_calibration_error(_softmax(logits / self._temperature), labels)

        return CalibrationResult(
            method=self.method_name,
            n_calibration=len(labels),
            metrics={
                "nll": best_nll,
                "ece": ece,
                "temperature": self._temperature,
            },
            threshold=self._temperature,
        )

    # ------------------------------------------------------------------
    def predict(self, x: Any) -> UQOutput:
        if not self._calibrated:
            self._temperature = self.temperature

        probs = self._ensemble_predict_proba(x)  # (M, C) or (M, B, C)
        if probs.ndim == 3:
            probs = probs[:, 0]  # single-input prediction

        # Apply temperature scaling
        logits = np.log(probs + 1e-12)
        scaled_probs = _softmax(logits / self._temperature, axis=-1)

        # Predictive mean
        mean_probs = scaled_probs.mean(axis=0)  # (C,)

        # Epistemic uncertainty — variance of mean-probability across members
        epistemic = float(scaled_probs.var(axis=0).sum())

        # Aleatoric uncertainty — expected entropy of each member
        per_member_entropy = -np.sum(scaled_probs * np.log(scaled_probs + 1e-12), axis=-1)
        aleatoric = float(per_member_entropy.mean())

        confidence = float(mean_probs.max())
        prediction = int(mean_probs.argmax())
        reject = bool(epistemic > self.reject_threshold)

        return UQOutput(
            prediction=prediction,
            aleatoric_unc=aleatoric,
            epistemic_unc=epistemic,
            confidence=confidence,
            reject=reject,
            audit={
                "n_members": int(probs.shape[0]),
                "member_predictions": scaled_probs.argmax(axis=-1).tolist(),
                "temperature": self._temperature,
            },
        )

    # ------------------------------------------------------------------
    def _ensemble_predict_proba(self, x: Any) -> np.ndarray:
        """Return per-member probabilities. Shape (M, ..., C)."""
        if isinstance(self.model, (list, tuple)):
            outputs = []
            for m in self.model:
                out = self._call_single(m, x)
                outputs.append(out)
            return np.stack(outputs, axis=0)
        else:
            # Single model that returns (M, ...) outputs
            out = self._call_single(self.model, x)
            if out.ndim >= 1 and out.shape[0] > 1:
                return out  # assume already stacked
            return np.stack([out, out], axis=0)  # fallback: 2 identical members

    def _call_single(self, model: Any, x: Any) -> np.ndarray:
        try:
            out = model(x)
            if hasattr(out, "detach"):
                arr = out.detach().cpu().numpy()
            else:
                arr = np.asarray(out)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            return _softmax(arr, axis=-1)
        except Exception:
            import torch
            if isinstance(x, np.ndarray):
                out = model(torch.from_numpy(x).float())
                arr = out.detach().cpu().numpy() if hasattr(out, "detach") else np.asarray(out)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                return _softmax(arr, axis=-1)
            raise


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _softmax_nll(logits: np.ndarray, labels: np.ndarray) -> float:
    probs = _softmax(logits)
    if labels.ndim == 1:
        return float(-np.mean(np.log(probs[np.arange(len(labels)), labels] + 1e-12)))
    return float(-np.mean(np.sum(labels * np.log(probs + 1e-12), axis=-1)))


def _expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
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
