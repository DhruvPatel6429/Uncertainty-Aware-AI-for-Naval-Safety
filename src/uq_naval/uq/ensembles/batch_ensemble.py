"""Batch-Ensembled network (Wen et al., 2020).

Shares most parameters across ensemble members and learns only rank-one
perturbations per member. Reduces both training and inference cost by an
order of magnitude vs. standard ensembles.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..base import UQInterface, register_uq_method
from ...types import CalibrationResult, UQOutput


@register_uq_method("ensemble_batch")
class BatchEnsemble(UQInterface):
    """Batch-ensembled network.

    The model is expected to be a callable that produces per-member outputs
    in shape (M, ...). At inference, the M outputs are aggregated like a
    standard ensemble.

    Parameters
    ----------
    model:
        A batch-ensembled model. The forward pass returns (M, ...) outputs.
    config:
        - ``n_members`` (int, default 4) — M ensemble members
        - ``temperature`` (float, default 1.0)
        - ``reject_threshold`` (float, default 0.5)
    """

    def __init__(self, model: Any, config: Optional[dict] = None) -> None:
        super().__init__(model=model, config=config)
        self.n_members: int = int(self.config.get("n_members", 4))
        self.temperature: float = float(self.config.get("temperature", 1.0))
        self.reject_threshold: float = float(self.config.get("reject_threshold", 0.5))
        self._temperature: float = self.temperature

    def calibrate(self, calibration_data: Any) -> CalibrationResult:
        # Same temperature scaling as DeepEnsemble
        all_probs, all_labels = [], []
        for batch in calibration_data:
            x, y = batch
            probs = self._batch_predict_proba(x)  # (M, B, C)
            mean_probs = probs.mean(axis=0)
            all_probs.append(np.atleast_2d(mean_probs))
            all_labels.append(np.atleast_1d(np.asarray(y)))

        probs = np.concatenate(all_probs, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        logits = np.log(probs + 1e-12)

        best_T, best_nll = 1.0, float("inf")
        for T in np.linspace(0.05, 5.0, 50):
            nll = _softmax_nll(logits / T, labels)
            if nll < best_nll:
                best_nll, best_T = nll, T

        self._temperature = float(best_T)
        self._calibrated = True
        self._calibration_state = {"temperature": self._temperature}

        return CalibrationResult(
            method=self.method_name,
            n_calibration=len(labels),
            metrics={
                "nll": best_nll,
                "temperature": self._temperature,
            },
            threshold=self._temperature,
            extra={"n_members": self.n_members},
        )

    def predict(self, x: Any) -> UQOutput:
        if not self._calibrated:
            self._temperature = self.temperature

        probs = self._batch_predict_proba(x)  # (M, C) or (M, B, C)
        if probs.ndim == 3:
            probs = probs[:, 0]

        logits = np.log(probs + 1e-12)
        scaled = _softmax(logits / self._temperature, axis=-1)

        mean_probs = scaled.mean(axis=0)
        epistemic = float(scaled.var(axis=0).sum())
        aleatoric = float((-scaled * np.log(scaled + 1e-12)).sum(axis=-1).mean())
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
                "n_members": self.n_members,
                "temperature": self._temperature,
                "method": "batch_ensemble",
            },
        )

    def _batch_predict_proba(self, x: Any) -> np.ndarray:
        try:
            out = self.model(x)
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
                out = self.model(torch.from_numpy(x).float())
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
