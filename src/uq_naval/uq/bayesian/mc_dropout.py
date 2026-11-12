"""MC Dropout Bayesian approximation (Gal & Ghahramani, 2016).

Approximates variational inference in a Gaussian process by applying dropout
stochastically at inference time. The mean of T stochastic forward passes
approximates the predictive mean; their variance decomposes into aleatoric
and epistemic components.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..base import UQInterface, register_uq_method
from ...types import CalibrationResult, UQOutput


@register_uq_method("bayesian")
class MCDropoutBayesian(UQInterface):
    """MC Dropout Bayesian approximation.

    Parameters
    ----------
    model:
        A callable (typically an ``nn.Module``) that supports a ``training``
        flag to enable/disable dropout. For MC Dropout to work, dropout layers
        must be active during the T stochastic passes; this is achieved by
        calling ``model.train()`` or per-layer ``module.dropout.p = ...``.
    config:
        - ``n_mc_samples``  (int, default 10)   — T stochastic forward passes
        - ``reject_threshold`` (float, default 0.5) — epistemic threshold above
          which the method sets ``UQOutput.reject = True``.
        - ``temperature`` (float, default 1.0) — post-hoc temperature scaling.
    """

    def __init__(self, model: Any, config: Optional[dict] = None) -> None:
        super().__init__(model=model, config=config)
        self.n_mc_samples: int = int(self.config.get("n_mc_samples", 10))
        self.reject_threshold: float = float(self.config.get("reject_threshold", 0.5))
        self.temperature: float = float(self.config.get("temperature", 1.0))
        # Calibration state — set by .calibrate()
        self._temperature: float = self.temperature
        self._calibration_state = {"temperature": self._temperature}

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    def calibrate(self, calibration_data: Any) -> CalibrationResult:
        """Temperature-scale on the calibration set.

        ``calibration_data`` is expected to be an iterable of (logits, label)
        pairs. We grid-search the temperature T in [0.05, 5.0] minimising NLL.
        """
        logits_list, labels_list = [], []
        for batch in calibration_data:
            x, y = batch
            with _no_grad(self.model):
                out = _safe_call(self.model, x)
                logits_list.append(np.atleast_2d(np.asarray(out)))
                labels_list.append(np.atleast_1d(np.asarray(y)))

        logits = np.concatenate(logits_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)

        # Grid-search temperature minimising NLL
        best_T, best_nll = 1.0, float("inf")
        for T in np.linspace(0.05, 5.0, 100):
            scaled = logits / T
            nll = _softmax_nll(scaled, labels)
            if nll < best_nll:
                best_nll, best_T = nll, T

        self._temperature = float(best_T)
        self._calibration_state = {"temperature": self._temperature}
        self._calibrated = True

        # ECE before / after
        ece_before = _expected_calibration_error(_softmax(logits), labels)
        ece_after = _expected_calibration_error(_softmax(logits / self._temperature), labels)

        return CalibrationResult(
            method=self.method_name,
            n_calibration=len(labels),
            metrics={
                "ece_before": ece_before,
                "ece_after": ece_after,
                "nll": best_nll,
                "temperature": self._temperature,
            },
            threshold=self._temperature,
            extra={"n_mc_samples": self.n_mc_samples},
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, x: Any) -> UQOutput:
        if not self._calibrated:
            # Allow prediction without explicit calibration — use default temperature
            self._temperature = self.temperature

        # Enable dropout for MC sampling
        _set_train_mode_with_dropout(self.model, train=True)

        # T stochastic forward passes
        samples = []
        with _no_grad(self.model):
            for _ in range(self.n_mc_samples):
                out = _safe_call(self.model, x)
                samples.append(np.asarray(out))

        # Disable dropout for any downstream use
        _set_train_mode_with_dropout(self.model, train=False)

        samples = np.stack(samples, axis=0)  # (T, ...)
        scaled = samples / self._temperature
        probs = _softmax(scaled, axis=-1)  # (T, ..., C)

        # Predictive mean
        mean_probs = probs.mean(axis=0)  # (..., C)

        # Epistemic uncertainty — variance of mean-probability across samples
        epistemic = probs.var(axis=0).sum(axis=-1)  # sum over classes

        # Aleatoric uncertainty — expected entropy of each sample
        per_sample_entropy = -np.sum(probs * np.log(probs + 1e-12), axis=-1)  # (T, ...)
        aleatoric = per_sample_entropy.mean(axis=0)

        # Confidence — max class probability after calibration
        max_probs = mean_probs.max(axis=-1)
        confidence = float(max_probs.mean()) if max_probs.size > 0 else 0.5

        # Predicted label
        if mean_probs.ndim >= 1:
            prediction = int(mean_probs.argmax(axis=-1).ravel()[0]) if mean_probs.size > 1 else int(mean_probs.ravel()[0])
        else:
            prediction = float(mean_probs)

        reject = bool(np.mean(epistemic) > self.reject_threshold)

        return UQOutput(
            prediction=prediction,
            aleatoric_unc=float(np.mean(aleatoric)),
            epistemic_unc=float(np.mean(epistemic)),
            prediction_set=None,  # MC Dropout does not produce sets; use Conformal wrapper
            confidence=confidence,
            reject=reject,
            audit={
                "n_mc_samples": self.n_mc_samples,
                "temperature": self._temperature,
                "sample_mean": float(np.mean(samples)),
                "sample_std": float(np.std(samples)),
            },
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_call(model: Any, x: Any) -> Any:
    """Call the model, handling both torch tensors and numpy arrays."""
    # If x is a torch tensor and model is an nn.Module, just call
    try:
        return model(x)
    except Exception:
        # Try to convert numpy to torch tensor
        import torch
        if isinstance(x, np.ndarray):
            x_t = torch.from_numpy(x).float()
            return model(x_t).detach().cpu().numpy()
        raise


def _no_grad(model: Any):
    """Context manager that disables gradients if model is a torch nn.Module."""
    try:
        import torch
        return torch.no_grad()
    except ImportError:
        from contextlib import nullcontext
        return nullcontext()


def _set_train_mode_with_dropout(model: Any, train: bool) -> None:
    """Enable/disable dropout layers without affecting BatchNorm.

    For MC Dropout we want dropout active but BatchNorm frozen. We toggle
    only ``torch.nn.Dropout`` and ``torch.nn.Dropout2d`` layers.
    """
    try:
        import torch.nn as nn
    except ImportError:
        return

    if not hasattr(model, "modules"):
        return

    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.AlphaDropout)):
            module.train() if train else module.eval()


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _softmax_nll(logits: np.ndarray, labels: np.ndarray) -> float:
    """Negative log-likelihood of labels under softmax(logits)."""
    probs = _softmax(logits)
    n = len(labels)
    if labels.ndim == 1:
        return float(-np.mean(np.log(probs[np.arange(n), labels] + 1e-12)))
    # one-hot
    return float(-np.mean(np.sum(labels * np.log(probs + 1e-12), axis=-1)))


def _expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """ECE: gap between predicted confidence and empirical accuracy."""
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
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += np.abs(bin_acc - bin_conf) * mask.sum() / n
    return float(ece)
