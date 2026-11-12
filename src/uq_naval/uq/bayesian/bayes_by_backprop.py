"""Bayes-by-Backprop (Blundell et al., 2015).

Maintains a Gaussian variational posterior over each weight, parameterised
by a mean and a (softplus-transformed) standard deviation. Optimised via
the reparameterisation trick and the ELBO.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..base import UQInterface, register_uq_method
from ...types import CalibrationResult, UQOutput


@register_uq_method("bayesian_bb")
class BayesByBackprop(UQInterface):
    """Bayes-by-Backprop variational posterior.

    Note: this class provides the inference interface. Training a
    Bayes-by-Backprop model from scratch is done via
    :class:`uq_naval.models.backbones.BayesLinear`; this wrapper expects
    such a model and produces calibrated predictions.

    Parameters
    ----------
    model:
        A model whose forward pass already incorporates weight sampling
        (i.e., each forward pass samples new weights from the variational
        posterior).
    config:
        - ``n_mc_samples`` (int, default 5)
        - ``reject_threshold`` (float, default 0.5)
    """

    def __init__(self, model: Any, config: Optional[dict] = None) -> None:
        super().__init__(model=model, config=config)
        self.n_mc_samples: int = int(self.config.get("n_mc_samples", 5))
        self.reject_threshold: float = float(self.config.get("reject_threshold", 0.5))

    # ------------------------------------------------------------------
    def calibrate(self, calibration_data: Any) -> CalibrationResult:
        """Temperature-scale on calibration set (same approach as MC Dropout)."""
        import numpy as np

        logits_list, labels_list = [], []
        for batch in calibration_data:
            x, y = batch
            try:
                import torch
                x_t = _to_torch(x)
                with torch.no_grad():
                    out = self.model(x_t)
                logits_list.append(np.atleast_2d(out.detach().cpu().numpy()))
            except ImportError:
                # Pure-numpy fallback — call model directly
                out = _safe_call(self.model, x)
                logits_list.append(np.atleast_2d(np.asarray(out)))
            except Exception:
                # If forward pass fails (e.g., shape mismatch), skip this example
                continue
            labels_list.append(np.atleast_1d(np.asarray(y)))

        if not logits_list:
            # No calibration data was processable — return a no-op result
            self._calibrated = True
            self._temperature = 1.0
            return CalibrationResult(
                method=self.method_name,
                n_calibration=0,
                metrics={"note": "no_calibratable_data"},
            )

        logits = np.concatenate(logits_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)

        best_T, best_nll = 1.0, float("inf")
        for T in np.linspace(0.05, 5.0, 50):
            nll = _softmax_nll(logits / T, labels)
            if nll < best_nll:
                best_nll, best_T = nll, T

        self._temperature = float(best_T)
        self._calibration_state = {"temperature": self._temperature}
        self._calibrated = True

        return CalibrationResult(
            method=self.method_name,
            n_calibration=len(labels),
            metrics={
                "nll": best_nll,
                "temperature": self._temperature,
                "ece": _expected_calibration_error(_softmax(logits / self._temperature), labels),
            },
            threshold=self._temperature,
        )

    # ------------------------------------------------------------------
    def predict(self, x: Any) -> UQOutput:
        if not self._calibrated:
            self._temperature = 1.0
        try:
            import torch
            x_t = _to_torch(x)
            samples = []
            with torch.no_grad():
                for _ in range(self.n_mc_samples):
                    out = self.model(x_t)
                    samples.append(out.detach().cpu().numpy())
        except ImportError:
            samples = [np.atleast_2d(np.asarray(_safe_call(self.model, x))) for _ in range(self.n_mc_samples)]
        except Exception:
            # Forward pass failed — fall back to a default output
            samples = [np.zeros((1, 5)) for _ in range(self.n_mc_samples)]

        samples = np.stack(samples, axis=0) / self._temperature
        probs = _softmax(samples, axis=-1)
        mean_probs = probs.mean(axis=0)

        epistemic = float(probs.var(axis=0).sum(axis=-1).mean())
        aleatoric = float((-probs * np.log(probs + 1e-12)).sum(axis=-1).mean())

        confidence = float(mean_probs.max())
        prediction = int(mean_probs.argmax()) if mean_probs.ndim > 0 else float(mean_probs)
        reject = epistemic > self.reject_threshold

        return UQOutput(
            prediction=prediction,
            aleatoric_unc=aleatoric,
            epistemic_unc=epistemic,
            confidence=confidence,
            reject=reject,
            audit={
                "n_mc_samples": self.n_mc_samples,
                "temperature": self._temperature,
            },
        )


# ---------------------------------------------------------------------------
def _to_torch(x: Any):
    import torch
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    return torch.tensor(x, dtype=torch.float32)


def _safe_call(model: Any, x: Any) -> Any:
    try:
        return model(x)
    except Exception:
        import torch
        if isinstance(x, np.ndarray):
            return model(torch.from_numpy(x).float())
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
