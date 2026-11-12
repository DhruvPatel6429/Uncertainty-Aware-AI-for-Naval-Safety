"""Evidential classification (Sensoy et al., 2018).

The network outputs the parameters of a Dirichlet distribution over
class-probability vectors. The Dirichlet concentration encodes belief
mass; total belief captures model confidence; epistemic uncertainty is
high when total belief is low.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..base import UQInterface, register_uq_method
from ...types import CalibrationResult, UQOutput


@register_uq_method("evidential_clf")
class EvidentialClassifier(UQInterface):
    """Evidential classification via Dirichlet outputs.

    Parameters
    ----------
    model:
        A callable whose output is the evidence vector ``e ∈ R^C_+`` (shape
        (batch, C) or (C,)). Class probabilities are alpha / S where
        alpha = e + 1 and S = sum(alpha).
    config:
        - ``reject_threshold`` (float, default 0.5) — total-belief threshold
          below which the method sets ``reject = True``.
        - ``kl_regularisation`` (float, default 0.1) — annealed KL term
          (used only during training; ignored at inference).
    """

    def __init__(self, model: Any, config: Optional[dict] = None) -> None:
        super().__init__(model=model, config=config)
        self.reject_threshold: float = float(self.config.get("reject_threshold", 0.5))
        self.kl_regularisation: float = float(self.config.get("kl_regularisation", 0.1))

    # ------------------------------------------------------------------
    def calibrate(self, calibration_data: Any) -> CalibrationResult:
        """Anneal the KL regularisation coefficient on the calibration set.

        We do not modify the model here — we simply record the empirical
        calibration quality and store the rejection threshold that achieves
        the target false-reject rate.
        """
        all_alphas, all_labels = [], []
        for batch in calibration_data:
            x, y = batch
            evidence = self._compute_evidence(x)
            alpha = evidence + 1.0
            all_alphas.append(np.atleast_2d(alpha))
            all_labels.append(np.atleast_1d(np.asarray(y)))

        alphas = np.concatenate(all_alphas, axis=0)
        labels = np.concatenate(all_labels, axis=0)

        # Expected calibration error on the predictive distribution
        S = alphas.sum(axis=-1, keepdims=True)
        probs = alphas / S
        ece = _expected_calibration_error(probs, labels)

        # Total belief = sum(evidence) — high belief = high confidence
        total_belief = alphas.sum(axis=-1) - alphas.shape[-1]
        # Find threshold achieving target false-reject rate of ~5%
        target_false_reject = 0.05
        sorted_belief = np.sort(total_belief)
        threshold_idx = int(np.ceil(target_false_reject * len(sorted_belief)))
        calibrated_threshold = float(sorted_belief[threshold_idx]) if threshold_idx < len(sorted_belief) else 0.5

        self.reject_threshold = calibrated_threshold
        self._calibrated = True
        self._calibration_state = {
            "reject_threshold": self.reject_threshold,
            "ece": ece,
            "mean_total_belief": float(np.mean(total_belief)),
        }

        return CalibrationResult(
            method=self.method_name,
            n_calibration=len(labels),
            metrics={
                "ece": ece,
                "mean_total_belief": float(np.mean(total_belief)),
                "reject_threshold": self.reject_threshold,
            },
            threshold=self.reject_threshold,
            extra=self._calibration_state,
        )

    # ------------------------------------------------------------------
    def predict(self, x: Any) -> UQOutput:
        evidence = self._compute_evidence(x)
        alpha = evidence + 1.0
        S = float(alpha.sum())
        probs = alpha / S

        # Predicted class
        prediction = int(probs.argmax())
        confidence = float(probs.max())

        # Uncertainty decomposition (Sensoy et al. 2018, eq. 5-7)
        # Aleatoric = expected entropy of the categorical
        aleatoric = float(-np.sum(probs * np.log(probs + 1e-12)))

        # Epistemic = Dirichlet uncertainty = (K - 1) / S  where K = n_classes
        K = float(len(probs))
        epistemic = float(K / S)

        # Total belief — high belief = high model confidence
        total_belief = float(alpha.sum() - K)
        reject = bool(total_belief < self.reject_threshold)

        return UQOutput(
            prediction=prediction,
            aleatoric_unc=aleatoric,
            epistemic_unc=epistemic,
            confidence=confidence,
            reject=reject,
            audit={
                "evidence": evidence.tolist(),
                "alpha": alpha.tolist(),
                "S": S,
                "total_belief": total_belief,
                "n_classes": int(K),
            },
        )

    # ------------------------------------------------------------------
    def _compute_evidence(self, x: Any) -> np.ndarray:
        """Compute the evidence vector e ∈ R^C_+."""
        try:
            out = self.model(x)
            if hasattr(out, "detach"):
                arr = out.detach().cpu().numpy()
            else:
                arr = np.asarray(out)
            # The model output should be non-negative evidence; apply softplus if not.
            if (arr < 0).any():
                arr = np.log1p(np.exp(arr))  # softplus
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            return arr
        except Exception:
            import torch
            if isinstance(x, np.ndarray):
                out = self.model(torch.from_numpy(x).float())
                arr = out.detach().cpu().numpy() if hasattr(out, "detach") else np.asarray(out)
                if (arr < 0).any():
                    arr = np.log1p(np.exp(arr))
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                return arr
            raise


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
