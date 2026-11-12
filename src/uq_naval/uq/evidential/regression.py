"""Evidential regression (Amini et al., 2020).

The network outputs the parameters (γ, ν, α, β) of a Normal-Inverse-Gamma
distribution over (mean, variance) pairs. Both aleatoric and epistemic
uncertainty are obtainable in a single forward pass.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..base import UQInterface, register_uq_method
from ...types import CalibrationResult, UQOutput


@register_uq_method("evidential_reg")
class EvidentialRegressor(UQInterface):
    """Evidential regression via Normal-Inverse-Gamma outputs.

    Parameters
    ----------
    model:
        A callable whose output is a 4-tuple (gamma, nu, alpha, beta):
        - gamma: predicted mean
        - nu:    virtual evidence count
        - alpha: prior strength
        - beta:  scale
    config:
        - ``reject_threshold`` (float, default 0.5) — epistemic threshold
        - ``coeff_reg`` (float, default 0.01) — regularisation coefficient
    """

    def __init__(self, model: Any, config: Optional[dict] = None) -> None:
        super().__init__(model=model, config=config)
        self.reject_threshold: float = float(self.config.get("reject_threshold", 0.5))
        self.coeff_reg: float = float(self.config.get("coeff_reg", 0.01))

    # ------------------------------------------------------------------
    def calibrate(self, calibration_data: Any) -> CalibrationResult:
        """Estimate calibration quality on a held-out set."""
        all_preds, all_targets = [], []
        for batch in calibration_data:
            x, y = batch
            gamma, _, _, _ = self._compute_nig(x)
            all_preds.append(np.atleast_1d(np.asarray(gamma)).ravel())
            all_targets.append(np.atleast_1d(np.asarray(y)).ravel())

        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)

        # RMSE and Brier-style calibration
        rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
        bias = float(np.mean(preds - targets))

        self._calibrated = True
        self._calibration_state = {"rmse": rmse, "bias": bias}

        return CalibrationResult(
            method=self.method_name,
            n_calibration=len(targets),
            metrics={"rmse": rmse, "bias": bias},
            extra=self._calibration_state,
        )

    # ------------------------------------------------------------------
    def predict(self, x: Any) -> UQOutput:
        gamma, nu, alpha, beta = self._compute_nig(x)

        # Predictive mean and variance (Amini et al. 2020)
        aleatoric = float(beta / (alpha - 1))  # expected variance
        epistemic = float(beta / ((alpha - 1) * nu))  # epistemic variance

        # Confidence = inverse of total uncertainty
        total = aleatoric + epistemic
        confidence = float(1.0 / (1.0 + total))

        reject = bool(epistemic > self.reject_threshold)

        return UQOutput(
            prediction=float(gamma),
            aleatoric_unc=aleatoric,
            epistemic_unc=epistemic,
            confidence=confidence,
            reject=reject,
            audit={
                "gamma": float(gamma),
                "nu": float(nu),
                "alpha": float(alpha),
                "beta": float(beta),
            },
        )

    # ------------------------------------------------------------------
    def _compute_nig(self, x: Any) -> tuple[float, float, float, float]:
        """Return (gamma, nu, alpha, beta) parameters of the NIG posterior."""
        try:
            out = self.model(x)
            if hasattr(out, "detach"):
                arr = out.detach().cpu().numpy()
            else:
                arr = np.asarray(out)
        except Exception:
            import torch
            if isinstance(x, np.ndarray):
                out = self.model(torch.from_numpy(x).float())
                arr = out.detach().cpu().numpy() if hasattr(out, "detach") else np.asarray(out)
            else:
                raise

        # Expect a 4-vector (gamma, nu, alpha, beta). Apply softplus to ensure positivity
        # for nu, alpha, beta. If shape is (4,) or (1, 4), unpack.
        if arr.ndim == 0:
            arr = arr.reshape(1)
        if arr.size >= 4:
            flat = arr.ravel()[:4]
        else:
            # Pad if model returns fewer than 4
            flat = np.pad(arr.ravel(), (0, 4 - arr.size), constant_values=1.0)

        gamma = float(flat[0])
        nu = float(np.log1p(np.exp(flat[1])))    # softplus
        alpha = float(np.log1p(np.exp(flat[2]))) + 1.0  # alpha > 1
        beta = float(np.log1p(np.exp(flat[3])))

        return gamma, nu, alpha, beta
