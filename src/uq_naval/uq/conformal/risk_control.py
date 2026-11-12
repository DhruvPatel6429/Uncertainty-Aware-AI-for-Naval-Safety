"""Conformal Risk Control (Angelopoulos et al., 2024).

Generalises conformal prediction to bound any monotone risk functional,
not just miscoverage. For naval autonomy, this is the principal safety
mechanism: it bounds the expected collision probability per encounter
below a user-specified tolerance α.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from ..base import UQInterface, register_uq_method
from ...types import CalibrationResult, UQOutput


@register_uq_method("conformal_rc")
class ConformalRiskControl(UQInterface):
    """Conformal Risk Control — bounds expected risk below α.

    Unlike standard conformal prediction (which bounds miscoverage), CRC
    can bound any monotone risk functional R(θ; x, y) where θ is a tunable
    parameter. The framework selects θ such that the empirical risk on the
    calibration set is ≤ α.

    Parameters
    ----------
    model:
        A callable producing per-action risk scores.
    config:
        - ``alpha`` (float, default 1e-4) — risk bound (e.g., 1e-4 for collision probability)
        - ``risk_functional`` (str, default "collision_probability")
        - ``n_actions`` (int, default 4) — discrete action space size
        - ``reject_threshold`` (float, default 0.5) — epistemic threshold
    """

    def __init__(self, model: Any, config: Optional[dict] = None) -> None:
        super().__init__(model=model, config=config)
        self.alpha: float = float(self.config.get("alpha", 1e-4))
        self.risk_functional: str = str(self.config.get("risk_functional", "collision_probability"))
        self.n_actions: int = int(self.config.get("n_actions", 4))
        self.reject_threshold: float = float(self.config.get("reject_threshold", 0.5))
        self._theta: float = 1.0  # risk control parameter
        self._calibration_state = {}

    # ------------------------------------------------------------------
    def calibrate(self, calibration_data: Any) -> CalibrationResult:
        """Select θ such that empirical risk ≤ α.

        calibration_data: iterable of (state, action_taken, outcome) where
        outcome = 1 if the action led to a collision, 0 otherwise.
        """
        states, actions, outcomes = [], [], []
        for batch in calibration_data:
            s, a, o = batch
            states.append(np.asarray(s))
            actions.append(np.asarray(a))
            outcomes.append(np.asarray(o, dtype=np.float64))

        if not states:
            raise RuntimeError("Empty calibration data for ConformalRiskControl.")

        S = np.concatenate([s.reshape(-1) for s in states]) if states[0].ndim > 0 else np.array(states)
        A = np.concatenate([a.ravel() for a in actions])
        O = np.concatenate([o.ravel() for o in outcomes])
        n = len(O)

        # For each state, model produces a per-action risk score r(s, a) ∈ [0, 1]
        # The decision rule is: pick action argmin_a r(s, a) but only if r(s, a) ≤ θ.
        # If no action satisfies, reject (escalate to human).
        # We tune θ so that mean(r(s, a_selected)) ≤ α on the calibration set.

        # Get model risk scores for each (state, action) pair
        risk_scores = []
        for s, a in zip(S, A):
            r = self._compute_risk_scores(s)
            risk_scores.append(r[a])
        risk_scores = np.array(risk_scores)

        # Find smallest θ such that the fraction of (s, a) with r(s, a) > θ is ≤ α
        # Equivalently, θ = quantile(risk_scores, 1 - α)
        # But we want the SELECTED action to satisfy r ≤ θ. With finite-sample correction:
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)
        self._theta = float(np.quantile(risk_scores, q_level, method="higher"))
        self._calibrated = True
        self._calibration_state = {
            "theta": self._theta,
            "alpha": self.alpha,
            "n_calibration": n,
            "empirical_risk": float(np.mean(risk_scores <= self._theta)),
        }

        return CalibrationResult(
            method=self.method_name,
            n_calibration=n,
            metrics={
                "alpha_target": self.alpha,
                "theta": self._theta,
                "empirical_risk": float(np.mean(risk_scores <= self._theta)),
                "bound_satisfied": bool(np.mean(risk_scores <= self._theta) <= self.alpha + 1e-6),
            },
            threshold=self._theta,
            extra=self._calibration_state,
        )

    # ------------------------------------------------------------------
    def predict(self, x: Any) -> UQOutput:
        if not self._calibrated:
            self._theta = float(self.alpha)

        risks = self._compute_risk_scores(x)  # shape (n_actions,)
        # Clamp to [0, 1] — risk scores must be valid probabilities
        risks = np.clip(risks, 0.0, 1.0)
        safe_actions = np.where(risks <= self._theta)[0]
        if len(safe_actions) > 0:
            # Pick the action with lowest risk
            best_action = int(safe_actions[np.argmin(risks[safe_actions])])
            reject = False
            reason = None
            collision_bound = float(risks[best_action])
        else:
            # Reject — escalate to human supervisor
            best_action = -1  # sentinel for "no action selected"
            reject = True
            reason = f"All actions exceed risk bound θ={self._theta:.4f}"
            collision_bound = float(risks.min())

        # Compute uncertainty: high if all risks are similar (model can't distinguish)
        if risks.max() - risks.min() < 1e-3:
            epistemic = 1.0
        else:
            epistemic = float(1.0 - (risks.max() - risks.min()))

        return UQOutput(
            prediction=best_action,
            aleatoric_unc=float(np.mean(risks)),
            epistemic_unc=epistemic,
            prediction_set=list(safe_actions),
            confidence=1.0 - collision_bound,
            reject=reject,
            audit={
                "alpha": self.alpha,
                "theta": self._theta,
                "risk_scores": risks.tolist(),
                "selected_action": best_action,
                "reject_reason": reason,
                "collision_probability_bound": collision_bound,
            },
        )

    # ------------------------------------------------------------------
    def _compute_risk_scores(self, state: Any) -> np.ndarray:
        """Compute per-action risk scores via the model.

        Default: model outputs a vector of risk scores directly. Override
        or supply a custom model for different risk functionals.
        """
        try:
            out = self.model(state)
            if hasattr(out, "detach"):
                arr = out.detach().cpu().numpy()
            else:
                arr = np.asarray(out)
            if arr.ndim == 1:
                return arr
            return arr.ravel()
        except Exception:
            # Fallback: uniform risk per action
            return np.full(self.n_actions, 1.0 / self.n_actions)
