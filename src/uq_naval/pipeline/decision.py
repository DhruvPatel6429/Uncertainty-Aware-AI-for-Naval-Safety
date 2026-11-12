"""Decision module — safety-critical capstone of the pipeline.

Implements Conformal Risk Control to bound the probability of unacceptable
outcomes (e.g., collision) below a user-specified threshold α, and triggers
a reject option when epistemic uncertainty exceeds an operational limit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from ..types import Decision, PlannedPath
from ..uq.base import UQInterface


@dataclass
class DecisionConfig:
    """Configuration for the decision module."""
    collision_prob_bound: float = 1e-4       # α target
    epistemic_reject_threshold: float = 0.7  # above this → reject
    fallback_action: str = "EMERGENCY_FALLBACK"
    enable_audit_log: bool = True


class DecisionModule:
    """Decision module wrapping Conformal Risk Control.

    The decision module receives the planner's recommended path and decides
    whether to:

    1. EXECUTE — autonomously execute the planned path
    2. REJECT_HUMAN — escalate to a human supervisor with structured
       explanation of uncertainty sources
    3. EMERGENCY_FALLBACK — execute a pre-defined conservative manoeuvre
       (reduce speed to minimum steerage, hold course, alert supervisor)

    The choice depends on the conformal risk bound and the pipeline's
    aggregated epistemic uncertainty.
    """

    def __init__(
        self,
        risk_controller: Optional[UQInterface] = None,
        config: Optional[DecisionConfig] = None,
    ):
        self.risk_controller = risk_controller
        self.config = config or DecisionConfig()
        self._audit_log: list[dict] = []

    def process(
        self,
        planned_path: PlannedPath,
        perception_uncertainty: float = 0.0,
        prediction_uncertainty: float = 0.0,
    ) -> Decision:
        """Decide whether to execute, escalate, or trigger fallback.

        Parameters
        ----------
        planned_path:
            Output from the planning module.
        perception_uncertainty:
            Mean epistemic uncertainty from the perception stage.
        prediction_uncertainty:
            Mean epistemic uncertainty from the prediction stage.
        """
        # Aggregate pipeline epistemic uncertainty
        total_epistemic = 0.4 * perception_uncertainty + 0.6 * prediction_uncertainty

        # Get conformal risk bound, if a risk controller is configured
        collision_bound = 0.0
        if self.risk_controller is not None:
            # Use the risk controller to evaluate the planned path
            # The "state" passed to the risk controller is a feature vector
            # summarising the planned path
            state = self._summarise_path(planned_path)
            risk_out = self.risk_controller.predict(state)
            collision_bound = float(risk_out.audit.get("collision_probability_bound", 0.0))

        # Decision logic — three branches
        if total_epistemic > self.config.epistemic_reject_threshold:
            action = "REJECT_HUMAN"
            reason = (
                f"Epistemic uncertainty {total_epistemic:.3f} exceeds threshold "
                f"{self.config.epistemic_reject_threshold:.3f}"
            )
        elif collision_bound > self.config.collision_prob_bound:
            action = "EMERGENCY_FALLBACK"
            reason = (
                f"Conformal risk bound {collision_bound:.2e} exceeds target "
                f"{self.config.collision_prob_bound:.2e}"
            )
        else:
            action = "EXECUTE"
            reason = None

        # Build audit trail entry
        audit_entry = {
            "action": action,
            "reason": reason,
            "perception_uncertainty": float(perception_uncertainty),
            "prediction_uncertainty": float(prediction_uncertainty),
            "total_epistemic": float(total_epistemic),
            "collision_prob_bound": float(collision_bound),
            "path_cost": float(planned_path.cost),
            "colregs_compliance": float(planned_path.colregs_compliance),
            "uncertainty_margin_m": float(planned_path.uncertainty_margin_m),
        }
        if self.config.enable_audit_log:
            self._audit_log.append(audit_entry)

        return Decision(
            action=action,
            recommended_path=planned_path if action == "EXECUTE" else None,
            collision_probability_bound=collision_bound,
            reject_reason=reason,
            audit_trail=audit_entry,
        )

    # ------------------------------------------------------------------
    @property
    def audit_log(self) -> list[dict]:
        return self._audit_log

    def clear_audit_log(self) -> None:
        self._audit_log.clear()

    def _summarise_path(self, path: PlannedPath) -> np.ndarray:
        """Compress a planned path into a feature vector for the risk controller."""
        if not path.waypoints_latlon:
            return np.zeros(4)
        wps = np.array(path.waypoints_latlon)
        speeds = np.array(path.speeds_kn)
        return np.array([
            float(np.std(wps[:, 0])),  # lat spread
            float(np.std(wps[:, 1])),  # lon spread
            float(np.mean(speeds)),
            float(path.uncertainty_margin_m),
        ])
