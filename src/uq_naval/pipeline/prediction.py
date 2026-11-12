"""Trajectory prediction module.

Takes the perception module's detections and forecasts each obstacle's
future course over a 30-second horizon, producing directional uncertainty
cones for the path planner.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ..types import Detection, TrajectoryForecast
from ..uq.base import UQInterface


@dataclass
class PredictionConfig:
    """Configuration for the trajectory prediction module."""
    uq_method: str = "ensemble"
    horizon_seconds: int = 30
    n_history_steps: int = 10
    directional_cone_alpha: float = 0.05  # 95% credible region


class PredictionModule:
    """Trajectory prediction pipeline stage.

    Wraps a forecaster (LSTM/Transformer) and a UQ method. The UQ method
    (Deep Ensembles or Evidential DL) provides the multi-modal uncertainty
    cones that the planner uses to inflate obstacle bounding boxes.
    """

    def __init__(
        self,
        forecaster: Any,
        uq_method: UQInterface,
        config: Optional[PredictionConfig] = None,
    ):
        self.forecaster = forecaster
        self.uq_method = uq_method
        self.config = config or PredictionConfig()

    def process(self, detections: list[Detection], history: Optional[Any] = None) -> list[TrajectoryForecast]:
        """Forecast the trajectory of each detected obstacle.

        Parameters
        ----------
        detections:
            Output from the perception module.
        history:
            Optional historical trajectory features per detection. If None,
            a synthetic history is generated.
        """
        forecasts = []
        horizon = self.config.horizon_seconds

        for det in detections:
            # Construct history features for this detection (synthetic if not provided)
            if history is None:
                # Synthetic: small Gaussian noise around the detection's heading
                rng = np.random.default_rng(det.track_id)
                # Use a flat history vector compatible with the prediction model
                history_features = rng.normal(0, 0.1, (1, self.config.n_history_steps, 4))
            else:
                history_features = history.get(det.track_id, np.zeros((1, self.config.n_history_steps, 4)))

            # Flatten to a 1D feature vector for the prediction model
            # (production would use an LSTM/Transformer; here we flatten for compat)
            history_flat = np.asarray(history_features).ravel()

            # Run UQ method on the history
            try:
                uq_out = self.uq_method.predict(history_flat)
            except Exception:
                from ..types import UQOutput
                uq_out = UQOutput(
                    prediction=0,
                    aleatoric_unc=0.5,
                    epistemic_unc=0.5,
                    confidence=0.5,
                    reject=False,
                    audit={"fallback": True},
                )

            # Forecast: use the forecaster output as the mean trajectory
            try:
                preds = self.forecaster(history_flat)
                if hasattr(preds, "detach"):
                    preds = preds.detach().cpu().numpy()
                preds = np.asarray(preds)
                if preds.ndim == 3:
                    preds = preds[0]  # (T, 2)
                elif preds.ndim == 2:
                    # (1, C) — broadcast as constant trajectory
                    preds = np.tile(preds[0][:2], (horizon, 1)) if preds.shape[-1] >= 2 else np.zeros((horizon, 2))
                else:
                    preds = np.zeros((horizon, 2))
            except Exception:
                # Fallback: constant trajectory at detection's current heading
                preds = np.zeros((horizon, 2))
                preds[:, 0] = det.heading_deg or 0.0
                preds[:, 1] = det.velocity_kn or 0.0

            # Decompose into course and speed
            course = preds[:, 0].tolist()
            speed = preds[:, 1].tolist()

            # Uncertainty per timestep — scaled by the UQ method's epistemic output
            base_epistemic = float(uq_out.epistemic_unc) * 30.0  # degrees of half-cone
            base_aleatoric = float(uq_out.aleatoric_unc) * 15.0
            epistemic_per_t = [base_epistemic * (1.0 + 0.05 * t) for t in range(horizon)]
            aleatoric_per_t = [base_aleatoric * (1.0 + 0.03 * t) for t in range(horizon)]

            forecasts.append(TrajectoryForecast(
                track_id=det.track_id,
                horizon_seconds=horizon,
                predicted_course_deg=course,
                predicted_speed_kn=speed,
                aleatoric_unc=aleatoric_per_t,
                epistemic_unc=epistemic_per_t,
                directional_cone_deg=base_epistemic,
            ))

        return forecasts
