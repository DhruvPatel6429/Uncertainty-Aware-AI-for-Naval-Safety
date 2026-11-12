"""End-to-end naval navigation pipeline.

Threads UQ methods through perception → prediction → planning → decision
and exposes a single ``process(sensor_input)`` entry point.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ..types import Decision
from ..uq.base import UQInterface
from .perception import PerceptionModule, PerceptionConfig
from .prediction import PredictionModule, PredictionConfig
from .planning import PlanningModule, PlanningConfig
from .decision import DecisionModule, DecisionConfig


@dataclass
class PipelineConfig:
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    planning: PlanningConfig = field(default_factory=PlanningConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)


class NavalNavigationPipeline:
    """End-to-end uncertainty-aware navigation pipeline.

    The pipeline composes four stages:

    1. **Perception** — Bayesian YOLO/U-Net + conformal score calibration
    2. **Prediction** — Evidential / ensemble trajectory forecasting
    3. **Planning** — UQ-aware A*/RRT* with COLREGs soft constraints
    4. **Decision** — Conformal risk control + reject option

    Each stage's UQ method is injected via configuration, allowing the
    operator to swap methods at runtime via a YAML config change.
    """

    def __init__(
        self,
        perception: PerceptionModule,
        prediction: PredictionModule,
        planning: PlanningModule,
        decision: DecisionModule,
        config: Optional[PipelineConfig] = None,
    ):
        self.perception = perception
        self.prediction = prediction
        self.planning = planning
        self.decision = decision
        self.config = config or PipelineConfig()

    def process(
        self,
        sensor_input: Any,
        own_state: dict[str, float],
        history: Optional[Any] = None,
    ) -> Decision:
        """Run the full pipeline on a single frame of sensor data.

        Parameters
        ----------
        sensor_input:
            Radar / EO / IR / sonar frame.
        own_state:
            Own-ship state — ``lat``, ``lon``, ``heading_deg``, ``speed_kn``.
        history:
            Optional historical trajectory features for the prediction module.

        Returns
        -------
        Decision
            The final decision (EXECUTE / REJECT_HUMAN / EMERGENCY_FALLBACK)
            with audit trail.
        """
        # Stage 1 — Perception
        detections = self.perception.process(sensor_input)
        perception_unc = float(np.mean([d.epistemic_unc for d in detections])) if detections else 0.0

        # Stage 2 — Trajectory prediction
        forecasts = self.prediction.process(detections, history=history)
        prediction_unc = (
            float(np.mean([np.mean(f.epistemic_unc) for f in forecasts]))
            if forecasts else 0.0
        )

        # Stage 3 — Path planning
        planned_path = self.planning.process(own_state, detections, forecasts)

        # Stage 4 — Decision
        decision = self.decision.process(
            planned_path=planned_path,
            perception_uncertainty=perception_unc,
            prediction_uncertainty=prediction_unc,
        )

        return decision


# Late import to avoid circular dependency
import numpy as np
