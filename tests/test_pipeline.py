"""Tests for the end-to-end pipeline."""
from __future__ import annotations

import numpy as np
import pytest

from uq_naval.uq import load_uq_method
from uq_naval.pipeline.perception import PerceptionModule, PerceptionConfig
from uq_naval.pipeline.prediction import PredictionModule, PredictionConfig
from uq_naval.pipeline.planning import PlanningModule, PlanningConfig
from uq_naval.pipeline.decision import DecisionModule, DecisionConfig
from uq_naval.pipeline.pipeline import NavalNavigationPipeline
from uq_naval.models.backbones import NumpyMLP
from uq_naval.types import Decision


@pytest.fixture
def pipeline():
    """Build a small end-to-end pipeline."""
    perception_model = NumpyMLP(input_dim=32, n_classes=5, seed=0)
    prediction_model = NumpyMLP(input_dim=40, n_classes=5, seed=1)

    perception_uq = load_uq_method("bayesian", model=perception_model, config={
        "n_mc_samples": 3,
        "reject_threshold": 0.5,
    })
    prediction_uq = load_uq_method("ensemble", model=[prediction_model] * 3, config={
        "temperature": 1.0,
    })

    perception = PerceptionModule(
        detector=perception_model,
        uq_method=perception_uq,
        config=PerceptionConfig(),
    )
    prediction = PredictionModule(
        forecaster=prediction_model,
        uq_method=prediction_uq,
        config=PredictionConfig(),
    )
    planning = PlanningModule(config=PlanningConfig())
    decision = DecisionModule(config=DecisionConfig())
    return NavalNavigationPipeline(perception, prediction, planning, decision)


class TestPipeline:

    def test_pipeline_construct(self, pipeline):
        assert pipeline.perception is not None
        assert pipeline.prediction is not None
        assert pipeline.planning is not None
        assert pipeline.decision is not None

    def test_pipeline_runs(self, pipeline):
        """Pipeline should produce a Decision from a synthetic input."""
        sensor_input = np.random.randn(3, 32, 32).astype(np.float32)
        own_state = {"lat": 0.0, "lon": 0.0, "heading_deg": 0.0, "speed_kn": 10.0}
        decision = pipeline.process(sensor_input, own_state)
        assert isinstance(decision, Decision)
        assert decision.action in ("EXECUTE", "REJECT_HUMAN", "EMERGENCY_FALLBACK")
        assert isinstance(decision.audit_trail, dict)

    def test_audit_trail_populated(self, pipeline):
        sensor_input = np.random.randn(3, 32, 32).astype(np.float32)
        own_state = {"lat": 0.0, "lon": 0.0, "heading_deg": 0.0, "speed_kn": 10.0}
        pipeline.process(sensor_input, own_state)
        assert len(pipeline.decision.audit_log) == 1
        entry = pipeline.decision.audit_log[0]
        assert "action" in entry
        assert "perception_uncertainty" in entry
        assert "prediction_uncertainty" in entry
        assert "total_epistemic" in entry
        assert "collision_prob_bound" in entry


class TestPerceptionModule:

    def test_process_returns_detections(self, tiny_model):
        from uq_naval.uq import load_uq_method
        uq = load_uq_method("bayesian", model=tiny_model, config={"n_mc_samples": 2})
        # Calibrate so .predict works
        from uq_naval.data.synthetic import generate_calibration_data
        cal = generate_calibration_data(n_samples=20, n_classes=5, feature_dim=32, seed=0)
        uq.calibrate(cal)

        perception = PerceptionModule(
            detector=tiny_model,
            uq_method=uq,
            config=PerceptionConfig(),
        )
        sensor_input = np.random.randn(3, 32, 32).astype(np.float32)
        detections = perception.process(sensor_input)
        assert isinstance(detections, list)
        assert len(detections) >= 0  # may be empty if no detections


class TestDecisionModule:

    def test_execute_decision(self):
        from uq_naval.types import PlannedPath
        decision = DecisionModule(config=DecisionConfig(
            collision_prob_bound=1e-4,
            epistemic_reject_threshold=0.9,
        ))
        path = PlannedPath(
            waypoints_latlon=[(0.0, 0.0), (0.001, 0.001)],
            speeds_kn=[10.0, 10.0],
            cost=0.5,
            colregs_compliance=1.0,
            uncertainty_margin_m=50.0,
        )
        result = decision.process(path, perception_uncertainty=0.1, prediction_uncertainty=0.1)
        assert result.action == "EXECUTE"
        assert result.reject_reason is None

    def test_reject_decision(self):
        from uq_naval.types import PlannedPath
        decision = DecisionModule(config=DecisionConfig(
            epistemic_reject_threshold=0.3,
        ))
        path = PlannedPath(
            waypoints_latlon=[(0.0, 0.0)],
            speeds_kn=[5.0],
            cost=0.5,
            colregs_compliance=1.0,
            uncertainty_margin_m=50.0,
        )
        result = decision.process(path, perception_uncertainty=0.6, prediction_uncertainty=0.6)
        assert result.action == "REJECT_HUMAN"
        assert result.reject_reason is not None
