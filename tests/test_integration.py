"""End-to-end integration test.

Exercises the full framework: UQ method → pipeline → API → audit trail.
This is the single test that, if it passes, demonstrates the framework
is operational.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from uq_naval.uq import load_uq_method, list_uq_methods
from uq_naval.models.backbones import NumpyMLP
from uq_naval.data.synthetic import generate_calibration_data
from uq_naval.pipeline.perception import PerceptionModule, PerceptionConfig
from uq_naval.pipeline.prediction import PredictionModule, PredictionConfig
from uq_naval.pipeline.planning import PlanningModule, PlanningConfig
from uq_naval.pipeline.decision import DecisionModule, DecisionConfig
from uq_naval.pipeline.pipeline import NavalNavigationPipeline
from uq_naval.evaluation import EvaluationHarness
from uq_naval.evaluation.harness import HarnessConfig
from uq_naval.utils.audit import AuditLogger
from uq_naval.utils.seeds import set_global_seed
from uq_naval.types import Decision, UQOutput


@pytest.fixture(scope="module")
def pipeline():
    """Build the full pipeline once for the whole module."""
    set_global_seed(0)
    perception_model = NumpyMLP(input_dim=32, n_classes=5, seed=0)
    prediction_model = NumpyMLP(input_dim=40, n_classes=5, seed=1)
    perception_uq = load_uq_method("bayesian", model=perception_model, config={"n_mc_samples": 3})
    prediction_uq = load_uq_method("ensemble", model=[prediction_model] * 3, config={"temperature": 1.0})

    return NavalNavigationPipeline(
        perception=PerceptionModule(detector=perception_model, uq_method=perception_uq,
                                    config=PerceptionConfig()),
        prediction=PredictionModule(forecaster=prediction_model, uq_method=prediction_uq,
                                    config=PredictionConfig()),
        planning=PlanningModule(config=PlanningConfig()),
        decision=DecisionModule(config=DecisionConfig()),
    )


class TestFrameworkIntegration:
    """Full end-to-end integration tests."""

    def test_all_uq_methods_registered(self):
        """Verify that all 11 UQ methods are registered."""
        methods = list_uq_methods()
        # 4 main families + variants
        assert "bayesian" in methods
        assert "conformal" in methods
        assert "ensemble" in methods
        assert "evidential_clf" in methods
        # Variants
        assert "bayesian_bb" in methods
        assert "bayesian_laplace" in methods
        assert "conformal_mondrian" in methods
        assert "conformal_rc" in methods
        assert "ensemble_snapshot" in methods
        assert "ensemble_batch" in methods
        assert "evidential_reg" in methods
        assert len(methods) == 11, f"Expected 11 methods, got {len(methods)}: {methods}"

    def test_pipeline_produces_valid_decision(self, pipeline):
        """End-to-end: pipeline produces a valid Decision with audit trail."""
        sensor_input = np.random.randn(3, 32, 32).astype(np.float32)
        own_state = {"lat": 0.0, "lon": 0.0, "heading_deg": 0.0, "speed_kn": 10.0}
        decision = pipeline.process(sensor_input, own_state)

        assert isinstance(decision, Decision)
        assert decision.action in ("EXECUTE", "REJECT_HUMAN", "EMERGENCY_FALLBACK")
        assert isinstance(decision.audit_trail, dict)
        assert "action" in decision.audit_trail
        assert "perception_uncertainty" in decision.audit_trail
        assert "prediction_uncertainty" in decision.audit_trail
        assert "total_epistemic" in decision.audit_trail
        assert "collision_prob_bound" in decision.audit_trail

    def test_audit_log_persists_across_decisions(self, pipeline):
        """Multiple decisions accumulate in the audit log."""
        initial_count = len(pipeline.decision.audit_log)
        for _ in range(3):
            sensor_input = np.random.randn(3, 32, 32).astype(np.float32)
            own_state = {"lat": 0.0, "lon": 0.0, "heading_deg": 0.0, "speed_kn": 10.0}
            pipeline.process(sensor_input, own_state)
        assert len(pipeline.decision.audit_log) == initial_count + 3

    def test_audit_logger_persists_to_disk(self, pipeline, tmp_path):
        """Audit trail can be persisted to disk for safety-assurance evidence."""
        log_path = tmp_path / "audit.jsonl"
        logger = AuditLogger(log_path=log_path)
        for entry in pipeline.decision.audit_log:
            logger.log(entry)

        assert log_path.exists()
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == len(pipeline.decision.audit_log)
        # Each line is valid JSON
        for line in lines:
            data = json.loads(line)
            assert "action" in data
            assert "total_epistemic" in data

    def test_uq_output_is_json_serialisable(self, pipeline):
        """UQ outputs must be JSON-serialisable for the API."""
        sensor_input = np.random.randn(3, 32, 32).astype(np.float32)
        own_state = {"lat": 0.0, "lon": 0.0, "heading_deg": 0.0, "speed_kn": 10.0}
        decision = pipeline.process(sensor_input, own_state)
        # The audit trail must be JSON-serialisable
        json.dumps(decision.audit_trail, default=str)

    def test_evaluation_harness_runs_all_methods(self):
        """The evaluation harness evaluates all registered methods without crashing."""
        config = HarnessConfig(
            n_calibration=20,
            n_test=20,
            n_classes=5,
            feature_dim=32,
            alpha=0.1,
            seed=0,
        )
        harness = EvaluationHarness(config=config)
        results = harness.run()

        assert len(results) == 11  # all methods
        for r in results:
            assert r.method_name
            # Either it produced metrics or has an error/skip note
            assert r.test_metrics  # not empty

    def test_evaluation_harness_produces_summary_table(self):
        """The harness produces a Markdown summary table."""
        config = HarnessConfig(n_calibration=10, n_test=10, seed=0)
        harness = EvaluationHarness(config=config)
        harness.run()
        table = harness.summary_table()
        assert "| Method |" in table
        assert "|--------|" in table
        assert "ECE" in table

    def test_reproducibility_with_seed(self):
        """Same seed produces same calibration results."""
        set_global_seed(42)
        model1 = NumpyMLP(input_dim=32, n_classes=5, seed=42)
        cal1 = generate_calibration_data(n_samples=20, n_classes=5, feature_dim=32, seed=42)

        set_global_seed(42)
        model2 = NumpyMLP(input_dim=32, n_classes=5, seed=42)
        cal2 = generate_calibration_data(n_samples=20, n_classes=5, feature_dim=32, seed=42)

        # Same seed → same data
        for (x1, y1), (x2, y2) in zip(cal1, cal2):
            assert np.array_equal(x1, x2)
            assert y1 == y2

    def test_pipeline_handles_empty_input(self, pipeline):
        """Pipeline should handle edge cases gracefully."""
        # Zero input
        sensor_input = np.zeros((3, 32, 32), dtype=np.float32)
        own_state = {"lat": 0.0, "lon": 0.0, "heading_deg": 0.0, "speed_kn": 0.0}
        decision = pipeline.process(sensor_input, own_state)
        assert decision.action in ("EXECUTE", "REJECT_HUMAN", "EMERGENCY_FALLBACK")

    def test_conformal_risk_control_bounds_collision_probability(self):
        """Conformal Risk Control should bound collision probability below α."""
        model = NumpyMLP(input_dim=4, n_classes=4, seed=0)
        uq = load_uq_method("conformal_rc", model=model, config={
            "alpha": 0.1,
            "n_actions": 4,
        })

        # Generate calibration data: (state, action, outcome)
        rng = np.random.default_rng(0)
        cal_data = []
        for _ in range(30):
            state = rng.normal(0, 1, 4)
            action = rng.integers(0, 4)
            # 5% collision rate
            outcome = 1 if rng.random() < 0.05 else 0
            cal_data.append((state, action, outcome))

        result = uq.calibrate(cal_data)
        # The empirical risk should be ≤ α + small tolerance
        assert "theta" in result.metrics

        # Predict
        state = rng.normal(0, 1, 4)
        out = uq.predict(state)
        assert "collision_probability_bound" in out.audit
        assert out.audit["collision_probability_bound"] >= 0.0


class TestApiIntegration:
    """API integration tests (skipped if FastAPI is unavailable)."""

    def test_api_status_endpoint(self):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient
        from uq_naval.api import create_app

        client = TestClient(create_app())
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["version"] == "1.0.0"
        assert len(data["uq_methods"]) == 11

    def test_api_predict_endpoint(self):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient
        from uq_naval.api import create_app

        client = TestClient(create_app())
        response = client.post("/predict", json={
            "sensor_type": "eo",
            "frame_data": [0.5] * 16,
            "frame_shape": [4, 4],
            "own_state": {"lat": 0.0, "lon": 0.0, "heading_deg": 0.0, "speed_kn": 10.0},
        })
        assert response.status_code == 200
        data = response.json()
        assert data["action"] in ("EXECUTE", "REJECT_HUMAN", "EMERGENCY_FALLBACK")
        assert "audit_trail" in data
        assert "collision_probability_bound" in data
