"""Tests for the evaluation harness + metrics."""
from __future__ import annotations

import numpy as np
import pytest

from uq_naval.evaluation import (
    expected_calibration_error,
    brier_score,
    empirical_coverage,
    ood_auroc,
    EvaluationHarness,
    ScenarioLibrary,
)
from uq_naval.evaluation.harness import HarnessConfig


class TestMetrics:

    def test_ece_perfect_calibration(self):
        # A perfectly calibrated predictor: confidence = accuracy = 1.0
        probs = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        labels = np.array([0, 0, 0])
        ece = expected_calibration_error(probs, labels)
        assert ece == pytest.approx(0.0, abs=1e-6)

    def test_ece_random_uniform(self):
        # Uniform probabilities → ECE should be non-zero
        probs = np.full((100, 5), 0.2)
        labels = np.random.randint(0, 5, 100)
        ece = expected_calibration_error(probs, labels)
        assert ece > 0.0

    def test_brier_score_perfect(self):
        probs = np.array([[1.0, 0.0], [0.0, 1.0]])
        labels = np.array([0, 1])
        assert brier_score(probs, labels) == pytest.approx(0.0, abs=1e-6)

    def test_brier_score_baseline(self):
        # Uniform prediction: Brier should be close to 0.5 for binary
        probs = np.full((100, 2), 0.5)
        labels = np.random.randint(0, 2, 100)
        bs = brier_score(probs, labels)
        assert 0.4 < bs < 0.6

    def test_empirical_coverage(self):
        sets = [[0, 1], [1], [0, 1, 2]]
        labels = np.array([0, 1, 5])
        # 2 out of 3 covered
        assert empirical_coverage(sets, labels) == pytest.approx(2 / 3)

    def test_ood_auroc_perfect(self):
        # OOD uncertainty always higher than in-dist
        in_dist = np.array([0.1, 0.2, 0.3])
        out_dist = np.array([0.7, 0.8, 0.9])
        assert ood_auroc(in_dist, out_dist) == pytest.approx(1.0)

    def test_ood_auroc_random(self):
        in_dist = np.array([0.5, 0.5, 0.5])
        out_dist = np.array([0.5, 0.5, 0.5])
        assert ood_auroc(in_dist, out_dist) == pytest.approx(0.5)


class TestEvaluationHarness:

    def test_runs_all_methods(self):
        """The harness should evaluate multiple methods without crashing.

        Methods that require unavailable dependencies (e.g., torch for
        Laplace approximation) gracefully report an error and continue
        to the next method rather than crashing the harness.
        """
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
        # Should attempt all 11 methods
        assert len(results) == 11

        # Each result should have a method name
        for r in results:
            assert r.method_name
            assert r.test_metrics  # not empty — either metrics or error

        # At least the numpy-compatible methods should produce results.
        # Methods requiring torch (Laplace) may fail gracefully in environments
        # where torch is unavailable — this is acceptable.
        successful = [r for r in results if r.n_test > 0 or "note" in r.test_metrics]
        assert len(successful) >= 8, (
            f"Expected ≥8 methods to produce results, got {len(successful)}. "
            f"Failures: {[(r.method_name, r.test_metrics.get('error', '')) for r in results if r.n_test == 0 and 'note' not in r.test_metrics]}"
        )

    def test_summary_table(self):
        config = HarnessConfig(n_calibration=10, n_test=10, seed=0)
        harness = EvaluationHarness(config=config)
        harness.run()
        table = harness.summary_table()
        assert isinstance(table, str)
        assert "Method" in table
        assert "ECE" in table

    def test_to_dict(self):
        config = HarnessConfig(n_calibration=10, n_test=10, seed=0)
        harness = EvaluationHarness(config=config)
        harness.run()
        d = harness.to_dict()
        assert "config" in d
        assert "results" in d
        assert isinstance(d["results"], list)


class TestScenarioLibrary:

    def test_list_scenarios(self):
        scenarios = ScenarioLibrary()
        names = scenarios.list_scenarios()
        assert "head_on" in names
        assert "crossing" in names
        assert "ood_stress" in names

    def test_run_scenario(self):
        scenarios = ScenarioLibrary(n_frames=5, seed=0)
        result = scenarios.run_scenario("head_on")
        assert result["scenario"] == "head_on"
        assert result["n_frames"] == 5
        assert len(result["frames"]) == 5
        assert "expected_outcome" in result

    def test_run_all(self):
        scenarios = ScenarioLibrary(n_frames=5, seed=0)
        results = scenarios.run_all()
        assert len(results) == len(scenarios.list_scenarios())
        for r in results:
            assert r["n_frames"] == 5
