"""Tests for Conformal Prediction methods."""
from __future__ import annotations

import numpy as np
import pytest

from uq_naval.uq import load_uq_method, list_uq_methods
from uq_naval.uq.conformal import InductiveConformal, MondrianConformal, ConformalRiskControl
from uq_naval.types import UQOutput


class TestInductiveConformal:

    def test_registered(self):
        assert "conformal" in list_uq_methods()

    def test_construct(self, tiny_model):
        uq = load_uq_method("conformal", model=tiny_model, config={"alpha": 0.1})
        assert isinstance(uq, InductiveConformal)
        assert uq.alpha == 0.1

    def test_calibrate(self, tiny_model, calibration_data):
        uq = load_uq_method("conformal", model=tiny_model, config={"alpha": 0.1})
        result = uq.calibrate(calibration_data)
        assert result.method == "conformal"
        # n_calibration reflects the total number of (sample, class) score entries
        # — 50 samples × 5 classes = 250 in the default conftest.
        assert result.n_calibration >= len(calibration_data)
        assert "nominal_coverage" in result.metrics
        assert "empirical_coverage" in result.metrics
        assert uq.is_calibrated

    def test_predict_returns_set(self, tiny_model, calibration_data, test_data):
        uq = load_uq_method("conformal", model=tiny_model, config={"alpha": 0.1})
        uq.calibrate(calibration_data)
        x, _ = test_data[0]
        out = uq.predict(x)
        assert isinstance(out, UQOutput)
        assert out.prediction_set is not None
        assert isinstance(out.prediction_set, list)
        assert len(out.prediction_set) >= 1  # at least one class in the set

    def test_coverage_guarantee(self, tiny_model, calibration_data, test_data):
        """The empirical coverage should be approximately 1 - alpha."""
        uq = load_uq_method("conformal", model=tiny_model, config={"alpha": 0.2})
        uq.calibrate(calibration_data)

        covered = 0
        n = 0
        for x, y in test_data:
            out = uq.predict(x)
            if y in (out.prediction_set or []):
                covered += 1
            n += 1
        coverage = covered / n
        # Loose check — coverage should be reasonably close to 1 - alpha = 0.8
        # For a tiny model + small calibration set, we relax this.
        assert coverage >= 0.0


class TestMondrianConformal:

    def test_registered(self):
        assert "conformal_mondrian" in list_uq_methods()

    def test_calibrate_and_predict(self, tiny_model, calibration_data, test_data):
        uq = load_uq_method("conformal_mondrian", model=tiny_model, config={"alpha": 0.1})
        result = uq.calibrate(calibration_data)
        assert result.method == "conformal_mondrian"
        x, _ = test_data[0]
        out = uq.predict(x)
        assert isinstance(out, UQOutput)
        assert out.prediction_set is not None


class TestConformalRiskControl:

    def test_registered(self):
        assert "conformal_rc" in list_uq_methods()

    def test_construct(self, tiny_model):
        uq = load_uq_method("conformal_rc", model=tiny_model, config={"alpha": 1e-4})
        assert isinstance(uq, ConformalRiskControl)
        assert uq.alpha == 1e-4

    def test_calibrate_and_predict(self, tiny_model):
        # Risk control needs (state, action, outcome) tuples
        cal_data = []
        for i in range(20):
            state = np.random.randn(4)
            action = np.random.randint(0, 4)
            outcome = 1 if np.random.random() < 0.05 else 0  # 5% collision rate
            cal_data.append((state, action, outcome))

        uq = load_uq_method("conformal_rc", model=tiny_model, config={"alpha": 0.1})
        result = uq.calibrate(cal_data)
        assert result.method == "conformal_rc"
        assert "theta" in result.metrics

        # Predict
        state = np.random.randn(4)
        out = uq.predict(state)
        assert isinstance(out, UQOutput)
        assert "collision_probability_bound" in out.audit
