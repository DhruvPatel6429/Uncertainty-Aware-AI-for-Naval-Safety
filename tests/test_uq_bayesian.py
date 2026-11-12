"""Tests for the Bayesian DL UQ methods."""
from __future__ import annotations

import numpy as np
import pytest

from uq_naval.uq import load_uq_method, list_uq_methods
from uq_naval.uq.base import UQInterface
from uq_naval.uq.bayesian import MCDropoutBayesian, LaplaceBayesian, BayesByBackprop
from uq_naval.types import UQOutput


class TestMCDropoutBayesian:

    def test_registered(self):
        assert "bayesian" in list_uq_methods()

    def test_construct(self, tiny_model):
        uq = load_uq_method("bayesian", model=tiny_model, config={"n_mc_samples": 3})
        assert isinstance(uq, MCDropoutBayesian)
        assert uq.method_name == "bayesian"
        assert uq.n_mc_samples == 3

    def test_calibrate(self, tiny_model, calibration_data):
        uq = load_uq_method("bayesian", model=tiny_model, config={"n_mc_samples": 3})
        result = uq.calibrate(calibration_data)
        assert result.method == "bayesian"
        assert result.n_calibration == 50
        assert "ece_before" in result.metrics
        assert "ece_after" in result.metrics
        assert "temperature" in result.metrics
        assert uq.is_calibrated

    def test_predict_returns_uq_output(self, tiny_model, calibration_data, test_data):
        uq = load_uq_method("bayesian", model=tiny_model, config={"n_mc_samples": 3})
        uq.calibrate(calibration_data)
        x, _ = test_data[0]
        out = uq.predict(x)
        assert isinstance(out, UQOutput)
        assert isinstance(out.prediction, (int, np.integer)) or hasattr(out.prediction, "__len__")
        assert 0.0 <= out.confidence <= 1.0
        assert out.epistemic_unc >= 0.0
        assert out.aleatoric_unc >= 0.0
        assert isinstance(out.reject, bool)

    def test_audit_trail(self, tiny_model, calibration_data, test_data):
        uq = load_uq_method("bayesian", model=tiny_model, config={"n_mc_samples": 3})
        uq.calibrate(calibration_data)
        x, _ = test_data[0]
        out = uq.predict(x)
        assert "n_mc_samples" in out.audit
        assert "temperature" in out.audit
        assert out.audit["n_mc_samples"] == 3

    def test_to_dict_serialisable(self, tiny_model, calibration_data, test_data):
        uq = load_uq_method("bayesian", model=tiny_model, config={"n_mc_samples": 3})
        uq.calibrate(calibration_data)
        x, _ = test_data[0]
        out = uq.predict(x)
        d = out.to_dict()
        # Should be JSON-serialisable
        import json
        json.dumps(d)


class TestLaplaceBayesian:

    def test_registered(self):
        assert "bayesian_laplace" in list_uq_methods()

    def test_construct(self, tiny_model):
        uq = load_uq_method("bayesian_laplace", model=tiny_model, config={"hessian_samples": 10})
        assert isinstance(uq, LaplaceBayesian)
        assert uq.hessian_samples == 10

    def test_calibrate_and_predict(self, tiny_model, calibration_data, test_data):
        uq = load_uq_method("bayesian_laplace", model=tiny_model, config={"hessian_samples": 5})
        # The Laplace approximation may skip if no torch model is available;
        # we still expect a CalibrationResult.
        try:
            result = uq.calibrate(calibration_data)
            assert result.method == "bayesian_laplace"
        except RuntimeError:
            # Expected when model is not a torch nn.Module
            pytest.skip("Laplace requires torch nn.Module with parameters")

        x, _ = test_data[0]
        out = uq.predict(x)
        assert isinstance(out, UQOutput)


class TestBayesByBackprop:

    def test_registered(self):
        assert "bayesian_bb" in list_uq_methods()

    def test_construct(self, tiny_model):
        uq = load_uq_method("bayesian_bb", model=tiny_model, config={"n_mc_samples": 2})
        assert isinstance(uq, BayesByBackprop)
        assert uq.n_mc_samples == 2
