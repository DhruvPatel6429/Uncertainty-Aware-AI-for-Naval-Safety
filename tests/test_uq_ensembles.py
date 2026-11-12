"""Tests for Deep Ensemble methods."""
from __future__ import annotations

import numpy as np
import pytest

from uq_naval.uq import load_uq_method, list_uq_methods
from uq_naval.uq.ensembles import DeepEnsemble, SnapshotEnsemble, BatchEnsemble
from uq_naval.types import UQOutput


class TestDeepEnsemble:

    def test_registered(self):
        assert "ensemble" in list_uq_methods()

    def test_construct_with_list(self, tiny_model):
        # Pass a list of M models
        models = [tiny_model] * 3
        uq = load_uq_method("ensemble", model=models, config={"temperature": 1.0})
        assert isinstance(uq, DeepEnsemble)

    def test_calibrate(self, tiny_model, calibration_data):
        models = [tiny_model] * 3
        uq = load_uq_method("ensemble", model=models)
        result = uq.calibrate(calibration_data)
        assert result.method == "ensemble"
        assert "temperature" in result.metrics
        assert uq.is_calibrated

    def test_predict(self, tiny_model, calibration_data, test_data):
        models = [tiny_model] * 3
        uq = load_uq_method("ensemble", model=models)
        uq.calibrate(calibration_data)
        x, _ = test_data[0]
        out = uq.predict(x)
        assert isinstance(out, UQOutput)
        assert "n_members" in out.audit
        assert out.audit["n_members"] == 3


class TestSnapshotEnsemble:

    def test_registered(self):
        assert "ensemble_snapshot" in list_uq_methods()

    def test_construct(self, tiny_model):
        models = [tiny_model] * 2
        uq = load_uq_method("ensemble_snapshot", model=models)
        assert isinstance(uq, SnapshotEnsemble)
        assert uq.method_name == "ensemble_snapshot"


class TestBatchEnsemble:

    def test_registered(self):
        assert "ensemble_batch" in list_uq_methods()

    def test_construct(self, tiny_model):
        uq = load_uq_method("ensemble_batch", model=tiny_model, config={"n_members": 4})
        assert isinstance(uq, BatchEnsemble)
        assert uq.n_members == 4
