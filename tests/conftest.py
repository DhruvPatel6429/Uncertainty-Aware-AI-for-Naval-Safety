"""Pytest configuration — shared fixtures."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make src importable without installation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


@pytest.fixture
def seed():
    """Global deterministic seed."""
    return 0


@pytest.fixture
def tiny_model():
    """Tiny numpy MLP for fast tests."""
    from uq_naval.models.backbones import NumpyMLP
    return NumpyMLP(input_dim=32, n_classes=5, seed=0)


@pytest.fixture
def calibration_data():
    """Small synthetic calibration set."""
    from uq_naval.data.synthetic import generate_calibration_data
    return generate_calibration_data(n_samples=50, n_classes=5, feature_dim=32, seed=0)


@pytest.fixture
def test_data():
    """Small synthetic test set."""
    from uq_naval.data.synthetic import generate_calibration_data
    return generate_calibration_data(n_samples=50, n_classes=5, feature_dim=32, seed=999)
