"""Tests for the FastAPI service (if FastAPI is installed)."""
from __future__ import annotations

import sys
import pytest

# Skip the entire module if fastapi is unavailable
fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from uq_naval.api import create_app


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


class TestAPI:

    def test_status(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["version"] == "1.0.0"
        assert isinstance(data["uq_methods"], list)
        assert len(data["uq_methods"]) >= 4  # at least the 4 main methods

    def test_uq_methods(self, client):
        response = client.get("/uq-methods")
        assert response.status_code == 200
        data = response.json()
        assert "methods" in data
        assert "bayesian" in data["methods"]
        assert "conformal" in data["methods"]
        assert "ensemble" in data["methods"]
        assert "evidential_clf" in data["methods"]

    def test_predict(self, client):
        response = client.post("/predict", json={
            "sensor_type": "eo",
            "frame_data": [0.5] * (32 * 32),
            "frame_shape": [32, 32],
            "own_state": {"lat": 0.0, "lon": 0.0, "heading_deg": 0.0, "speed_kn": 10.0},
        })
        assert response.status_code == 200
        data = response.json()
        assert data["action"] in ("EXECUTE", "REJECT_HUMAN", "EMERGENCY_FALLBACK")
        assert "audit_trail" in data
        assert "collision_probability_bound" in data

    def test_audit_log(self, client):
        # First make a prediction so the pipeline is initialised
        client.post("/predict", json={
            "sensor_type": "eo",
            "frame_data": [0.5] * (32 * 32),
            "frame_shape": [32, 32],
            "own_state": {"lat": 0.0, "lon": 0.0, "heading_deg": 0.0, "speed_kn": 10.0},
        })
        response = client.get("/audit-log")
        assert response.status_code == 200
        data = response.json()
        assert "entries" in data
