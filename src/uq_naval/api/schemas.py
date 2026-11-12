"""Pydantic schemas for the FastAPI service."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class SensorInput(BaseModel):
    """Sensor input for a single pipeline invocation."""
    sensor_type: str = Field(default="eo", description="eo | radar | sonar | ais")
    frame_data: list[float] = Field(
        default_factory=list,
        description="Flattened sensor frame (e.g., 3*32*32 = 3072 floats for a 32x32 RGB image)"
    )
    frame_shape: tuple[int, ...] = Field(default=(32, 32), description="Shape of the frame")
    own_state: dict[str, float] = Field(
        default_factory=lambda: {"lat": 0.0, "lon": 0.0, "heading_deg": 0.0, "speed_kn": 10.0},
        description="Own-ship state",
    )


class DecisionResponse(BaseModel):
    """Pipeline decision output."""
    action: str = Field(description="EXECUTE | REJECT_HUMAN | EMERGENCY_FALLBACK")
    collision_probability_bound: float = Field(default=0.0)
    reject_reason: Optional[str] = None
    audit_trail: dict[str, Any] = Field(default_factory=dict)
    n_detections: int = Field(default=0)
    mean_epistemic_uncertainty: float = Field(default=0.0)


class PipelineStatus(BaseModel):
    """Pipeline / service status."""
    status: str = "ok"
    version: str = "1.0.0"
    uq_methods: list[str] = Field(default_factory=list)
    pipeline_loaded: bool = False


class UQMethodList(BaseModel):
    """List of available UQ methods."""
    methods: list[str]
