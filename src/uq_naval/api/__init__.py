"""FastAPI service layer — exposes the pipeline as a REST + WebSocket API.

The API is the operational interface to the framework: it accepts sensor
frames, runs the full pipeline, and returns decisions with audit trails.
"""
from .main import create_app
from .schemas import (
    SensorInput,
    DecisionResponse,
    PipelineStatus,
    UQMethodList,
)

__all__ = [
    "create_app",
    "SensorInput",
    "DecisionResponse",
    "PipelineStatus",
    "UQMethodList",
]
