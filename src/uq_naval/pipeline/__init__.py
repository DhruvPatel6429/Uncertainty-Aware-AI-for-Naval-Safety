"""Pipeline stages — Perception → Prediction → Planning → Decision.

Each stage consumes the previous stage's output and produces structured
observations that flow through the pipeline. UQ methods are injected
externally via configuration.
"""
from .perception import PerceptionModule
from .prediction import PredictionModule
from .planning import PlanningModule
from .decision import DecisionModule
from .pipeline import NavalNavigationPipeline

__all__ = [
    "PerceptionModule",
    "PredictionModule",
    "PlanningModule",
    "DecisionModule",
    "NavalNavigationPipeline",
]
