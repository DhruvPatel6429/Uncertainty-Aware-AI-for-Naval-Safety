"""Shared types for the UQ-AI Naval framework.

These types form the contract between UQ methods, pipeline stages, and the
evaluation harness. Every UQ method returns a UQOutput, every pipeline stage
consumes and produces PipelineObservation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# UQ output — the universal contract
# ---------------------------------------------------------------------------

class UncertaintyType(str, Enum):
    """Categorisation of an uncertainty source for audit-trail purposes."""
    ALEATORIC = "aleatoric"      # Irreducible data noise
    EPISTEMIC = "epistemic"      # Model ignorance (reducible with data)
    DISTRIBUTIONAL = "distributional"  # OOD / out-of-support


@dataclass
class UQOutput:
    """Universal output of every UQ method.

    Attributes
    ----------
    prediction:
        Point prediction. Type depends on the task: np.ndarray for classification
        (probabilities), float for regression, list for structured prediction.
    aleatoric_unc:
        Irreducible uncertainty (data noise). Same shape as `prediction` or a scalar.
    epistemic_unc:
        Model uncertainty (reducible with data). Same shape as `aleatoric_unc`.
    prediction_set:
        Conformal prediction set, when applicable. None for non-conformal methods.
    confidence:
        Calibrated confidence in [0, 1]. For classification, this is the
        conformal-calibrated max probability; for regression, 1 - epistemic_unc.
    reject:
        True if the UQ method recommends deferring to a human supervisor
        (epistemic uncertainty exceeds operational threshold).
    audit:
        Dict of method-specific audit fields (e.g., per-class Dirichlet
        concentrations, ensemble member agreement, MC sample variance).
    """
    prediction: Any
    aleatoric_unc: float | np.ndarray
    epistemic_unc: float | np.ndarray
    prediction_set: Optional[Sequence[Any]] = None
    confidence: float = 0.0
    reject: bool = False
    audit: dict[str, Any] = field(default_factory=dict)

    def total_uncertainty(self) -> float | np.ndarray:
        """Total uncertainty = aleatoric + epistemic."""
        return np.add(self.aleatoric_unc, self.epistemic_unc)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to dict (for JSON over API / audit-log)."""
        def _to_jsonable(x: Any) -> Any:
            if isinstance(x, np.ndarray):
                return x.tolist()
            if isinstance(x, (np.floating, np.integer)):
                return x.item()
            return x
        return {
            "prediction": _to_jsonable(self.prediction),
            "aleatoric_unc": _to_jsonable(self.aleatoric_unc),
            "epistemic_unc": _to_jsonable(self.epistemic_unc),
            "prediction_set": _to_jsonable(self.prediction_set),
            "confidence": float(self.confidence),
            "reject": bool(self.reject),
            "audit": {k: _to_jsonable(v) for k, v in self.audit.items()},
        }


# ---------------------------------------------------------------------------
# Pipeline observation — passed between stages
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """A single obstacle detection from the perception stage."""
    track_id: int
    label: str
    bbox_xyxy: tuple[float, float, float, float]  # (x1, y1, x2, y2) in image px
    position_latlon: Optional[tuple[float, float]] = None  # (lat, lon) if georeferenced
    velocity_kn: Optional[float] = None  # knots
    heading_deg: Optional[float] = None  # 0 = north, 90 = east
    confidence: float = 0.0  # calibrated confidence
    epistemic_unc: float = 0.0
    is_ood: bool = False


@dataclass
class TrajectoryForecast:
    """Multi-horizon trajectory forecast for an obstacle."""
    track_id: int
    horizon_seconds: int
    predicted_course_deg: list[float]  # per-timestep heading predictions
    predicted_speed_kn: list[float]
    aleatoric_unc: list[float]
    epistemic_unc: list[float]
    directional_cone_deg: float  # 95% credible region half-width


@dataclass
class PlannedPath:
    """A path plan from the planning stage."""
    waypoints_latlon: list[tuple[float, float]]
    speeds_kn: list[float]
    cost: float
    colregs_compliance: float  # 1.0 = fully compliant
    uncertainty_margin_m: float
    reject: bool = False


@dataclass
class Decision:
    """The decision module's final output."""
    action: str  # "EXECUTE" | "REJECT_HUMAN" | "EMERGENCY_FALLBACK"
    recommended_path: Optional[PlannedPath] = None
    collision_probability_bound: float = 0.0  # conformal risk bound
    reject_reason: Optional[str] = None
    audit_trail: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Calibration result
# ---------------------------------------------------------------------------

@dataclass
class CalibrationResult:
    """Returned by UQMethod.calibrate()."""
    method: str
    n_calibration: int
    metrics: dict[str, float]  # ECE, Brier, coverage, etc.
    threshold: Optional[float] = None  # conformal quantile, if applicable
    extra: dict[str, Any] = field(default_factory=dict)
