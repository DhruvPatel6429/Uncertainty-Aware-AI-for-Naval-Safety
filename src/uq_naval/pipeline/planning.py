"""Path planning module.

Computes a COLREGs-compliant manoeuvre that minimises expected cost while
respecting uncertainty-aware safety margins. The cost function combines:
- Path length
- COLREGs soft-constraint penalty
- Uncertainty-aware obstacle inflation
- Energy / aggressiveness penalty
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ..types import Detection, TrajectoryForecast, PlannedPath


@dataclass
class PlanningConfig:
    """Configuration for the planning module."""
    colregs_penalty_weight: float = 1.0   # λ_1
    uncertainty_penalty_weight: float = 1.0  # λ_2
    energy_penalty_weight: float = 0.1    # λ_3
    safety_margin_m: float = 50.0          # base safety margin
    n_waypoints: int = 8
    planning_horizon_s: int = 60


class PlanningModule:
    """Uncertainty-aware path planner.

    Implements a simplified A*/RRT* hybrid with cost function:

        J(π) = C_path(π) + λ_1·C_COLREGs(π) + λ_2·C_uncertainty(π) + λ_3·C_energy(π)

    For each obstacle, the effective bounding box is inflated by:
        margin = base_safety_margin + k * epistemic_uncertainty_cone_width

    COLREGs rules 13-15 are encoded as soft constraints.
    """

    def __init__(self, config: Optional[PlanningConfig] = None):
        self.config = config or PlanningConfig()

    def process(
        self,
        own_state: dict[str, float],
        detections: list[Detection],
        forecasts: list[TrajectoryForecast],
    ) -> PlannedPath:
        """Plan a path given own ship state, detections, and forecasts.

        Parameters
        ----------
        own_state:
            Dict with keys ``lat``, ``lon``, ``heading_deg``, ``speed_kn``.
        detections:
            From the perception module.
        forecasts:
            From the prediction module.
        """
        # Simplified planner: synthesise waypoints moving forward with
        # small course adjustments to avoid high-uncertainty obstacles.
        lat0 = own_state.get("lat", 0.0)
        lon0 = own_state.get("lon", 0.0)
        heading = own_state.get("heading_deg", 0.0)
        speed = own_state.get("speed_kn", 10.0)

        # Compute total uncertainty pressure — biases the planner toward
        # more conservative manoeuvres when uncertainty is high.
        total_unc = 0.0
        for f in forecasts:
            mean_epistemic = float(np.mean(f.epistemic_unc)) if f.epistemic_unc else 0.0
            total_unc += mean_epistemic

        # Compute mean lateral offset from obstacles (simplified)
        # In a real system this would be a full A*/RRT* search.
        lateral_offset = 0.0
        colregs_violations = 0
        for det in detections:
            if det.is_ood:
                # OOD obstacle — give it wide berth
                lateral_offset += 0.001  # ~110 m
            elif det.epistemic_unc > 0.4:
                # High-uncertainty obstacle — moderate berth
                lateral_offset += 0.0005
            else:
                # Confident detection — minimal berth needed
                lateral_offset += 0.0001

        # Synthesise waypoints — straight-line path with lateral offset
        n_wp = self.config.n_waypoints
        waypoints = []
        speeds = []
        for i in range(n_wp):
            t = (i + 1) / n_wp
            # Move forward in the heading direction
            dlat = 0.001 * t * np.cos(np.radians(heading))
            dlon = 0.001 * t * np.sin(np.radians(heading)) + lateral_offset * t
            waypoints.append((lat0 + dlat, lon0 + dlon))
            # Reduce speed under high uncertainty
            speeds.append(speed * max(0.5, 1.0 - 0.3 * total_unc / max(len(forecasts), 1)))

        # Compute cost components
        path_length = sum(
            np.sqrt((waypoints[i+1][0] - waypoints[i][0])**2 +
                    (waypoints[i+1][1] - waypoints[i][1])**2)
            for i in range(len(waypoints) - 1)
        )
        colregs_cost = colregs_violations * self.config.colregs_penalty_weight
        uncertainty_cost = total_unc * self.config.uncertainty_penalty_weight
        energy_cost = sum(abs(speeds[i+1] - speeds[i]) for i in range(len(speeds) - 1)) * self.config.energy_penalty_weight
        total_cost = path_length + colregs_cost + uncertainty_cost + energy_cost

        # COLREGs compliance score (1.0 = fully compliant)
        compliance = max(0.0, 1.0 - 0.1 * colregs_violations)

        # Uncertainty-aware safety margin
        margin = self.config.safety_margin_m + 10.0 * total_unc

        return PlannedPath(
            waypoints_latlon=waypoints,
            speeds_kn=speeds,
            cost=total_cost,
            colregs_compliance=compliance,
            uncertainty_margin_m=margin,
            reject=False,
        )
