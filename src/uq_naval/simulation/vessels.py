"""Vessel kinematics model — realistic ship dynamics.

Models 6 vessel classes with distinct manoeuvring characteristics based on
publicly available naval architecture references:

- Cargo vessel: large, slow turns, high inertia
- Tanker: very large, very slow turns, very high inertia
- Passenger vessel: medium, smooth predictable course
- Fishing vessel: small, erratic manoeuvres, variable speed
- Naval vessel: medium, fast manoeuvres, varied speed
- Small craft: very small, very fast manoeuvres

Each vessel is simulated as a kinematic model with:
- Position (lat, lon) in degrees
- Heading (deg, 0 = north, clockwise)
- Speed (knots)
- Turn rate (deg/s)
- Acceleration limit (kn/s)

The model is intentionally simple (point-mass kinematics with bounded turn
rate and acceleration) — sufficient for testing perception, prediction, and
planning modules without requiring full hydrodynamic simulation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


# ───────────────────────────────────────────────────────────
# Vessel classes
# ───────────────────────────────────────────────────────────

class VesselClass(str, Enum):
    """Enumeration of vessel classes with distinct kinematic profiles."""
    CARGO = "cargo"
    TANKER = "tanker"
    PASSENGER = "passenger"
    FISHING = "fishing"
    NAVAL = "naval"
    SMALL_CRAFT = "small_craft"


@dataclass
class VesselKinematics:
    """Kinematic parameters for a vessel class.

    Values are approximate operational ranges, not exact specifications.
    They are sufficient for scenario generation and pipeline testing.
    """
    name: str
    length_m: float                # overall length (m)
    beam_m: float                   # beam (m)
    max_speed_kn: float             # maximum sustained speed (kn)
    cruise_speed_kn: float          # typical cruising speed (kn)
    max_turn_rate_dps: float        # maximum turn rate (deg/s)
    typical_turn_rate_dps: float    # typical turn rate in normal operation (deg/s)
    accel_kn_per_s: float           # acceleration limit (kn/s)
    decel_kn_per_s: float           # deceleration limit (kn/s)
    color: str                      # display colour for visualisation


VESSEL_CLASSES: dict[VesselClass, VesselKinematics] = {
    VesselClass.CARGO: VesselKinematics(
        name="Cargo Vessel",
        length_m=200, beam_m=32,
        max_speed_kn=24, cruise_speed_kn=18,
        max_turn_rate_dps=0.8, typical_turn_rate_dps=0.3,
        accel_kn_per_s=0.05, decel_kn_per_s=0.10,
        color="#0F2845",
    ),
    VesselClass.TANKER: VesselKinematics(
        name="Tanker",
        length_m=250, beam_m=44,
        max_speed_kn=16, cruise_speed_kn=12,
        max_turn_rate_dps=0.5, typical_turn_rate_dps=0.2,
        accel_kn_per_s=0.03, decel_kn_per_s=0.06,
        color="#1E4275",
    ),
    VesselClass.PASSENGER: VesselKinematics(
        name="Passenger Vessel",
        length_m=180, beam_m=28,
        max_speed_kn=22, cruise_speed_kn=18,
        max_turn_rate_dps=1.5, typical_turn_rate_dps=0.5,
        accel_kn_per_s=0.10, decel_kn_per_s=0.15,
        color="#3F8A5E",
    ),
    VesselClass.FISHING: VesselKinematics(
        name="Fishing Vessel",
        length_m=30, beam_m=8,
        max_speed_kn=12, cruise_speed_kn=8,
        max_turn_rate_dps=5.0, typical_turn_rate_dps=1.5,
        accel_kn_per_s=0.30, decel_kn_per_s=0.40,
        color="#C9963A",
    ),
    VesselClass.NAVAL: VesselKinematics(
        name="Naval Vessel",
        length_m=120, beam_m=15,
        max_speed_kn=30, cruise_speed_kn=18,
        max_turn_rate_dps=3.0, typical_turn_rate_dps=1.0,
        accel_kn_per_s=0.20, decel_kn_per_s=0.30,
        color="#B5443A",
    ),
    VesselClass.SMALL_CRAFT: VesselKinematics(
        name="Small Craft",
        length_m=8, beam_m=3,
        max_speed_kn=35, cruise_speed_kn=22,
        max_turn_rate_dps=10.0, typical_turn_rate_dps=3.0,
        accel_kn_per_s=0.50, decel_kn_per_s=0.60,
        color="#E0A53D",
    ),
}


# ───────────────────────────────────────────────────────────
# Vessel state + dynamics
# ───────────────────────────────────────────────────────────

@dataclass
class VesselState:
    """Instantaneous state of a vessel."""
    time_s: float
    lat: float                  # degrees
    lon: float                  # degrees
    heading_deg: float          # 0 = north, clockwise
    speed_kn: float             # knots
    turn_rate_dps: float = 0.0  # current turn rate (deg/s)


class Vessel:
    """A simulated vessel with realistic kinematics.

    The vessel follows a kinematic model:
        lat(t+dt) = lat(t) + speed * cos(heading) * dt
        lon(t+dt) = lon(t) + speed * sin(heading) * dt
        heading(t+dt) = heading(t) + turn_rate * dt
        speed(t+dt) = speed(t) + accel * dt  (clamped to [0, max_speed])
    """

    # Earth-related constants
    METERS_PER_DEG_LAT = 111_320.0   # ≈ 111.32 km per degree of latitude
    # Meters per degree of longitude varies with latitude; we use a fixed reference
    # since scenario scales are small (< 50 km).

    def __init__(
        self,
        vessel_id: str,
        vessel_class: VesselClass,
        initial_state: VesselState,
        reference_lat: float = 13.0,  # ~ Visakhapatnam latitude
    ):
        self.vessel_id = vessel_id
        self.vessel_class = vessel_class
        self.kinematics = VESSEL_CLASSES[vessel_class]
        self.state = initial_state
        self.reference_lat = reference_lat
        self.history: list[VesselState] = [initial_state]
        # Planned manoeuvres (time, target_heading, target_speed)
        self._manoeuvres: list[tuple[float, float, float]] = []

    @property
    def current_state(self) -> VesselState:
        return self.history[-1]

    @property
    def position_m(self) -> tuple[float, float]:
        """Position in metres relative to scenario origin (lat=reference_lat, lon=0)."""
        return (
            (self.state.lat - self.reference_lat) * self.METERS_PER_DEG_LAT,
            self.state.lon * self.METERS_PER_DEG_LAT * np.cos(np.radians(self.reference_lat)),
        )

    def schedule_manoeuvre(
        self,
        at_time_s: float,
        target_heading_deg: float,
        target_speed_kn: float,
    ) -> None:
        """Schedule a manoeuvre: at the given time, the vessel will steer toward
        the target heading and adjust speed to the target."""
        self._manoeuvres.append((at_time_s, target_heading_deg, target_speed_kn))
        self._manoeuvres.sort()

    def step(self, dt_s: float) -> VesselState:
        """Advance the vessel by dt seconds."""
        s = self.current_state
        new_time = s.time_s + dt_s

        # Check for any scheduled manoeuvre whose time has come.
        # The target heading/speed persists across subsequent steps until
        # another manoeuvre overrides it.
        while self._manoeuvres and self._manoeuvres[0][0] <= new_time:
            t, h, sp = self._manoeuvres.pop(0)
            self._target_heading = h
            self._target_speed = sp

        # Use the persisted target if set, otherwise the current state
        target_heading = getattr(self, "_target_heading", s.heading_deg)
        target_speed = getattr(self, "_target_speed", s.speed_kn)

        # Compute heading delta and apply bounded turn rate
        heading_delta = (target_heading - s.heading_deg + 540) % 360 - 180  # shortest path
        max_turn_this_step = self.kinematics.max_turn_rate_dps * dt_s
        turn_step = float(np.clip(heading_delta, -max_turn_this_step, max_turn_this_step))
        new_heading = (s.heading_deg + turn_step) % 360

        # Compute speed delta and apply bounded acceleration
        speed_delta = target_speed - s.speed_kn
        max_accel = self.kinematics.accel_kn_per_s * dt_s if speed_delta > 0 else self.kinematics.decel_kn_per_s * dt_s
        speed_step = float(np.clip(speed_delta, -max_accel, max_accel))
        new_speed = float(np.clip(s.speed_kn + speed_step, 0.0, self.kinematics.max_speed_kn))

        # Convert speed to m/s for position update (1 kn = 0.5144 m/s)
        speed_ms = new_speed * 0.5144

        # Position update — heading 0 = north (positive lat direction)
        # Distance in metres
        north_m = speed_ms * np.cos(np.radians(new_heading)) * dt_s
        east_m = speed_ms * np.sin(np.radians(new_heading)) * dt_s

        # Convert back to degrees
        dlat = north_m / self.METERS_PER_DEG_LAT
        dlon = east_m / (self.METERS_PER_DEG_LAT * np.cos(np.radians(self.reference_lat)))

        new_state = VesselState(
            time_s=new_time,
            lat=s.lat + dlat,
            lon=s.lon + dlon,
            heading_deg=new_heading,
            speed_kn=new_speed,
            turn_rate_dps=turn_step / dt_s if dt_s > 0 else 0.0,
        )
        self.history.append(new_state)
        self.state = new_state
        return new_state

    def simulate(self, duration_s: float, dt_s: float = 1.0) -> list[VesselState]:
        """Simulate forward for the given duration."""
        n_steps = int(duration_s / dt_s)
        for _ in range(n_steps):
            self.step(dt_s)
        return self.history

    def to_trajectory_array(self) -> np.ndarray:
        """Return trajectory as Nx6 array: time, lat, lon, heading, speed, turn_rate."""
        return np.array([
            [s.time_s, s.lat, s.lon, s.heading_deg, s.speed_kn, s.turn_rate_dps]
            for s in self.history
        ])

    def __repr__(self) -> str:
        s = self.current_state
        return (
            f"Vessel(id={self.vessel_id!r}, class={self.vessel_class.value}, "
            f"lat={s.lat:.4f}, lon={s.lon:.4f}, "
            f"heading={s.heading_deg:.1f}°, speed={s.speed_kn:.1f} kn, "
            f"t={s.time_s:.1f}s)"
        )


# ───────────────────────────────────────────────────────────
# Factory functions
# ───────────────────────────────────────────────────────────

def create_vessel(
    vessel_id: str,
    vessel_class: VesselClass,
    start_lat: float,
    start_lon: float,
    start_heading_deg: float,
    start_speed_kn: Optional[float] = None,
    reference_lat: float = 13.0,
) -> Vessel:
    """Create a vessel at the given position with the given kinematics."""
    kin = VESSEL_CLASSES[vessel_class]
    speed = start_speed_kn if start_speed_kn is not None else kin.cruise_speed_kn
    return Vessel(
        vessel_id=vessel_id,
        vessel_class=vessel_class,
        initial_state=VesselState(
            time_s=0.0,
            lat=start_lat,
            lon=start_lon,
            heading_deg=start_heading_deg,
            speed_kn=speed,
        ),
        reference_lat=reference_lat,
    )
