"""COLREGs encounter geometry.

Models the encounter classifications defined by the International Regulations
for Preventing Collisions at Sea, 1972 (COLREGs):

- Rule 13: Overtaking — vessel approaching from astern (>22.5° abaft beam)
- Rule 14: Head-on — vessels approaching on reciprocal or near-reciprocal courses
- Rule 15: Crossing — courses cross at an angle other than head-on or overtaking

For each encounter type, the module generates the appropriate initial geometry
between an "own vessel" and one or more "target vessels", and computes the
COLREGs-mandated action (give-way / stand-on).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from .vessels import (
    Vessel,
    VesselClass,
    VESSEL_CLASSES,
    create_vessel,
)


class EncounterType(str, Enum):
    """COLREGs encounter classifications."""
    HEAD_ON = "head_on"            # Rule 14
    CROSSING = "crossing"          # Rule 15
    OVERTAKING = "overtaking"      # Rule 13
    STAND_ON = "stand_on"          # Own vessel has right of way
    GIVE_WAY = "give_way"          # Own vessel must keep clear
    NO_ENCOUNTER = "no_encounter"


@dataclass
class EncounterClassification:
    """Result of classifying an encounter between two vessels."""
    encounter_type: EncounterType
    own_role: str               # "stand_on" | "give_way" | "neither"
    relative_bearing_deg: float  # bearing of target from own vessel (0 = north)
    range_m: float              # distance between vessels
    cpa_m: float                # closest point of approach (m)
    tcpa_s: float               # time to CPA (s)


# ───────────────────────────────────────────────────────────
# Encounter classification
# ───────────────────────────────────────────────────────────

def relative_bearing(own_heading_deg: float, target_lat: float, target_lon: float,
                     own_lat: float, own_lon: float, reference_lat: float = 13.0) -> float:
    """Compute the bearing of target from own vessel, relative to own heading.

    Returns angle in degrees, 0 = ahead, 90 = starboard beam, 180 = astern,
    270 = port beam.
    """
    METERS_PER_DEG_LAT = 111_320.0
    # Convert lat/lon difference to metres
    dn = (target_lat - own_lat) * METERS_PER_DEG_LAT
    de = (target_lon - own_lon) * METERS_PER_DEG_LAT * np.cos(np.radians(reference_lat))

    # True bearing of target from own vessel (0 = north, clockwise)
    true_bearing = np.degrees(np.arctan2(de, dn)) % 360

    # Relative bearing (0 = ahead)
    rel = (true_bearing - own_heading_deg) % 360
    return float(rel)


def classify_encounter(
    own: Vessel,
    target: Vessel,
    reference_lat: float = 13.0,
) -> EncounterClassification:
    """Classify the COLREGs encounter between own vessel and a target.

    Based on:
    - Relative bearing of target from own vessel
    - Difference in heading between own and target
    - Range and closing rate

    The classification follows the standard COLREGs geometry:
    - **Head-on** (Rule 14): target bearing within ±6° of ahead AND reciprocal
      heading within ±6°
    - **Crossing** (Rule 15): target bearing on either bow (0-90° port or starboard)
      but not head-on
    - **Overtaking** (Rule 13): target bearing more than 22.5° abaft the beam
      (i.e., bearing in arc 112.5° to 247.5° relative)
    """
    own_state = own.current_state
    tgt_state = target.current_state

    # Relative bearing (0 = ahead, 180 = astern)
    rel_brg = relative_bearing(
        own_state.heading_deg, tgt_state.lat, tgt_state.lon,
        own_state.lat, own_state.lon, reference_lat,
    )

    # Heading difference (shortest path, -180 to +180)
    heading_diff = (tgt_state.heading_deg - own_state.heading_deg + 540) % 360 - 180

    # Range in metres
    METERS_PER_DEG_LAT = 111_320.0
    dn = (tgt_state.lat - own_state.lat) * METERS_PER_DEG_LAT
    de = (tgt_state.lon - own_state.lon) * METERS_PER_DEG_LAT * np.cos(np.radians(reference_lat))
    range_m = float(np.hypot(dn, de))

    # CPA and TCPA — simplified calculation based on current velocities
    own_vel_n = own_state.speed_kn * 0.5144 * np.cos(np.radians(own_state.heading_deg))
    own_vel_e = own_state.speed_kn * 0.5144 * np.sin(np.radians(own_state.heading_deg))
    tgt_vel_n = tgt_state.speed_kn * 0.5144 * np.cos(np.radians(tgt_state.heading_deg))
    tgt_vel_e = tgt_state.speed_kn * 0.5144 * np.sin(np.radians(tgt_state.heading_deg))

    rel_vel_n = tgt_vel_n - own_vel_n
    rel_vel_e = tgt_vel_e - own_vel_e
    rel_speed = float(np.hypot(rel_vel_n, rel_vel_e))

    # TCPA = -dot(relative_position, relative_velocity) / |relative_velocity|^2
    if rel_speed > 1e-6:
        tcpa = -(dn * rel_vel_n + de * rel_vel_e) / (rel_speed ** 2)
    else:
        tcpa = float("inf")

    # CPA = |relative_position + relative_velocity * TCPA|
    if tcpa > 0 and tcpa < 1e6:
        cpa_pos_n = dn + rel_vel_n * tcpa
        cpa_pos_e = de + rel_vel_e * tcpa
        cpa = float(np.hypot(cpa_pos_n, cpa_pos_e))
    else:
        cpa = range_m  # not closing — current range is the closest

    # Classification
    # Check overtaking first: if own is faster and target is ahead with similar heading
    own_speed_ms = own_state.speed_kn * 0.5144
    target_speed_ms = tgt_state.speed_kn * 0.5144
    closing_rate = rel_speed  # positive = closing

    # Overtaking: target ahead (rel_brg ≈ 0), similar heading, own faster
    if (abs(rel_brg) < 6 or abs(rel_brg - 360) < 6) and abs(heading_diff) < 45:
        if own_speed_ms > target_speed_ms + 0.5:  # own is meaningfully faster
            encounter = EncounterType.OVERTAKING
            own_role = "give_way"  # overtaking vessel gives way
        else:
            encounter = EncounterType.HEAD_ON
            own_role = "give_way"  # both give way in head-on
    elif abs(rel_brg) < 6 or abs(rel_brg - 360) < 6:
        # Target ahead but heading differs significantly
        if abs(heading_diff) > 174 or abs(heading_diff + 360) > 174:
            encounter = EncounterType.HEAD_ON
            own_role = "give_way"
        else:
            encounter = EncounterType.CROSSING
            own_role = "stand_on" if heading_diff > 0 else "give_way"
    elif 112.5 <= rel_brg <= 247.5:
        # Target is abaft the beam — own is being overtaken
        encounter = EncounterType.OVERTAKING
        own_role = "stand_on"  # overtaking vessel gives way
    elif 6 <= rel_brg < 112.5:
        # Target on starboard bow
        encounter = EncounterType.CROSSING
        own_role = "give_way"  # give way to starboard
    else:
        # Target on port bow
        encounter = EncounterType.CROSSING
        own_role = "stand_on"

    return EncounterClassification(
        encounter_type=encounter,
        own_role=own_role,
        relative_bearing_deg=rel_brg,
        range_m=range_m,
        cpa_m=cpa,
        tcpa_s=tcpa,
    )


# ───────────────────────────────────────────────────────────
# Encounter generators
# ───────────────────────────────────────────────────────────

def generate_head_on_encounter(
    own_class: VesselClass = VesselClass.NAVAL,
    target_class: VesselClass = VesselClass.CARGO,
    initial_range_m: float = 5000.0,
    own_speed_kn: float = 18.0,
    target_speed_kn: float = 18.0,
    reference_lat: float = 13.0,
) -> tuple[Vessel, Vessel]:
    """Generate a head-on encounter (Rule 14).

    Two vessels approach each other on near-reciprocal courses.
    """
    # Own vessel at origin, heading north
    own = create_vessel(
        vessel_id="OWN",
        vessel_class=own_class,
        start_lat=reference_lat,
        start_lon=0.0,
        start_heading_deg=0.0,
        start_speed_kn=own_speed_kn,
        reference_lat=reference_lat,
    )

    # Target vessel ahead, heading south (reciprocal)
    METERS_PER_DEG_LAT = 111_320.0
    initial_lat_offset = initial_range_m / METERS_PER_DEG_LAT
    target = create_vessel(
        vessel_id="TGT1",
        vessel_class=target_class,
        start_lat=reference_lat + initial_lat_offset,
        start_lon=0.0,
        start_heading_deg=180.0,  # reciprocal
        start_speed_kn=target_speed_kn,
        reference_lat=reference_lat,
    )
    return own, target


def generate_crossing_encounter(
    own_class: VesselClass = VesselClass.NAVAL,
    target_class: VesselClass = VesselClass.CARGO,
    initial_range_m: float = 4000.0,
    crossing_angle_deg: float = 45.0,
    own_speed_kn: float = 18.0,
    target_speed_kn: float = 14.0,
    reference_lat: float = 13.0,
) -> tuple[Vessel, Vessel]:
    """Generate a crossing encounter (Rule 15).

    Target approaches from the starboard bow at the specified crossing angle.
    Own vessel is the give-way vessel.
    """
    own = create_vessel(
        vessel_id="OWN",
        vessel_class=own_class,
        start_lat=reference_lat,
        start_lon=0.0,
        start_heading_deg=0.0,
        start_speed_kn=own_speed_kn,
        reference_lat=reference_lat,
    )

    # Place target on starboard bow, on a course that crosses own's track
    METERS_PER_DEG_LAT = 111_320.0
    # Target bearing from own vessel
    target_bearing_true = crossing_angle_deg  # 0 = north (ahead), 90 = east
    target_lat_offset = initial_range_m * np.cos(np.radians(target_bearing_true)) / METERS_PER_DEG_LAT
    target_lon_offset = (
        initial_range_m * np.sin(np.radians(target_bearing_true))
        / (METERS_PER_DEG_LAT * np.cos(np.radians(reference_lat)))
    )
    # Target heading: roughly opposite of own's heading, rotated by crossing angle
    # so courses cross. A target heading of (180 - 2*crossing_angle) gives a
    # perpendicular-ish crossing geometry.
    target_heading = (180 - 2 * crossing_angle_deg) % 360

    target = create_vessel(
        vessel_id="TGT1",
        vessel_class=target_class,
        start_lat=reference_lat + target_lat_offset,
        start_lon=target_lon_offset,
        start_heading_deg=target_heading,
        start_speed_kn=target_speed_kn,
        reference_lat=reference_lat,
    )
    return own, target


def generate_overtaking_encounter(
    own_class: VesselClass = VesselClass.NAVAL,
    target_class: VesselClass = VesselClass.CARGO,
    initial_range_m: float = 2000.0,
    own_speed_kn: float = 22.0,
    target_speed_kn: float = 12.0,
    reference_lat: float = 13.0,
) -> tuple[Vessel, Vessel]:
    """Generate an overtaking encounter (Rule 13).

    Own vessel approaches a slower target from astern.
    Own vessel is the give-way vessel (overtaking).
    """
    own = create_vessel(
        vessel_id="OWN",
        vessel_class=own_class,
        start_lat=reference_lat,
        start_lon=0.0,
        start_heading_deg=0.0,
        start_speed_kn=own_speed_kn,
        reference_lat=reference_lat,
    )

    METERS_PER_DEG_LAT = 111_320.0
    target_lat_offset = initial_range_m / METERS_PER_DEG_LAT
    target = create_vessel(
        vessel_id="TGT1",
        vessel_class=target_class,
        start_lat=reference_lat + target_lat_offset,
        start_lon=0.0,
        start_heading_deg=0.0,  # same direction
        start_speed_kn=target_speed_kn,
        reference_lat=reference_lat,
    )
    return own, target
