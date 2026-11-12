"""Maritime scenario generator.

Combines vessel kinematics, COLREGs encounter geometry, and sensor models
into complete scenario specifications. Each scenario produces:
- A list of vessels (own + targets) with full trajectory history
- Sensor observations (radar, AIS, EO/IR) at each timestep
- Ground truth encounter classifications

Scenarios
---------
- ``head_on_clear``         — Rule 14, clear conditions
- ``crossing_starboard``    — Rule 15, own is give-way
- ``crossing_port``         — Rule 15, own is stand-on
- ``overtaking``            — Rule 13, own overtakes slower vessel
- ``multi_vessel_dense``   — 4+ vessels in dense traffic
- ``sea_clutter_heavy``     — High radar clutter (sea state 5+)
- ``ais_dropout``           — Frequent AIS message loss
- ``low_visibility``        — EO/IR occlusion + low contrast
- ``distribution_shift``    — Vessel classes unseen in nominal scenarios
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .vessels import Vessel, VesselClass, VesselState, create_vessel
from .colregs import (
    EncounterClassification,
    EncounterType,
    classify_encounter,
    generate_head_on_encounter,
    generate_crossing_encounter,
    generate_overtaking_encounter,
)
from .sensors import (
    RadarSensor,
    AISSensor,
    EOIRSensor,
    RadarObservation,
    AISMessage,
    EOIRFrame,
)


@dataclass
class MaritimeScenario:
    """A complete maritime scenario with vessels, sensors, and observations."""
    name: str
    description: str
    duration_s: float
    dt_s: float
    own_vessel: Vessel
    target_vessels: list[Vessel]
    radar: RadarSensor
    ais: AISSensor
    eoir: EOIRSensor
    # Recorded observations (populated by .run())
    radar_observations: list[list[RadarObservation]] = field(default_factory=list)
    ais_messages: list[list[AISMessage]] = field(default_factory=list)
    eoir_frames: list[EOIRFrame] = field(default_factory=list)
    encounter_classifications: list[list[EncounterClassification]] = field(default_factory=list)
    # Weather / environment
    sea_state: int = 2  # Beaufort 0-12
    visibility_km: float = 10.0

    @property
    def all_vessels(self) -> list[Vessel]:
        return [self.own_vessel, *self.target_vessels]

    def run(self) -> None:
        """Execute the scenario, recording observations at each timestep."""
        n_steps = int(self.duration_s / self.dt_s)
        self.radar_observations = []
        self.ais_messages = []
        self.eoir_frames = []
        self.encounter_classifications = []

        for step in range(n_steps):
            time_s = step * self.dt_s

            # Step all vessels
            for vessel in self.all_vessels:
                vessel.step(self.dt_s)

            # Observe with all sensors
            self.radar_observations.append(self.radar.observe(self.all_vessels, time_s))
            self.ais_messages.append(self.ais.observe(self.all_vessels, time_s))
            self.eoir_frames.append(self.eoir.observe(self.all_vessels, time_s))

            # Classify encounters between own vessel and each target
            classifications: list[EncounterClassification] = []
            for target in self.target_vessels:
                cls = classify_encounter(self.own_vessel, target)
                classifications.append(cls)
            self.encounter_classifications.append(classifications)

    def summary(self) -> dict:
        """Return a summary of the scenario for reporting."""
        return {
            "name": self.name,
            "description": self.description,
            "duration_s": self.duration_s,
            "n_timesteps": int(self.duration_s / self.dt_s),
            "n_vessels": len(self.all_vessels),
            "own_vessel_class": self.own_vessel.vessel_class.value,
            "target_vessel_classes": [t.vessel_class.value for t in self.target_vessels],
            "sea_state": self.sea_state,
            "visibility_km": self.visibility_km,
            "total_radar_observations": sum(len(obs) for obs in self.radar_observations),
            "total_ais_messages": sum(len(msgs) for msgs in self.ais_messages),
            "total_eoir_frames": len(self.eoir_frames),
            "total_eoir_detections": sum(len(f.detections) for f in self.eoir_frames),
        }


# ───────────────────────────────────────────────────────────
# Scenario generator
# ───────────────────────────────────────────────────────────

SCENARIO_NAMES = [
    "head_on_clear",
    "crossing_starboard",
    "crossing_port",
    "overtaking",
    "multi_vessel_dense",
    "sea_clutter_heavy",
    "ais_dropout",
    "low_visibility",
    "distribution_shift",
]

SCENARIO_DESCRIPTIONS = {
    "head_on_clear": "Two vessels approach on reciprocal courses in clear weather (Rule 14).",
    "crossing_starboard": "Target approaches from starboard bow; own vessel is give-way (Rule 15).",
    "crossing_port": "Target approaches from port bow; own vessel is stand-on (Rule 15).",
    "overtaking": "Own vessel overtakes a slower target from astern (Rule 13).",
    "multi_vessel_dense": "Four targets in dense traffic — multiple simultaneous encounters.",
    "sea_clutter_heavy": "High sea state (5) with heavy radar clutter and reduced detection probability.",
    "ais_dropout": "Frequent AIS message dropouts due to channel congestion.",
    "low_visibility": "Reduced EO/IR visibility due to fog — high occlusion probability.",
    "distribution_shift": "Vessel classes unseen in nominal scenarios (small craft, fishing).",
}


def list_scenarios() -> list[str]:
    """List available scenario names."""
    return list(SCENARIO_NAMES)


def generate_scenario(
    name: str,
    duration_s: float = 120.0,
    dt_s: float = 1.0,
    seed: int = 0,
) -> MaritimeScenario:
    """Generate a complete maritime scenario by name."""
    if name not in SCENARIO_NAMES:
        raise ValueError(f"Unknown scenario: {name!r}. Available: {SCENARIO_NAMES}")

    description = SCENARIO_DESCRIPTIONS[name]

    if name == "head_on_clear":
        return _generate_head_on_clear(duration_s, dt_s, seed, description)
    elif name == "crossing_starboard":
        return _generate_crossing_starboard(duration_s, dt_s, seed, description)
    elif name == "crossing_port":
        return _generate_crossing_port(duration_s, dt_s, seed, description)
    elif name == "overtaking":
        return _generate_overtaking(duration_s, dt_s, seed, description)
    elif name == "multi_vessel_dense":
        return _generate_multi_vessel(duration_s, dt_s, seed, description)
    elif name == "sea_clutter_heavy":
        return _generate_sea_clutter_heavy(duration_s, dt_s, seed, description)
    elif name == "ais_dropout":
        return _generate_ais_dropout(duration_s, dt_s, seed, description)
    elif name == "low_visibility":
        return _generate_low_visibility(duration_s, dt_s, seed, description)
    elif name == "distribution_shift":
        return _generate_distribution_shift(duration_s, dt_s, seed, description)
    raise ValueError(f"Scenario {name!r} not implemented")


# ───────────────────────────────────────────────────────────
# Individual scenario generators
# ───────────────────────────────────────────────────────────

def _generate_head_on_clear(duration_s, dt_s, seed, description) -> MaritimeScenario:
    own, target = generate_head_on_encounter(
        own_class=VesselClass.NAVAL,
        target_class=VesselClass.CARGO,
        initial_range_m=5000.0,
        own_speed_kn=18.0,
        target_speed_kn=18.0,
    )
    return MaritimeScenario(
        name="head_on_clear",
        description=description,
        duration_s=duration_s,
        dt_s=dt_s,
        own_vessel=own,
        target_vessels=[target],
        radar=RadarSensor(seed=seed),
        ais=AISSensor(seed=seed),
        eoir=EOIRSensor(seed=seed),
        sea_state=2,
        visibility_km=10.0,
    )


def _generate_crossing_starboard(duration_s, dt_s, seed, description) -> MaritimeScenario:
    own, target = generate_crossing_encounter(
        own_class=VesselClass.NAVAL,
        target_class=VesselClass.CARGO,
        initial_range_m=4000.0,
        crossing_angle_deg=45.0,
        own_speed_kn=18.0,
        target_speed_kn=14.0,
    )
    # Schedule own vessel to alter course to starboard after 60 s (give-way action)
    own.schedule_manoeuvre(at_time_s=60.0, target_heading_deg=30.0, target_speed_kn=15.0)
    return MaritimeScenario(
        name="crossing_starboard",
        description=description,
        duration_s=duration_s,
        dt_s=dt_s,
        own_vessel=own,
        target_vessels=[target],
        radar=RadarSensor(seed=seed),
        ais=AISSensor(seed=seed),
        eoir=EOIRSensor(seed=seed),
    )


def _generate_crossing_port(duration_s, dt_s, seed, description) -> MaritimeScenario:
    own, target = generate_crossing_encounter(
        own_class=VesselClass.NAVAL,
        target_class=VesselClass.PASSENGER,
        initial_range_m=4000.0,
        crossing_angle_deg=-45.0,  # target from port
        own_speed_kn=18.0,
        target_speed_kn=20.0,
    )
    return MaritimeScenario(
        name="crossing_port",
        description=description,
        duration_s=duration_s,
        dt_s=dt_s,
        own_vessel=own,
        target_vessels=[target],
        radar=RadarSensor(seed=seed),
        ais=AISSensor(seed=seed),
        eoir=EOIRSensor(seed=seed),
    )


def _generate_overtaking(duration_s, dt_s, seed, description) -> MaritimeScenario:
    own, target = generate_overtaking_encounter(
        own_class=VesselClass.NAVAL,
        target_class=VesselClass.CARGO,
        initial_range_m=2000.0,
        own_speed_kn=22.0,
        target_speed_kn=12.0,
    )
    return MaritimeScenario(
        name="overtaking",
        description=description,
        duration_s=duration_s,
        dt_s=dt_s,
        own_vessel=own,
        target_vessels=[target],
        radar=RadarSensor(seed=seed),
        ais=AISSensor(seed=seed),
        eoir=EOIRSensor(seed=seed),
    )


def _generate_multi_vessel(duration_s, dt_s, seed, description) -> MaritimeScenario:
    own = create_vessel("OWN", VesselClass.NAVAL, 13.0, 0.0, 0.0, 18.0)
    targets = [
        # Target ahead, reciprocal — head-on
        create_vessel("T1", VesselClass.CARGO, 13.045, 0.0, 180.0, 18.0),
        # Target starboard bow, crossing
        create_vessel("T2", VesselClass.PASSENGER, 13.030, 0.030, 225.0, 16.0),
        # Target astern, overtaking
        create_vessel("T3", VesselClass.SMALL_CRAFT, 12.980, 0.0, 0.0, 28.0),
        # Target port beam, crossing
        create_vessel("T4", VesselClass.FISHING, 13.010, -0.035, 60.0, 8.0),
    ]
    return MaritimeScenario(
        name="multi_vessel_dense",
        description=description,
        duration_s=duration_s,
        dt_s=dt_s,
        own_vessel=own,
        target_vessels=targets,
        radar=RadarSensor(seed=seed),
        ais=AISSensor(seed=seed),
        eoir=EOIRSensor(seed=seed),
    )


def _generate_sea_clutter_heavy(duration_s, dt_s, seed, description) -> MaritimeScenario:
    own, target = generate_head_on_encounter(
        own_class=VesselClass.NAVAL,
        target_class=VesselClass.CARGO,
        initial_range_m=4500.0,
        own_speed_kn=15.0,  # reduced speed in heavy sea
        target_speed_kn=12.0,
    )
    return MaritimeScenario(
        name="sea_clutter_heavy",
        description=description,
        duration_s=duration_s,
        dt_s=dt_s,
        own_vessel=own,
        target_vessels=[target],
        radar=RadarSensor(
            clutter_rate=12.0,        # heavy sea clutter
            detection_probability=0.75,  # reduced detection
            seed=seed,
        ),
        ais=AISSensor(seed=seed),
        eoir=EOIRSensor(seed=seed),
        sea_state=5,
        visibility_km=6.0,
    )


def _generate_ais_dropout(duration_s, dt_s, seed, description) -> MaritimeScenario:
    own, target = generate_crossing_encounter(
        own_class=VesselClass.NAVAL,
        target_class=VesselClass.CARGO,
        initial_range_m=4000.0,
        crossing_angle_deg=45.0,
    )
    return MaritimeScenario(
        name="ais_dropout",
        description=description,
        duration_s=duration_s,
        dt_s=dt_s,
        own_vessel=own,
        target_vessels=[target],
        radar=RadarSensor(seed=seed),
        ais=AISSensor(dropout_probability=0.4, update_interval_s=10.0, seed=seed),
        eoir=EOIRSensor(seed=seed),
    )


def _generate_low_visibility(duration_s, dt_s, seed, description) -> MaritimeScenario:
    own, target = generate_crossing_encounter(
        own_class=VesselClass.NAVAL,
        target_class=VesselClass.PASSENGER,
        initial_range_m=3000.0,
        crossing_angle_deg=30.0,
    )
    return MaritimeScenario(
        name="low_visibility",
        description=description,
        duration_s=duration_s,
        dt_s=dt_s,
        own_vessel=own,
        target_vessels=[target],
        radar=RadarSensor(seed=seed),
        ais=AISSensor(seed=seed),
        eoir=EOIRSensor(
            noise_std=0.20,            # high pixel noise
            occlusion_probability=0.30,  # frequent occlusion
            seed=seed,
        ),
        sea_state=3,
        visibility_km=1.5,  # fog
    )


def _generate_distribution_shift(duration_s, dt_s, seed, description) -> MaritimeScenario:
    """Scenario with vessel classes not seen in nominal scenarios — tests OOD detection."""
    own = create_vessel("OWN", VesselClass.NAVAL, 13.0, 0.0, 0.0, 18.0)
    targets = [
        create_vessel("T1", VesselClass.SMALL_CRAFT, 13.030, 0.020, 180.0, 30.0),
        create_vessel("T2", VesselClass.FISHING, 13.020, -0.015, 90.0, 6.0),
    ]
    return MaritimeScenario(
        name="distribution_shift",
        description=description,
        duration_s=duration_s,
        dt_s=dt_s,
        own_vessel=own,
        target_vessels=targets,
        radar=RadarSensor(seed=seed),
        ais=AISSensor(seed=seed),
        eoir=EOIRSensor(seed=seed),
    )


# ───────────────────────────────────────────────────────────
# Scenario generator class (programmatic interface)
# ───────────────────────────────────────────────────────────

class ScenarioGenerator:
    """Programmatic interface for generating scenarios.

    Example:
        >>> gen = ScenarioGenerator(seed=0)
        >>> scenario = gen.generate("head_on_clear")
        >>> scenario.run()
        >>> print(scenario.summary())
    """

    def __init__(self, default_duration_s: float = 120.0, default_dt_s: float = 1.0, seed: int = 0):
        self.default_duration_s = default_duration_s
        self.default_dt_s = default_dt_s
        self.seed = seed

    def generate(
        self,
        name: str,
        duration_s: Optional[float] = None,
        dt_s: Optional[float] = None,
    ) -> MaritimeScenario:
        return generate_scenario(
            name=name,
            duration_s=duration_s or self.default_duration_s,
            dt_s=dt_s or self.default_dt_s,
            seed=self.seed,
        )

    def generate_all(self) -> list[MaritimeScenario]:
        """Generate all scenarios and run them."""
        scenarios = []
        for name in SCENARIO_NAMES:
            s = self.generate(name)
            s.run()
            scenarios.append(s)
        return scenarios
