"""Maritime scenario simulator.

A proper maritime scenario simulator with realistic vessel kinematics, COLREGs
encounter geometries, sensor noise models, and a scenario playback visualizer.

Subpackages
-----------
- :mod:`vessels`    — vessel class definitions and kinematics
- :mod:`colregs`    — COLREGs encounter geometry
- :mod:`sensors`    — radar / AIS / EO-IR sensor noise models
- :mod:`scenarios` — scripted and stochastic scenario generation
- :mod:`playback`   — visualization and scenario playback
"""
from .vessels import Vessel, VesselClass, VESSEL_CLASSES, VesselState, create_vessel
from .colregs import (
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
from .scenarios import (
    MaritimeScenario,
    ScenarioGenerator,
    list_scenarios,
    generate_scenario,
)
from .playback import (
    plot_scenario_tracks,
    plot_encounter_geometry,
    plot_sensor_observations,
    render_scenario_animation,
)

__all__ = [
    "Vessel",
    "VesselClass",
    "VESSEL_CLASSES",
    "VesselState",
    "create_vessel",
    "EncounterType",
    "classify_encounter",
    "generate_head_on_encounter",
    "generate_crossing_encounter",
    "generate_overtaking_encounter",
    "RadarSensor",
    "AISSensor",
    "EOIRSensor",
    "RadarObservation",
    "AISMessage",
    "EOIRFrame",
    "MaritimeScenario",
    "ScenarioGenerator",
    "list_scenarios",
    "generate_scenario",
    "plot_scenario_tracks",
    "plot_encounter_geometry",
    "plot_sensor_observations",
    "render_scenario_animation",
]
