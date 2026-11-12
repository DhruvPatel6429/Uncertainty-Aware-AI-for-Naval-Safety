# Maritime Scenario Simulator

A proper maritime scenario simulator with realistic vessel kinematics, COLREGs
encounter geometries, sensor noise models, and a scenario playback visualizer.

## Overview

The simulator replaces the toy synthetic data with realistic maritime scenarios
suitable for testing the UQ-AI Naval pipeline. It models:

- **6 vessel classes** with distinct kinematic profiles (cargo, tanker, passenger,
  fishing, naval, small craft)
- **3 sensor modalities** with realistic error characteristics (radar, AIS, EO/IR)
- **9 scripted scenarios** covering COLREGs encounters and stress-test conditions
- **Encounter classification** following COLREGs Rules 13, 14, 15
- **Visualization** — vessel tracks, encounter geometry (CPA/TCPA), sensor observations

## Quick Start

```bash
# List available scenarios
python scripts/generate_scenarios.py --list

# Generate a single scenario + plots
python scripts/generate_scenarios.py --name head_on_clear --output reports/scenarios/

# Generate all 9 scenarios
python scripts/generate_scenarios.py --all --output reports/scenarios/

# Render a scenario animation (GIF)
python scripts/generate_scenarios.py --name crossing_starboard --animation reports/scenarios/crossing.gif
```

## Vessel Classes

| Class | Length | Max Speed | Cruise | Max Turn | Description |
|---|---|---|---|---|---|
| Cargo | 200 m | 24 kn | 18 kn | 0.8 °/s | Large, slow turns, high inertia |
| Tanker | 250 m | 16 kn | 12 kn | 0.5 °/s | Very large, very slow turns |
| Passenger | 180 m | 22 kn | 18 kn | 1.5 °/s | Smooth predictable course |
| Fishing | 30 m | 12 kn | 8 kn | 5.0 °/s | Erratic manoeuvres, variable speed |
| Naval | 120 m | 30 kn | 18 kn | 3.0 °/s | Fast manoeuvres, varied speed |
| Small Craft | 8 m | 35 kn | 22 kn | 10.0 °/s | Very fast manoeuvres |

## Sensor Models

### Radar
- Bearing error: Gaussian, σ = 1°
- Range error: Gaussian, σ = 30 m
- Detection probability: 0.95 (reduced in heavy sea state)
- Sea clutter: Poisson-distributed false returns (rate configurable)
- Maximum range: 24 km

### AIS
- Position error: Gaussian, σ = 10 m (Class A)
- Speed error: Gaussian, σ = 0.1 kn
- Heading error: Gaussian, σ = 0.5°
- Message dropout: 5% (configurable)
- Update interval: 3 s (configurable)

### EO/IR
- Image size: 64×64×3
- Pixel noise: Gaussian, σ = 0.05
- Occlusion probability: 5% (configurable)
- Field of view: 60°

## Scenarios

| Scenario | Description | Vessels | Stress Condition |
|---|---|---|---|
| `head_on_clear` | Two vessels on reciprocal courses | 2 | Clear weather |
| `crossing_starboard` | Target from starboard bow | 2 | Own is give-way |
| `crossing_port` | Target from port bow | 2 | Own is stand-on |
| `overtaking` | Own overtakes slower target | 2 | Rule 13 |
| `multi_vessel_dense` | 4 targets in dense traffic | 5 | Multiple simultaneous encounters |
| `sea_clutter_heavy` | High sea state (5) | 2 | Heavy radar clutter, reduced detection |
| `ais_dropout` | Channel congestion | 2 | 40% AIS message loss |
| `low_visibility` | Fog | 2 | High EO/IR noise + occlusion |
| `distribution_shift` | Unseen vessel classes | 3 | Small craft + fishing (OOD) |

## Programmatic Usage

```python
from uq_naval.simulation import (
    ScenarioGenerator,
    plot_scenario_tracks,
    plot_encounter_geometry,
)

# Generate and run a scenario
gen = ScenarioGenerator(seed=0)
scenario = gen.generate("crossing_starboard", duration_s=120.0)
scenario.run()

# Access observations
print(scenario.summary())
print(f"Radar observations: {len(scenario.radar_observations)}")
print(f"AIS messages: {len(scenario.ais_messages)}")

# Generate plots
plot_scenario_tracks(scenario, output_path="tracks.png")
plot_encounter_geometry(scenario, output_path="encounter.png")
```

## Architecture

```
simulation/
├── vessels.py      — Vessel kinematics (6 classes, scheduled manoeuvres)
├── colregs.py      — COLREGs encounter geometry (Rules 13, 14, 15)
├── sensors.py      — Radar, AIS, EO/IR sensor noise models
├── scenarios.py    — 9 scripted scenarios + ScenarioGenerator
└── playback.py     — Matplotlib visualizations (tracks, CPA/TCPA, sensors)
```

## Testing

```bash
pytest tests/test_simulation.py -v
```

All 28 simulator tests pass, covering:
- Vessel kinematics (6 classes, manoeuvres, trajectory export)
- COLREGs encounter classification (head-on, crossing, overtaking)
- Sensor models (radar clutter, AIS dropouts, EO/IR frames)
- Scenario generation (all 9 scenarios run end-to-end)
- Playback visualization (PNG generation verified)
