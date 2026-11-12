"""CLI script — generate maritime scenarios and visualizations.

Usage:
    # List available scenarios
    python scripts/generate_scenarios.py --list

    # Generate a single scenario + plots
    python scripts/generate_scenarios.py --name head_on_clear --output reports/scenarios/

    # Generate all scenarios
    python scripts/generate_scenarios.py --all --output reports/scenarios/

    # Render scenario animation (GIF)
    python scripts/generate_scenarios.py --name crossing_starboard --animation reports/scenarios/crossing.gif
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from uq_naval.simulation import (
    list_scenarios,
    generate_scenario,
    plot_scenario_tracks,
    plot_encounter_geometry,
    plot_sensor_observations,
    render_scenario_animation,
)
from uq_naval.utils.logging import configure_logging, get_logger


def main():
    parser = argparse.ArgumentParser(description="Generate maritime scenarios + visualizations")
    parser.add_argument("--list", action="store_true", help="List available scenarios")
    parser.add_argument("--name", type=str, help="Scenario name to generate")
    parser.add_argument("--all", action="store_true", help="Generate all scenarios")
    parser.add_argument("--output", type=str, default="reports/scenarios/", help="Output directory")
    parser.add_argument("--duration", type=float, default=120.0, help="Scenario duration in seconds")
    parser.add_argument("--dt", type=float, default=1.0, help="Timestep in seconds")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--animation", type=str, default=None, help="Render scenario animation to this path (GIF)")
    args = parser.parse_args()

    configure_logging()
    log = get_logger("uq_naval.scenarios")

    if args.list:
        print("Available scenarios:")
        for name in list_scenarios():
            print(f"  - {name}")
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        log.info(f"Generating all {len(list_scenarios())} scenarios...")
        summaries = []
        for name in list_scenarios():
            summary = _generate_and_plot(name, output_dir, args.duration, args.dt, args.seed, log)
            summaries.append(summary)
        # Write combined summary
        summary_path = output_dir / "all_scenarios_summary.json"
        summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        log.info(f"All scenarios generated. Summary: {summary_path}")
        return

    if not args.name:
        parser.error("--name is required (or use --all)")

    _generate_and_plot(args.name, output_dir, args.duration, args.dt, args.seed, log)

    # Animation
    if args.animation:
        log.info(f"Rendering animation for {args.name}...")
        scenario = generate_scenario(args.name, duration_s=args.duration, dt_s=args.dt, seed=args.seed)
        scenario.run()
        anim_path = render_scenario_animation(scenario, args.animation)
        log.info(f"Animation saved to: {anim_path}")


def _generate_and_plot(name, output_dir, duration_s, dt_s, seed, log):
    log.info(f"Generating scenario: {name}")
    scenario = generate_scenario(name, duration_s=duration_s, dt_s=dt_s, seed=seed)
    scenario.run()

    summary = scenario.summary()
    log.info(f"  Duration: {summary['duration_s']}s, vessels: {summary['n_vessels']}, "
             f"radar obs: {summary['total_radar_observations']}, "
             f"AIS msgs: {summary['total_ais_messages']}")

    # Save summary JSON
    summary_path = output_dir / f"{name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Plot tracks
    plot_scenario_tracks(scenario, output_path=output_dir / f"{name}_tracks.png")
    log.info(f"  Tracks plot: {output_dir / f'{name}_tracks.png'}")

    # Plot encounter geometry
    try:
        plot_encounter_geometry(scenario, output_path=output_dir / f"{name}_encounter.png")
        log.info(f"  Encounter plot: {output_dir / f'{name}_encounter.png'}")
    except Exception as e:
        log.warning(f"  Encounter plot failed: {e}")

    # Plot sensor observations
    try:
        plot_sensor_observations(scenario, output_path=output_dir / f"{name}_sensors.png")
        log.info(f"  Sensor plot: {output_dir / f'{name}_sensors.png'}")
    except Exception as e:
        log.warning(f"  Sensor plot failed: {e}")

    return summary


if __name__ == "__main__":
    main()
