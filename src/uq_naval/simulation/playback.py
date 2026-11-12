"""Scenario playback and visualization.

Generates matplotlib plots of vessel tracks, encounter geometry, and sensor
observations. These visualizations are suitable for inclusion in a DRDO
evaluation report or for review during scenario playback.

All plots use the project's navy + gold colour palette and produce PNG files
suitable for inclusion in the proposal document.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

# Configure matplotlib for headless rendering with consistent style
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .vessels import Vessel, VESSEL_CLASSES, VesselClass
from .colregs import EncounterClassification, EncounterType, classify_encounter
from .scenarios import MaritimeScenario


# Project palette
NAVY = "#0F2845"
NAVY_LIGHT = "#1E4275"
GOLD = "#C9963A"
GREEN = "#3F8A5E"
AMBER = "#E0A53D"
RED = "#B5443A"
GREY_LIGHT = "#C2D0E2"
GREY_MED = "#8FA3BF"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.edgecolor": GREY_MED,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": GREY_LIGHT,
    "grid.linewidth": 0.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 150,
})


def _vessel_trajectory_xy(vessel: Vessel) -> tuple[np.ndarray, np.ndarray]:
    """Return vessel trajectory as (north_m, east_m) arrays."""
    states = vessel.history
    METERS_PER_DEG_LAT = 111_320.0
    ref_lat = vessel.reference_lat
    north = np.array([(s.lat - ref_lat) * METERS_PER_DEG_LAT for s in states])
    east = np.array([s.lon * METERS_PER_DEG_LAT * np.cos(np.radians(ref_lat)) for s in states])
    return north, east


def plot_scenario_tracks(
    scenario: MaritimeScenario,
    output_path: Optional[str | Path] = None,
    title: Optional[str] = None,
) -> Optional[Path]:
    """Plot vessel tracks for a scenario.

    Shows own vessel (navy) and target vessels (colour-coded by class) with
    their full trajectories. Start positions marked with circles, end
    positions marked with arrows indicating final heading.
    """
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)

    # Own vessel
    n_own, e_own = _vessel_trajectory_xy(scenario.own_vessel)
    ax.plot(e_own, n_own, color=NAVY, linewidth=2.0, label=f"Own ({scenario.own_vessel.vessel_class.value})", zorder=5)
    ax.scatter(e_own[0], n_own[0], color=NAVY, s=80, zorder=6, marker="o", edgecolor="white", linewidth=1.5)
    # Final heading arrow
    final = scenario.own_vessel.current_state
    arrow_len = 200
    ax.annotate(
        "",
        xy=(e_own[-1] + arrow_len * np.sin(np.radians(final.heading_deg)),
            n_own[-1] + arrow_len * np.cos(np.radians(final.heading_deg))),
        xytext=(e_own[-1], n_own[-1]),
        arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.5),
    )

    # Target vessels
    for i, target in enumerate(scenario.target_vessels):
        n_t, e_t = _vessel_trajectory_xy(target)
        color = VESSEL_CLASSES[target.vessel_class].color
        ax.plot(e_t, n_t, color=color, linewidth=1.5, linestyle="--",
                label=f"{target.vessel_id} ({target.vessel_class.value})", zorder=4)
        ax.scatter(e_t[0], n_t[0], color=color, s=50, zorder=5, marker="s", edgecolor="white", linewidth=1)

        # Final heading arrow
        final_t = target.current_state
        ax.annotate(
            "",
            xy=(e_t[-1] + arrow_len * 0.7 * np.sin(np.radians(final_t.heading_deg)),
                n_t[-1] + arrow_len * 0.7 * np.cos(np.radians(final_t.heading_deg))),
            xytext=(e_t[-1], n_t[-1]),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.2, linestyle="--"),
        )

    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title(title or f"Scenario: {scenario.name}", color=NAVY)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.set_aspect("equal")
    ax.grid(True, color=GREY_LIGHT, linewidth=0.4)

    # Add compass
    ax.annotate(
        "N",
        xy=(0.97, 0.97), xycoords="axes fraction",
        fontsize=11, fontweight="bold", color=NAVY, ha="center", va="center",
    )
    ax.annotate(
        "",
        xy=(0.97, 0.92), xytext=(0.97, 0.99),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.5),
    )

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out)
        plt.close(fig)
        return out
    plt.close(fig)
    return None


def plot_encounter_geometry(
    scenario: MaritimeScenario,
    output_path: Optional[str | Path] = None,
) -> Optional[Path]:
    """Plot CPA / TCPA evolution over the scenario duration.

    Shows closest point of approach (CPA) and time to CPA (TCPA) for each
    target vessel over the scenario duration. This is the key safety metric
    for COLREGs compliance assessment.
    """
    if not scenario.encounter_classifications:
        raise RuntimeError("Scenario must be run before plotting encounter geometry.")

    n_targets = len(scenario.target_vessels)
    if n_targets == 0:
        return None

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), constrained_layout=True, sharex=True)

    times = np.arange(len(scenario.encounter_classifications)) * scenario.dt_s

    # CPA plot
    ax = axes[0]
    for i in range(n_targets):
        cpas = [cls[i].cpa_m for cls in scenario.encounter_classifications]
        color = VESSEL_CLASSES[scenario.target_vessels[i].vessel_class].color
        ax.plot(times, cpas, color=color, linewidth=1.5, label=f"Target {i+1}")
    ax.axhline(y=500, color=RED, linestyle=":", linewidth=1, alpha=0.7, label="Safety threshold (500 m)")
    ax.set_ylabel("CPA (m)")
    ax.set_title("Closest Point of Approach", color=NAVY)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, color=GREY_LIGHT, linewidth=0.4)

    # TCPA plot
    ax = axes[1]
    for i in range(n_targets):
        tcpas = [cls[i].tcpa_s for cls in scenario.encounter_classifications]
        color = VESSEL_CLASSES[scenario.target_vessels[i].vessel_class].color
        # Clip inf values for plotting
        tcpas_clipped = [min(t, scenario.duration_s * 2) for t in tcpas]
        ax.plot(times, tcpas_clipped, color=color, linewidth=1.5, label=f"Target {i+1}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("TCPA (s)")
    ax.set_title("Time to Closest Point of Approach", color=NAVY)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, color=GREY_LIGHT, linewidth=0.4)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out)
        plt.close(fig)
        return out
    plt.close(fig)
    return None


def plot_sensor_observations(
    scenario: MaritimeScenario,
    output_path: Optional[str | Path] = None,
) -> Optional[Path]:
    """Plot sensor observation counts over the scenario duration.

    Shows radar detections (vessels + clutter), AIS messages received, and
    EO/IR detections per frame.
    """
    if not scenario.radar_observations:
        raise RuntimeError("Scenario must be run before plotting sensor observations.")

    times = np.arange(len(scenario.radar_observations)) * scenario.dt_s

    radar_vessels = [sum(1 for o in obs if not o.is_clutter) for obs in scenario.radar_observations]
    radar_clutter = [sum(1 for o in obs if o.is_clutter) for obs in scenario.radar_observations]
    ais_count = [len(msgs) for msgs in scenario.ais_messages]
    eoir_detections = [len(f.detections) for f in scenario.eoir_frames]

    fig, axes = plt.subplots(3, 1, figsize=(9, 7), constrained_layout=True, sharex=True)

    # Radar
    ax = axes[0]
    ax.fill_between(times, 0, radar_vessels, color=NAVY, alpha=0.5, label="Vessel detections")
    ax.fill_between(times, radar_vessels, np.array(radar_vessels) + np.array(radar_clutter),
                    color=GREY_MED, alpha=0.5, label="Sea clutter")
    ax.set_ylabel("Radar returns")
    ax.set_title("Radar Observations", color=NAVY)
    ax.legend(loc="upper right", fontsize=9)

    # AIS
    ax = axes[1]
    ax.step(times, ais_count, color=GOLD, linewidth=1.5, where="mid", label="AIS messages")
    ax.set_ylabel("AIS messages")
    ax.set_title("AIS Reception", color=NAVY)
    ax.legend(loc="upper right", fontsize=9)

    # EO/IR
    ax = axes[2]
    ax.plot(times, eoir_detections, color=GREEN, linewidth=1.5, label="EO/IR detections")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Detections per frame")
    ax.set_title("EO/IR Detections", color=NAVY)
    ax.legend(loc="upper right", fontsize=9)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out)
        plt.close(fig)
        return out
    plt.close(fig)
    return None


def render_scenario_animation(
    scenario: MaritimeScenario,
    output_path: str | Path,
    interval_ms: int = 100,
) -> Path:
    """Render the scenario as an animated GIF showing vessel tracks over time.

    The animation shows the scenario evolving timestep by timestep, with own
    vessel (navy) and target vessels (colour-coded) moving along their
    trajectories.
    """
    import matplotlib.animation as animation

    if not scenario.radar_observations:
        raise RuntimeError("Scenario must be run before rendering animation.")

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)

    # Compute full trajectory bounds for fixed axis
    all_n, all_e = [], []
    for vessel in scenario.all_vessels:
        n, e = _vessel_trajectory_xy(vessel)
        all_n.append(n)
        all_e.append(e)
    all_n = np.concatenate(all_n)
    all_e = np.concatenate(all_e)
    margin = 200
    ax.set_xlim(all_e.min() - margin, all_e.max() + margin)
    ax.set_ylim(all_n.min() - margin, all_n.max() + margin)
    ax.set_aspect("equal")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title(f"Scenario: {scenario.name}", color=NAVY)
    ax.grid(True, color=GREY_LIGHT, linewidth=0.4)

    # Pre-compute trajectories
    own_n, own_e = _vessel_trajectory_xy(scenario.own_vessel)
    target_trajs = [
        _vessel_trajectory_xy(t) for t in scenario.target_vessels
    ]

    # Lines to update
    own_line, = ax.plot([], [], color=NAVY, linewidth=2.0, label="Own vessel")
    own_dot, = ax.plot([], [], "o", color=NAVY, markersize=10, markeredgecolor="white", markeredgewidth=1.5)

    target_lines = []
    target_dots = []
    for i, target in enumerate(scenario.target_vessels):
        color = VESSEL_CLASSES[target.vessel_class].color
        line, = ax.plot([], [], color=color, linewidth=1.5, linestyle="--",
                       label=f"{target.vessel_id} ({target.vessel_class.value})")
        dot, = ax.plot([], [], "s", color=color, markersize=7, markeredgecolor="white", markeredgewidth=1)
        target_lines.append(line)
        target_dots.append(dot)

    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)

    def init():
        own_line.set_data([], [])
        own_dot.set_data([], [])
        for line, dot in zip(target_lines, target_dots):
            line.set_data([], [])
            dot.set_data([], [])
        return [own_line, own_dot, *target_lines, *target_dots]

    def update(frame):
        own_line.set_data(own_e[:frame+1], own_n[:frame+1])
        own_dot.set_data([own_e[frame]], [own_n[frame]])
        for i, (line, dot) in enumerate(zip(target_lines, target_dots)):
            n, e = target_trajs[i]
            line.set_data(e[:frame+1], n[:frame+1])
            dot.set_data([e[frame]], [n[frame]])
        time_label = ax.text(0.02, 0.98, f"t = {frame * scenario.dt_s:.1f} s",
                             transform=ax.transAxes, fontsize=10, color=NAVY,
                             verticalalignment="top",
                             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=GREY_MED))
        return [own_line, own_dot, *target_lines, *target_dots, time_label]

    n_frames = len(scenario.own_vessel.history)
    anim = animation.FuncAnimation(
        fig, update, init_func=init, frames=n_frames,
        interval=interval_ms, blit=False, repeat=True,
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out), writer="pillow", fps=10)
    plt.close(fig)
    return out
