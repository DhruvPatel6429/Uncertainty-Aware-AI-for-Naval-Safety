"""Tests for the maritime scenario simulator."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from uq_naval.simulation import (
    Vessel,
    VesselClass,
    VESSEL_CLASSES,
    VesselState,
    create_vessel,
    EncounterType,
    classify_encounter,
    generate_head_on_encounter,
    generate_crossing_encounter,
    generate_overtaking_encounter,
    RadarSensor,
    AISSensor,
    EOIRSensor,
    MaritimeScenario,
    ScenarioGenerator,
    list_scenarios,
    generate_scenario,
    plot_scenario_tracks,
    plot_encounter_geometry,
    plot_sensor_observations,
)


# ─── Vessel kinematics ───

class TestVesselKinematics:

    def test_vessel_class_definitions(self):
        """All 6 vessel classes are defined with kinematic parameters."""
        for cls in VesselClass:
            kin = VESSEL_CLASSES[cls]
            assert kin.name
            assert kin.max_speed_kn > 0
            assert kin.cruise_speed_kn <= kin.max_speed_kn
            assert kin.max_turn_rate_dps > 0

    def test_create_vessel(self):
        v = create_vessel("TEST", VesselClass.CARGO, 13.0, 0.0, 0.0, 18.0)
        assert v.vessel_id == "TEST"
        assert v.vessel_class == VesselClass.CARGO
        assert v.current_state.speed_kn == 18.0
        assert v.current_state.heading_deg == 0.0

    def test_vessel_step_advances_time(self):
        v = create_vessel("T1", VesselClass.NAVAL, 13.0, 0.0, 0.0, 18.0)
        initial_time = v.current_state.time_s
        v.step(dt_s=1.0)
        assert v.current_state.time_s == pytest.approx(initial_time + 1.0)

    def test_vessel_moves_north_when_heading_zero(self):
        """Vessel heading 0 (north) should move to higher latitude."""
        v = create_vessel("T1", VesselClass.NAVAL, 13.0, 0.0, 0.0, 18.0)
        initial_lat = v.current_state.lat
        v.step(dt_s=10.0)
        assert v.current_state.lat > initial_lat

    def test_vessel_speed_clamped_to_max(self):
        """Speed should not exceed the vessel class maximum."""
        v = create_vessel("T1", VesselClass.CARGO, 13.0, 0.0, 0.0, 30.0)
        # Already at max — should not exceed it
        v.step(dt_s=1.0)
        assert v.current_state.speed_kn <= VESSEL_CLASSES[VesselClass.CARGO].max_speed_kn + 1e-6

    def test_vessel_simulate_returns_history(self):
        v = create_vessel("T1", VesselClass.PASSENGER, 13.0, 0.0, 0.0, 18.0)
        history = v.simulate(duration_s=30.0, dt_s=1.0)
        assert len(history) == 31  # initial + 30 steps

    def test_scheduled_manoeuvre_changes_heading(self):
        v = create_vessel("T1", VesselClass.SMALL_CRAFT, 13.0, 0.0, 0.0, 20.0)
        v.schedule_manoeuvre(at_time_s=5.0, target_heading_deg=90.0, target_speed_kn=20.0)
        v.simulate(duration_s=20.0, dt_s=1.0)
        # After 20 s, heading should be close to 90 (within turn rate limit)
        assert 60 < v.current_state.heading_deg < 120

    def test_to_trajectory_array(self):
        v = create_vessel("T1", VesselClass.NAVAL, 13.0, 0.0, 0.0, 18.0)
        v.simulate(duration_s=10.0, dt_s=1.0)
        arr = v.to_trajectory_array()
        assert arr.shape == (11, 6)  # 11 timesteps × 6 fields


# ─── COLREGs encounter geometry ───

class TestColregsEncounters:

    def test_head_on_encounter_geometry(self):
        own, target = generate_head_on_encounter(initial_range_m=5000.0)
        own.simulate(duration_s=1.0, dt_s=1.0)
        target.simulate(duration_s=1.0, dt_s=1.0)
        cls = classify_encounter(own, target)
        assert cls.encounter_type == EncounterType.HEAD_ON
        assert cls.range_m > 4000  # still far apart

    def test_crossing_encounter_geometry(self):
        own, target = generate_crossing_encounter(
            initial_range_m=4000.0, crossing_angle_deg=45.0,
        )
        own.simulate(duration_s=1.0, dt_s=1.0)
        target.simulate(duration_s=1.0, dt_s=1.0)
        cls = classify_encounter(own, target)
        # Target should be on starboard bow → crossing, own is give-way
        assert cls.encounter_type in (EncounterType.CROSSING, EncounterType.HEAD_ON)
        # CPA should be finite
        assert cls.cpa_m > 0
        assert cls.tcpa_s > 0

    def test_overtaking_encounter_geometry(self):
        own, target = generate_overtaking_encounter(
            initial_range_m=2000.0, own_speed_kn=22.0, target_speed_kn=12.0,
        )
        own.simulate(duration_s=1.0, dt_s=1.0)
        target.simulate(duration_s=1.0, dt_s=1.0)
        cls = classify_encounter(own, target)
        # Target is ahead, same direction, slower → own overtaking
        assert cls.encounter_type == EncounterType.OVERTAKING

    def test_classify_encounter_returns_valid_fields(self):
        own, target = generate_head_on_encounter()
        own.simulate(duration_s=1.0, dt_s=1.0)
        target.simulate(duration_s=1.0, dt_s=1.0)
        cls = classify_encounter(own, target)
        assert 0 <= cls.relative_bearing_deg < 360
        assert cls.range_m >= 0
        assert cls.cpa_m >= 0
        assert cls.own_role in ("stand_on", "give_way", "neither")


# ─── Sensors ───

class TestSensors:

    def test_radar_observes_vessel(self):
        own, target = generate_head_on_encounter(initial_range_m=3000.0)
        own.simulate(duration_s=1.0, dt_s=1.0)
        target.simulate(duration_s=1.0, dt_s=1.0)
        radar = RadarSensor(seed=0)
        obs = radar.observe([own, target], time_s=1.0)
        # Should have at least one vessel detection (own or target)
        vessel_obs = [o for o in obs if not o.is_clutter]
        assert len(vessel_obs) >= 1
        # Each observation should have valid fields
        for o in obs:
            assert 0 <= o.bearing_deg < 360
            assert o.range_m >= 0

    def test_radar_generates_clutter(self):
        own, target = generate_head_on_encounter()
        own.simulate(duration_s=1.0, dt_s=1.0)
        target.simulate(duration_s=1.0, dt_s=1.0)
        radar = RadarSensor(clutter_rate=5.0, seed=0)
        obs = radar.observe([own, target], time_s=1.0)
        clutter = [o for o in obs if o.is_clutter]
        # With rate=5, expected ~5 clutter returns (with variance)
        assert len(clutter) >= 1

    def test_ais_emits_messages(self):
        own, target = generate_head_on_encounter()
        own.simulate(duration_s=1.0, dt_s=1.0)
        target.simulate(duration_s=1.0, dt_s=1.0)
        ais = AISSensor(seed=0)
        msgs = ais.observe([own, target], time_s=1.0)
        assert len(msgs) >= 1
        for m in msgs:
            assert m.mmsi.startswith("MMSI")
            assert m.speed_kn >= 0
            assert 0 <= m.heading_deg < 360

    def test_ais_respects_update_interval(self):
        own, _ = generate_head_on_encounter()
        own.simulate(duration_s=10.0, dt_s=1.0)
        ais = AISSensor(update_interval_s=5.0, seed=0)
        # First call should emit
        msgs1 = ais.observe([own], time_s=1.0)
        # Second call too soon — should not emit
        msgs2 = ais.observe([own], time_s=2.0)
        # Third call after interval — should emit
        msgs3 = ais.observe([own], time_s=6.0)
        assert len(msgs1) == 1
        assert len(msgs2) == 0
        assert len(msgs3) == 1

    def test_eoir_generates_frame(self):
        own, target = generate_head_on_encounter(initial_range_m=1500.0)
        own.simulate(duration_s=1.0, dt_s=1.0)
        target.simulate(duration_s=1.0, dt_s=1.0)
        eoir = EOIRSensor(seed=0)
        frame = eoir.observe([own, target], time_s=1.0)
        assert frame.image.shape == (64, 64, 3)
        assert isinstance(frame.detections, list)


# ─── Scenario generator ───

class TestScenarioGenerator:

    def test_list_scenarios(self):
        names = list_scenarios()
        assert "head_on_clear" in names
        assert "crossing_starboard" in names
        assert "overtaking" in names
        assert "multi_vessel_dense" in names
        assert "sea_clutter_heavy" in names
        assert "distribution_shift" in names
        assert len(names) == 9

    def test_generate_head_on_clear(self):
        s = generate_scenario("head_on_clear", duration_s=10.0, dt_s=1.0)
        assert s.name == "head_on_clear"
        assert s.own_vessel is not None
        assert len(s.target_vessels) == 1
        assert s.radar is not None
        assert s.ais is not None
        assert s.eoir is not None

    def test_scenario_run_produces_observations(self):
        s = generate_scenario("head_on_clear", duration_s=10.0, dt_s=1.0)
        s.run()
        assert len(s.radar_observations) == 10
        assert len(s.ais_messages) == 10
        assert len(s.eoir_frames) == 10
        assert len(s.encounter_classifications) == 10

    def test_scenario_summary(self):
        s = generate_scenario("crossing_starboard", duration_s=30.0, dt_s=1.0)
        s.run()
        summary = s.summary()
        assert summary["name"] == "crossing_starboard"
        assert summary["duration_s"] == 30.0
        assert summary["n_vessels"] == 2
        assert summary["total_radar_observations"] > 0
        assert summary["total_ais_messages"] > 0

    def test_unknown_scenario_raises(self):
        with pytest.raises(ValueError, match="Unknown scenario"):
            generate_scenario("nonexistent")

    def test_scenario_generator_class(self):
        gen = ScenarioGenerator(seed=0)
        s = gen.generate("overtaking", duration_s=10.0, dt_s=1.0)
        s.run()
        assert s.summary()["name"] == "overtaking"

    def test_all_scenarios_run_without_error(self):
        """All 9 scenarios should run end-to-end without crashing."""
        for name in list_scenarios():
            s = generate_scenario(name, duration_s=10.0, dt_s=1.0)
            s.run()
            assert len(s.radar_observations) == 10, f"{name} failed"


# ─── Playback / visualization ───

class TestPlayback:

    def test_plot_scenario_tracks(self, tmp_path):
        s = generate_scenario("head_on_clear", duration_s=30.0, dt_s=1.0)
        s.run()
        out = plot_scenario_tracks(s, output_path=tmp_path / "tracks.png")
        assert out is not None
        assert out.exists()
        assert out.stat().st_size > 1000  # non-trivial PNG

    def test_plot_encounter_geometry(self, tmp_path):
        s = generate_scenario("crossing_starboard", duration_s=30.0, dt_s=1.0)
        s.run()
        out = plot_encounter_geometry(s, output_path=tmp_path / "encounter.png")
        assert out is not None
        assert out.exists()

    def test_plot_sensor_observations(self, tmp_path):
        s = generate_scenario("sea_clutter_heavy", duration_s=30.0, dt_s=1.0)
        s.run()
        out = plot_sensor_observations(s, output_path=tmp_path / "sensors.png")
        assert out is not None
        assert out.exists()

    def test_plot_before_run_raises(self, tmp_path):
        s = generate_scenario("head_on_clear")
        with pytest.raises(RuntimeError, match="must be run"):
            plot_encounter_geometry(s, output_path=tmp_path / "x.png")
