"""
UQ-AI Naval — Guided Demo for DRDO/NRB Reviewers
=================================================

This notebook walks through the complete UQ-AI Naval framework:
  1. Listing available UQ methods
  2. Calibrating a single method (Bayesian DL)
  3. Producing a calibrated, uncertainty-decomposed prediction
  4. Running the end-to-end navigation pipeline
  5. Inspecting the audit trail
  6. Running the full evaluation harness across all methods
  7. Stress-testing with scripted maritime scenarios

Run interactively with:  jupyter notebook demo.ipynb
Or as a script with:     python demo.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Make src importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from uq_naval.uq import list_uq_methods, load_uq_method
from uq_naval.models.backbones import NumpyMLP
from uq_naval.data.synthetic import generate_calibration_data, SyntheticScenarioGenerator
from uq_naval.pipeline.perception import PerceptionModule, PerceptionConfig
from uq_naval.pipeline.prediction import PredictionModule, PredictionConfig
from uq_naval.pipeline.planning import PlanningModule, PlanningConfig
from uq_naval.pipeline.decision import DecisionModule, DecisionConfig
from uq_naval.pipeline.pipeline import NavalNavigationPipeline
from uq_naval.evaluation import EvaluationHarness
from uq_naval.evaluation.harness import HarnessConfig
from uq_naval.utils.seeds import set_global_seed
from uq_naval.utils.audit import AuditLogger


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def main():
    set_global_seed(0)

    # ------------------------------------------------------------------
    section("1. Available UQ Methods")
    # ------------------------------------------------------------------
    methods = list_uq_methods()
    print(f"The framework registers {len(methods)} UQ methods:")
    for m in methods:
        print(f"  - {m}")

    # ------------------------------------------------------------------
    section("2. Calibrate a Single Method (Bayesian DL with MC Dropout)")
    # ------------------------------------------------------------------
    model = NumpyMLP(input_dim=32, n_classes=5, seed=0)
    uq = load_uq_method("bayesian", model=model, config={
        "n_mc_samples": 10,
        "reject_threshold": 0.5,
    })
    print(f"UQ method: {uq}")

    cal_data = generate_calibration_data(n_samples=100, n_classes=5, feature_dim=32, seed=0)
    cal_result = uq.calibrate(cal_data)
    print(f"\nCalibration result:")
    print(f"  Method: {cal_result.method}")
    print(f"  N calibration: {cal_result.n_calibration}")
    print(f"  ECE before: {cal_result.metrics.get('ece_before', 'N/A'):.4f}")
    print(f"  ECE after:  {cal_result.metrics.get('ece_after', 'N/A'):.4f}")
    print(f"  Temperature: {cal_result.metrics.get('temperature', 'N/A'):.4f}")

    # ------------------------------------------------------------------
    section("3. Produce a Calibrated Prediction with Uncertainty Decomposition")
    # ------------------------------------------------------------------
    test_x = cal_data[0][0]
    out = uq.predict(test_x)
    print(f"Prediction:        {out.prediction}")
    print(f"Confidence:        {out.confidence:.4f}")
    print(f"Aleatoric unc:     {out.aleatoric_unc:.4f}  (irreducible data noise)")
    print(f"Epistemic unc:     {out.epistemic_unc:.4f}  (model ignorance)")
    print(f"Reject:            {out.reject}")
    print(f"Audit fields:      {list(out.audit.keys())}")

    # ------------------------------------------------------------------
    section("4. Run the End-to-End Navigation Pipeline")
    # ------------------------------------------------------------------
    perception_model = NumpyMLP(input_dim=32, n_classes=5, seed=0)
    prediction_model = NumpyMLP(input_dim=40, n_classes=5, seed=1)
    perception_uq = load_uq_method("bayesian", model=perception_model, config={"n_mc_samples": 5})
    prediction_uq = load_uq_method("ensemble", model=[prediction_model] * 3, config={"temperature": 1.0})

    pipeline = NavalNavigationPipeline(
        perception=PerceptionModule(detector=perception_model, uq_method=perception_uq,
                                    config=PerceptionConfig()),
        prediction=PredictionModule(forecaster=prediction_model, uq_method=prediction_uq,
                                    config=PredictionConfig()),
        planning=PlanningModule(config=PlanningConfig()),
        decision=DecisionModule(config=DecisionConfig()),
    )

    sensor_input = np.random.randn(3, 32, 32).astype(np.float32)
    own_state = {"lat": 0.0, "lon": 0.0, "heading_deg": 0.0, "speed_kn": 10.0}
    decision = pipeline.process(sensor_input, own_state)

    print(f"Decision: {decision.action}")
    print(f"Collision probability bound: {decision.collision_probability_bound:.6f}")
    if decision.reject_reason:
        print(f"Reject reason: {decision.reject_reason}")
    print(f"\nAudit trail:")
    print(json.dumps(decision.audit_trail, indent=2))

    # ------------------------------------------------------------------
    section("5. Inspect the Audit Trail")
    # ------------------------------------------------------------------
    audit = pipeline.decision.audit_log
    print(f"Audit log entries: {len(audit)}")
    if audit:
        print(f"\nFirst entry:")
        print(json.dumps(audit[0], indent=2))

    # Persist to disk for safety-assurance evidence
    audit_logger = AuditLogger(log_path="reports/audit_trail.jsonl")
    audit_logger.log(decision.audit_trail)
    print(f"\nAudit trail persisted to: reports/audit_trail.jsonl")

    # ------------------------------------------------------------------
    section("6. Run the Full Evaluation Harness Across All Methods")
    # ------------------------------------------------------------------
    config = HarnessConfig(
        n_calibration=50,
        n_test=50,
        n_classes=5,
        feature_dim=32,
        alpha=0.1,
        seed=0,
    )
    harness = EvaluationHarness(config=config)
    harness.run()

    print(harness.summary_table())

    # ------------------------------------------------------------------
    section("7. Maritime Scenario Simulator (Realistic)")
    # ------------------------------------------------------------------
    from uq_naval.simulation import (
        list_scenarios as sim_list_scenarios,
        generate_scenario,
        plot_scenario_tracks,
    )
    sim_scenarios = sim_list_scenarios()
    print(f"Available scenarios ({len(sim_scenarios)}):")
    for name in sim_scenarios:
        print(f"  - {name}")

    print("\nGenerating one scenario end-to-end:")
    s = generate_scenario("crossing_starboard", duration_s=30.0, dt_s=1.0)
    s.run()
    summary = s.summary()
    print(f"  Scenario: {summary['name']}")
    print(f"  Vessels: {summary['n_vessels']} (own: {summary['own_vessel_class']}, targets: {summary['target_vessel_classes']})")
    print(f"  Duration: {summary['duration_s']}s, timesteps: {summary['n_timesteps']}")
    print(f"  Radar observations: {summary['total_radar_observations']}")
    print(f"  AIS messages: {summary['total_ais_messages']}")
    print(f"  EO/IR detections: {summary['total_eoir_detections']}")

    # Save the tracks plot
    plot_scenario_tracks(s, output_path="reports/scenario_demo.png")
    print(f"  Tracks plot saved: reports/scenario_demo.png")

    # ------------------------------------------------------------------
    section("Demo Complete")
    # ------------------------------------------------------------------
    print("""
Key takeaways for DRDO reviewers:
  ✓ 11 UQ methods registered behind a unified plug-in interface
  ✓ Each method produces calibrated, decomposed uncertainty (aleatoric + epistemic)
  ✓ End-to-end pipeline produces EXECUTE / REJECT_HUMAN / EMERGENCY_FALLBACK decisions
  ✓ Every decision is logged with full audit trail for safety-assurance evidence
  ✓ Evaluation harness produces the comparison table reported in the proposal
  ✓ Scenario library exercises COLREGs and stress-test conditions

Next steps:
  - Read the 32-page proposal: UQ_Naval_Proposal.pdf
  - Browse the source: src/uq_naval/
  - Run the test suite: pytest tests/ -v
  - Start the API: python scripts/serve.py
  - Open the dashboard: cd dashboard && npm run dev
""")


if __name__ == "__main__":
    main()
