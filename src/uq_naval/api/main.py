"""FastAPI application factory.

Creates the service that exposes the UQ-AI Naval pipeline via REST endpoints,
including the experimental study endpoints (corruptions, reliability diagrams,
risk-coverage curves, multi-seed evaluation, cost-minimised deferral).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..uq import list_uq_methods, load_uq_method
from ..pipeline.perception import PerceptionModule, PerceptionConfig
from ..pipeline.prediction import PredictionModule, PredictionConfig
from ..pipeline.planning import PlanningModule, PlanningConfig
from ..pipeline.decision import DecisionModule, DecisionConfig
from ..pipeline.pipeline import NavalNavigationPipeline
from ..models.backbones import NumpyMLP
from ..models.ship_models.resnet import ShipResNet50, SmallShipCNN, SHIP_CLASSES
from ..evaluation.corruptions import CorruptionSuite, CorruptionType
from ..evaluation.extended_metrics import (
    compute_reliability_diagram,
    compute_risk_coverage_curve,
    fpr95,
    uncertainty_error_auroc,
    brier_decomposition,
)
from ..evaluation.harness import EvaluationHarness, HarnessConfig
from ..evaluation.multi_seed import MultiSeedRunner
from ..uq.deferral import CostMatrix, CostMinimisingDeferral
from .schemas import (
    SensorInput,
    DecisionResponse,
    PipelineStatus,
    UQMethodList,
)


def create_app():
    """Create and configure the FastAPI application."""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(
        title="UQ-AI Naval API",
        description="Uncertainty-Aware AI for Safety-Critical Naval Applications",
        version="1.0.0",
    )

    # CORS for the dashboard
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Lazy-loaded pipeline (built on first /predict call)
    state: dict = {"pipeline": None}

    def _build_pipeline() -> NavalNavigationPipeline:
        perception_model = NumpyMLP(input_dim=32, n_classes=5)
        prediction_model = NumpyMLP(input_dim=40, n_classes=5)

        perception_uq = load_uq_method("bayesian", model=perception_model, config={
            "n_mc_samples": 5,
            "reject_threshold": 0.5,
        })
        prediction_uq = load_uq_method("ensemble", model=[prediction_model] * 3, config={
            "temperature": 1.0,
        })

        perception = PerceptionModule(
            detector=perception_model,
            uq_method=perception_uq,
            config=PerceptionConfig(),
        )
        prediction = PredictionModule(
            forecaster=prediction_model,
            uq_method=prediction_uq,
            config=PredictionConfig(),
        )
        planning = PlanningModule(config=PlanningConfig())
        decision = DecisionModule(config=DecisionConfig())
        return NavalNavigationPipeline(perception, prediction, planning, decision)

    # ─── Core pipeline endpoints ──────────────────────────

    @app.get("/", response_model=PipelineStatus)
    def status() -> PipelineStatus:
        return PipelineStatus(
            status="ok",
            version="1.0.0",
            uq_methods=list_uq_methods(),
            pipeline_loaded=state["pipeline"] is not None,
        )

    @app.get("/uq-methods", response_model=UQMethodList)
    def uq_methods() -> UQMethodList:
        return UQMethodList(methods=list_uq_methods())

    @app.post("/predict", response_model=DecisionResponse)
    def predict(input: SensorInput) -> DecisionResponse:
        if state["pipeline"] is None:
            state["pipeline"] = _build_pipeline()

        frame = np.array(input.frame_data, dtype=np.float32).reshape(input.frame_shape)
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=0)

        decision = state["pipeline"].process(
            sensor_input=frame,
            own_state=input.own_state,
        )

        return DecisionResponse(
            action=decision.action,
            collision_probability_bound=decision.collision_probability_bound,
            reject_reason=decision.reject_reason,
            audit_trail=decision.audit_trail,
            n_detections=0,
            mean_epistemic_uncertainty=decision.audit_trail.get("total_epistemic", 0.0),
        )

    @app.get("/audit-log")
    def get_audit_log():
        if state["pipeline"] is None:
            return {"entries": []}
        return {"entries": state["pipeline"].decision.audit_log}

    # ─── Model info endpoints ─────────────────────────────

    @app.get("/model-info")
    def get_model_info():
        """Return information about the ship classification model."""
        return {
            "backbone": "ResNet-50",
            "pretrained_on": "ImageNet",
            "fine_tuned_on": "Ship dataset (synthetic for demo)",
            "n_classes": len(SHIP_CLASSES),
            "ship_classes": SHIP_CLASSES,
            "input_size": [224, 224, 3],
            "imagenet_normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "available_variants": [
                {"id": "resnet50", "name": "ResNet-50 (ImageNet pretrained)", "available": True},
                {"id": "small_cnn", "name": "Small CNN (fast testing)", "available": True},
            ],
        }

    # ─── Corruption suite endpoints ───────────────────────

    @app.get("/corruptions")
    def list_corruptions():
        """List all available corruption types and severity levels."""
        suite = CorruptionSuite()
        return suite.summary()

    @app.post("/corruptions/apply")
    def apply_corruption(
        image_data: list[float],
        image_shape: list[int],
        corruption_type: str,
        severity: int,
    ):
        """Apply a single corruption to an image and return the corrupted result."""
        suite = CorruptionSuite()
        try:
            ctype = CorruptionType(corruption_type)
        except ValueError:
            return {"error": f"Unknown corruption type: {corruption_type}"}
        if severity not in [1, 2, 3, 4, 5]:
            return {"error": f"Severity must be 1-5, got {severity}"}

        image = np.array(image_data, dtype=np.float32).reshape(image_shape)
        corrupted = suite.apply(image, ctype, severity)
        return {
            "corrupted_image": corrupted.ravel().tolist(),
            "corruption_type": corruption_type,
            "severity": severity,
            "shape": list(corrupted.shape),
        }

    @app.get("/corruptions/grid")
    def corruption_grid():
        """Return the full corruption × severity grid with descriptions."""
        suite = CorruptionSuite()
        descriptions = {
            "fog": "Maritime fog — reduces contrast, adds white veil",
            "rain": "Rain — vertical streaks, brightness reduction",
            "low_light": "Low light / night — darkens, adds Poisson noise",
            "glare": "Sun glare — bright spot with halo on sensor",
            "noise": "Gaussian sensor noise",
        }
        return {
            "corruptions": [
                {
                    "id": c.value,
                    "name": c.value.replace("_", " ").title(),
                    "description": descriptions.get(c.value, ""),
                    "severities": [1, 2, 3, 4, 5],
                }
                for c in suite.corruption_types
            ],
            "total_variants": 25,
        }

    # ─── Experiment endpoints ─────────────────────────────

    @app.get("/experiments/reliability")
    def get_reliability_diagram(
        n_samples: int = 200,
        n_bins: int = 15,
        seed: int = 0,
    ):
        """Compute a reliability diagram from synthetic data.

        Returns binned confidence vs. accuracy data suitable for plotting
        a reliability diagram in the frontend.
        """
        from ..data.synthetic import generate_calibration_data
        data = generate_calibration_data(n_samples=n_samples, n_classes=5, feature_dim=32, seed=seed)

        # Use a simple numpy MLP to produce probabilities
        model = NumpyMLP(input_dim=32, n_classes=5, seed=seed)
        probs_list, labels_list = [], []
        for x, y in data:
            logits = model(x)
            # Squeeze batch dimension — NumpyMLP returns (1, C)
            logits = np.asarray(logits).ravel()
            probs = np.exp(logits) / np.exp(logits).sum()
            probs_list.append(probs)
            labels_list.append(y)

        probs = np.stack(probs_list)
        labels = np.array(labels_list)

        diagram = compute_reliability_diagram(probs, labels, n_bins=n_bins)
        return {
            "bin_centers": diagram.bin_centers.tolist(),
            "bin_accuracies": diagram.bin_accuracies.tolist(),
            "bin_confidences": diagram.bin_confidences.tolist(),
            "bin_counts": diagram.bin_counts.tolist(),
            "n_bins": diagram.n_bins,
            "ece": diagram.ece,
            "provenance": "illustrative",
        }

    @app.get("/experiments/risk-coverage")
    def get_risk_coverage_curve(
        n_samples: int = 200,
        n_thresholds: int = 20,
        seed: int = 0,
    ):
        """Compute a risk-coverage curve from synthetic data."""
        from ..data.synthetic import generate_calibration_data
        data = generate_calibration_data(n_samples=n_samples, n_classes=5, feature_dim=32, seed=seed)

        model = NumpyMLP(input_dim=32, n_classes=5, seed=seed)
        probs_list, labels_list = [], []
        for x, y in data:
            logits = model(x)
            logits = np.asarray(logits).ravel()
            probs = np.exp(logits) / np.exp(logits).sum()
            probs_list.append(probs)
            labels_list.append(y)

        probs = np.stack(probs_list)
        labels = np.array(labels_list)
        predictions = probs.argmax(axis=-1)
        uncertainties = 1.0 - probs.max(axis=-1)

        curve = compute_risk_coverage_curve(uncertainties, predictions, labels, n_thresholds)
        return {
            "points": curve.to_dict(),
            "n_points": curve.n_points,
            "provenance": "illustrative",
        }

    @app.get("/experiments/multi-seed")
    def get_multi_seed_results(
        n_seeds: int = 3,
        n_calibration: int = 30,
        n_test: int = 30,
    ):
        """Run multi-seed evaluation and return mean ± std results."""
        config = HarnessConfig(
            n_calibration=n_calibration,
            n_test=n_test,
            n_classes=5,
            feature_dim=32,
            alpha=0.1,
            seed=0,
        )
        runner = MultiSeedRunner(n_seeds=n_seeds, base_config=config)
        runner.run()
        return runner.to_dict()

    @app.get("/experiments/fpr95")
    def get_fpr95(n_in_dist: int = 100, n_out_dist: int = 100, seed: int = 0):
        """Compute FPR95 for OOD detection using synthetic data."""
        rng = np.random.default_rng(seed)
        # In-dist uncertainties: low (model is confident)
        in_dist = rng.uniform(0, 0.3, n_in_dist)
        # OOD uncertainties: high (model is uncertain)
        out_dist = rng.uniform(0.5, 1.0, n_out_dist)
        result = fpr95(in_dist, out_dist)
        return {
            "fpr95": result,
            "n_in_dist": n_in_dist,
            "n_out_dist": n_out_dist,
            "provenance": "illustrative",
        }

    @app.get("/experiments/brier-decomposition")
    def get_brier_decomposition(n_samples: int = 200, seed: int = 0):
        """Compute Brier score decomposition from synthetic data."""
        from ..data.synthetic import generate_calibration_data
        data = generate_calibration_data(n_samples=n_samples, n_classes=5, feature_dim=32, seed=seed)
        model = NumpyMLP(input_dim=32, n_classes=5, seed=seed)
        probs_list, labels_list = [], []
        for x, y in data:
            logits = model(x)
            logits = np.asarray(logits).ravel()
            probs = np.exp(logits) / np.exp(logits).sum()
            probs_list.append(probs)
            labels_list.append(y)

        probs = np.stack(probs_list)
        labels = np.array(labels_list)
        decomp = brier_decomposition(probs, labels, n_bins=10)
        return {
            "brier": decomp.brier,
            "reliability": decomp.reliability,
            "resolution": decomp.resolution,
            "uncertainty": decomp.uncertainty,
            "provenance": "illustrative",
        }

    # ─── Cost-minimised deferral endpoint ─────────────────

    @app.post("/deferral/evaluate")
    def evaluate_deferral(request: dict):
        """Evaluate the cost-minimised deferral rule for a single sample.

        Expects a JSON body: {"probs": [0.9, 0.05, 0.02, 0.01, 0.01, 0.01]}

        Returns the expected cost of predicting vs. deferring, and whether
        the model should defer to a human supervisor.
        """
        probs = request.get("probs") if isinstance(request, dict) else request
        if not probs or not isinstance(probs, list):
            return {"error": "Expected 'probs' as a list of floats"}
        cm = CostMatrix.default_naval(n_classes=6)
        rule = CostMinimisingDeferral(cm)
        probs_arr = np.array(probs)
        result = rule.evaluate(probs_arr)
        return {
            "predicted_class": result.predicted_class,
            "confidence": result.confidence,
            "expected_cost_predict": result.expected_cost_predict,
            "expected_cost_defer": result.expected_cost_defer,
            "deferred": result.deferred,
            "top_2_classes": list(result.top_2_classes),
            "top_2_probs": list(result.top_2_probs),
            "cost_matrix": cm.costs.tolist(),
            "defer_cost": cm.defer_cost,
        }

    @app.get("/deferral/cost-matrix")
    def get_cost_matrix():
        """Return the default naval cost matrix."""
        cm = CostMatrix.default_naval(n_classes=6)
        return {
            "n_classes": cm.n_classes,
            "costs": cm.costs.tolist(),
            "defer_cost": cm.defer_cost,
            "ship_classes": SHIP_CLASSES,
            "description": {
                "0,4": "Predict cargo when actually naval → catastrophic miss (cost=10)",
                "4,0": "Predict naval when actually cargo → false alarm (cost=3)",
                "diagonal": "Correct prediction (cost=0)",
                "other": "Standard misclassification (cost=1)",
            },
        }

    return app
