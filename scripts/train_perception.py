"""CLI script — train the perception module.

Usage:
    python scripts/train_perception.py \\
        --config configs/pipeline/perception.yaml \\
        --uq-method bayesian \\
        --epochs 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running without installation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from uq_naval.uq import load_uq_method
from uq_naval.models.backbones import NumpyMLP, SimpleMLP
from uq_naval.data.synthetic import generate_calibration_data
from uq_naval.utils.seeds import set_global_seed
from uq_naval.utils.logging import configure_logging, get_logger


def main():
    parser = argparse.ArgumentParser(description="Train UQ-AI Naval perception module")
    parser.add_argument("--config", type=str, default="configs/pipeline/perception.yaml")
    parser.add_argument("--uq-method", type=str, default="bayesian")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--n-calibration", type=int, default=200)
    parser.add_argument("--n-classes", type=int, default=5)
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_global_seed(args.seed)
    configure_logging()
    log = get_logger("uq_naval.train_perception")

    log.info(f"Training perception module with UQ method: {args.uq_method}")
    log.info(f"Args: {vars(args)}")

    # Build model (numpy MLP — torch would be used in production)
    model = NumpyMLP(input_dim=args.feature_dim, n_classes=args.n_classes, seed=args.seed)

    # Build UQ method
    uq = load_uq_method(args.uq_method, model=model, config={
        "n_mc_samples": 10,
        "reject_threshold": 0.5,
        "alpha": 0.1,
    })
    log.info(f"UQ method: {uq}")

    # Generate calibration data
    cal_data = generate_calibration_data(
        n_samples=args.n_calibration,
        n_classes=args.n_classes,
        feature_dim=args.feature_dim,
        seed=args.seed,
    )

    # Calibrate
    result = uq.calibrate(cal_data)
    log.info(f"Calibration result: {result}")

    # Run a single prediction to verify
    test_x = cal_data[0][0]
    out = uq.predict(test_x)
    log.info(f"Sample prediction: {out}")

    print(json.dumps({
        "method": args.uq_method,
        "calibration_metrics": result.metrics,
        "n_calibration": result.n_calibration,
        "sample_prediction": out.to_dict(),
    }, indent=2))


if __name__ == "__main__":
    main()
