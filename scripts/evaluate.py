"""CLI script — run the evaluation harness on all UQ methods.

Usage:
    python scripts/evaluate.py \\
        --config configs/eval/default.yaml \\
        --output reports/evaluation.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from uq_naval.evaluation import EvaluationHarness
from uq_naval.evaluation.harness import HarnessConfig
from uq_naval.utils.seeds import set_global_seed
from uq_naval.utils.logging import configure_logging, get_logger


def main():
    parser = argparse.ArgumentParser(description="Run UQ-AI Naval evaluation harness")
    parser.add_argument("--config", type=str, default="configs/eval/default.yaml")
    parser.add_argument("--output", type=str, default="reports/evaluation.json")
    parser.add_argument("--methods", type=str, nargs="*", default=None,
                        help="Subset of UQ methods to evaluate. Default: all.")
    parser.add_argument("--n-calibration", type=int, default=100)
    parser.add_argument("--n-test", type=int, default=100)
    parser.add_argument("--n-classes", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_global_seed(args.seed)
    configure_logging()
    log = get_logger("uq_naval.evaluate")

    config = HarnessConfig(
        n_calibration=args.n_calibration,
        n_test=args.n_test,
        n_classes=args.n_classes,
        alpha=args.alpha,
        seed=args.seed,
    )
    harness = EvaluationHarness(config=config)

    log.info(f"Running harness on methods: {args.methods or 'all'}")
    results = harness.run(method_names=args.methods)
    log.info(f"Harness complete. {len(results)} methods evaluated.")

    # Write JSON output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(harness.to_dict(), f, indent=2)

    # Print summary table
    print("\n" + "=" * 80)
    print("UQ-AI Naval Evaluation Harness — Summary")
    print("=" * 80)
    print(harness.summary_table())
    print(f"\nFull results written to: {output_path}")


if __name__ == "__main__":
    main()
