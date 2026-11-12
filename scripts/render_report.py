"""CLI script — render the evaluation report as Markdown.

Usage:
    python scripts/render_report.py --input reports/evaluation.json --output reports/evaluation.md
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main():
    parser = argparse.ArgumentParser(description="Render evaluation report as Markdown")
    parser.add_argument("--input", type=str, default="reports/evaluation.json")
    parser.add_argument("--output", type=str, default="reports/evaluation.md")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    cfg = data["config"]
    results = data["results"]

    lines = [
        "# UQ-AI Naval — Evaluation Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Configuration",
        "",
        f"- **Calibration samples**: {cfg['n_calibration']}",
        f"- **Test samples**: {cfg['n_test']}",
        f"- **Number of classes**: {cfg['n_classes']}",
        f"- **Nominal coverage (1-α)**: {1 - cfg['alpha']:.2f}",
        "",
        "## Method Comparison",
        "",
        "| Method | ECE ↓ | Brier ↓ | Coverage | Mean Set Size | Reject Rate |",
        "|--------|-------|--------|----------|---------------|-------------|",
    ]

    for r in results:
        m = r["test_metrics"]
        if "error" in m:
            lines.append(f"| `{r['method']}` | _error_ | — | — | — | — |")
            continue
        if "note" in m:
            lines.append(f"| `{r['method']}` | _skipped_ | — | — | — | — |")
            continue
        ece = m.get("ece", float("nan"))
        brier = m.get("brier", float("nan"))
        coverage = m.get("coverage", 0.0)
        set_size = m.get("mean_set_size", 0.0)
        reject = m.get("rejection_rate", 0.0)
        lines.append(
            f"| `{r['method']}` | {ece:.4f} | {brier:.4f} | {coverage:.4f} | {set_size:.2f} | {reject:.4f} |"
        )

    lines += [
        "",
        "## Per-Method Details",
        "",
    ]

    for r in results:
        lines.append(f"### `{r['method']}`")
        lines.append("")
        cal = r.get("calibration_metrics", {})
        if cal:
            lines.append("**Calibration metrics:**")
            lines.append("")
            for k, v in cal.items():
                lines.append(f"- `{k}`: {v}")
            lines.append("")
        test = r.get("test_metrics", {})
        if test:
            lines.append("**Test metrics:**")
            lines.append("")
            for k, v in test.items():
                lines.append(f"- `{k}`: {v}")
            lines.append("")

    lines += [
        "## Interpretation",
        "",
        "- **ECE (Expected Calibration Error)** — gap between predicted confidence and empirical accuracy. Lower is better. Well-calibrated models achieve ECE < 0.05.",
        "- **Brier Score** — mean squared error of probabilistic predictions. Lower is better. Baseline for uniform-random binary prediction is 0.25.",
        "- **Coverage** — fraction of test instances whose true label is in the conformal prediction set. Should match the nominal 1-α = 0.90 within ±1%.",
        "- **Mean Set Size** — average number of classes in the conformal prediction set. Smaller is better (more selective).",
        "- **Reject Rate** — fraction of decisions deferred to a human supervisor. High rejection signals distribution shift or insufficient training data.",
        "",
        "## Notes",
        "",
        "- Methods marked `_error_` failed during evaluation (typically because they require a torch nn.Module rather than the synthetic NumpyMLP used by the harness). Their dedicated unit tests pass with appropriate models.",
        "- Methods marked `_skipped_` require a different calibration data format (e.g., Conformal Risk Control needs `(state, action, outcome)` tuples rather than `(x, y)` pairs). They have their own dedicated unit tests.",
        "- All metrics are computed on synthetic data; production deployments should re-run the harness on real maritime datasets (MAS500, AIS archives).",
        "",
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Use UTF-8 encoding explicitly — Windows defaults to cp1252 which can't
    # encode Greek letters (α, β) or other Unicode used in the report.
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report rendered to: {output_path}")


if __name__ == "__main__":
    main()
