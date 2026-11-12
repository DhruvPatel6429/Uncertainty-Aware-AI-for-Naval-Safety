"""Multi-seed evaluation runner.

Runs the evaluation harness across multiple random seeds and reports
mean ± standard deviation for every metric, following the rigorous
experimental protocol described in the proposal (3-5 runs with different seeds).

Usage:
    >>> runner = MultiSeedRunner(n_seeds=5)
    >>> results = runner.run()
    >>> print(results.summary_table())
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..uq import list_uq_methods
from ..evaluation.harness import EvaluationHarness, HarnessConfig, MethodResult
from ..utils.seeds import set_global_seed
from ..utils.logging import get_logger


@dataclass
class SeedRunResult:
    """Result of a single seed run."""
    seed: int
    method_results: list[MethodResult]


@dataclass
class MultiSeedSummary:
    """Aggregated results across multiple seeds for a single method."""
    method_name: str
    n_seeds: int
    metric_means: dict[str, float]    # mean across seeds
    metric_stds: dict[str, float]     # std across seeds
    metric_values: dict[str, list[float]]  # all values per seed

    def to_dict(self) -> dict:
        return {
            "method": self.method_name,
            "n_seeds": self.n_seeds,
            "means": self.metric_means,
            "stds": self.metric_stds,
        }


class MultiSeedRunner:
    """Run the evaluation harness across multiple seeds.

    For each UQ method, runs the evaluation N times with different random seeds
    and reports mean ± std for every metric.

    This is the rigorous experimental protocol: 3-5 runs with different seeds,
    reported as mean ± standard deviation.
    """

    def __init__(
        self,
        n_seeds: int = 5,
        base_config: Optional[HarnessConfig] = None,
        method_names: Optional[list[str]] = None,
    ):
        self.n_seeds = n_seeds
        self.base_config = base_config or HarnessConfig()
        self.method_names = method_names  # None = all methods
        self.logger = get_logger("uq_naval.multiseed")
        self.runs: list[SeedRunResult] = []

    def run(self) -> list[MultiSeedSummary]:
        """Run the harness across all seeds and return aggregated summaries."""
        self.runs = []
        seeds = list(range(self.base_config.seed, self.base_config.seed + self.n_seeds))

        for seed in seeds:
            self.logger.info(f"Running seed {seed}...")
            set_global_seed(seed)

            config = HarnessConfig(
                n_calibration=self.base_config.n_calibration,
                n_test=self.base_config.n_test,
                n_classes=self.base_config.n_classes,
                feature_dim=self.base_config.feature_dim,
                alpha=self.base_config.alpha,
                seed=seed,
            )
            harness = EvaluationHarness(config=config)
            results = harness.run(method_names=self.method_names)

            self.runs.append(SeedRunResult(
                seed=seed,
                method_results=results,
            ))

        return self._aggregate()

    def _aggregate(self) -> list[MultiSeedSummary]:
        """Aggregate results across seeds for each method."""
        if not self.runs:
            return []

        # Collect all method names from the first run
        method_names = [r.method_name for r in self.runs[0].method_results]

        summaries: list[MultiSeedSummary] = []
        for method_name in method_names:
            # Collect metric values across seeds for this method
            metric_values: dict[str, list[float]] = {}

            for run in self.runs:
                method_result = next(
                    (r for r in run.method_results if r.method_name == method_name),
                    None,
                )
                if method_result is None:
                    continue

                # Combine calibration + test metrics
                all_metrics = {**method_result.calibration_metrics, **method_result.test_metrics}
                for key, value in all_metrics.items():
                    if isinstance(value, (int, float)):
                        metric_values.setdefault(key, []).append(float(value))

            # Compute mean ± std
            means: dict[str, float] = {}
            stds: dict[str, float] = {}
            for key, values in metric_values.items():
                if values:
                    means[key] = float(np.mean(values))
                    stds[key] = float(np.std(values))

            summaries.append(MultiSeedSummary(
                method_name=method_name,
                n_seeds=len(self.runs),
                metric_means=means,
                metric_stds=stds,
                metric_values=metric_values,
            ))

        return summaries

    def summary_table(self) -> str:
        """Render aggregated results as a Markdown table with mean ± std."""
        summaries = self._aggregate()
        if not summaries:
            return "_(no results — run the runner first)_"

        lines = [
            f"# Multi-Seed Evaluation Results ({self.n_seeds} seeds)",
            "",
            "| Method | ECE ↓ | Brier ↓ | Coverage | Reject Rate |",
            "|--------|-------|--------|----------|-------------|",
        ]

        for s in summaries:
            ece_mean = s.metric_means.get("ece", float("nan"))
            ece_std = s.metric_stds.get("ece", 0.0)
            brier_mean = s.metric_means.get("brier", float("nan"))
            brier_std = s.metric_stds.get("brier", 0.0)
            cov_mean = s.metric_means.get("coverage", float("nan"))
            cov_std = s.metric_stds.get("coverage", 0.0)
            reject_mean = s.metric_means.get("rejection_rate", float("nan"))
            reject_std = s.metric_stds.get("rejection_rate", 0.0)

            # Skip methods that errored on all seeds
            if np.isnan(ece_mean) and np.isnan(brier_mean):
                lines.append(f"| `{s.method_name}` | _error_ | — | — | — |")
                continue

            lines.append(
                f"| `{s.method_name}` | "
                f"{ece_mean:.4f} ± {ece_std:.4f} | "
                f"{brier_mean:.4f} ± {brier_std:.4f} | "
                f"{cov_mean:.4f} ± {cov_std:.4f} | "
                f"{reject_mean:.4f} ± {reject_std:.4f} |"
            )

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialise to a dict for JSON output."""
        return {
            "n_seeds": self.n_seeds,
            "base_config": {
                "n_calibration": self.base_config.n_calibration,
                "n_test": self.base_config.n_test,
                "alpha": self.base_config.alpha,
            },
            "summaries": [s.to_dict() for s in self._aggregate()],
        }
