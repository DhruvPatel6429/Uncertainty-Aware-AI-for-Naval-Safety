"""Scenario library — scripted maritime encounters for stress-testing.

Each scenario exercises a specific failure mode or COLREGs situation:
- ``head_on``, ``crossing``, ``overtaking``: COLREGs rules 13-15
- ``multi_vessel``: dense-traffic edge cases
- ``ood_stress``: distribution-shifted sensor conditions
- ``sensor_failure``: simulated radar/sonar degradation
- ``adversarial``: perturbed inputs (simulated spoofing)
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..data.synthetic import SyntheticScenarioGenerator


class ScenarioLibrary:
    """Library of scripted maritime scenarios for evaluation.

    Each scenario returns a list of sensor frames + ground-truth metadata.
    """

    SCENARIOS = SyntheticScenarioGenerator.SCENARIOS + ["sensor_failure", "adversarial"]

    def __init__(self, n_frames: int = 30, seed: int = 0):
        self.n_frames = n_frames
        self.generator = SyntheticScenarioGenerator(seed=seed)
        self.rng = np.random.default_rng(seed)

    def list_scenarios(self) -> list[str]:
        return list(self.SCENARIOS)

    def run_scenario(self, scenario: str) -> dict:
        """Run a single scenario and return frames + ground truth."""
        if scenario in SyntheticScenarioGenerator.SCENARIOS:
            frames = self.generator.generate(scenario, n_frames=self.n_frames)
        elif scenario == "sensor_failure":
            frames = self._generate_sensor_failure()
        elif scenario == "adversarial":
            frames = self._generate_adversarial()
        else:
            raise ValueError(f"Unknown scenario: {scenario}")

        return {
            "scenario": scenario,
            "n_frames": len(frames),
            "frames": frames,
            "expected_outcome": self._expected_outcome(scenario),
        }

    def run_all(self) -> list[dict]:
        """Run every scenario in the library."""
        return [self.run_scenario(s) for s in self.SCENARIOS]

    def _generate_sensor_failure(self) -> list[np.ndarray]:
        """Simulate radar degradation — frames progressively noisier."""
        frames = []
        for t in range(self.n_frames):
            noise_level = 0.1 + 0.5 * (t / self.n_frames)
            frame = self.rng.normal(0.5, noise_level, (3, 64, 64))
            frames.append(frame.astype(np.float32))
        return frames

    def _generate_adversarial(self) -> list[np.ndarray]:
        """Simulate adversarial perturbations on clean frames."""
        base = self.generator.generate("head_on", n_frames=self.n_frames)
        adversarial = []
        for frame in base:
            # FGSM-style perturbation (simulated)
            perturbation = self.rng.choice([-1, 1], size=frame.shape).astype(np.float32) * 0.1
            adversarial.append(np.clip(frame + perturbation, 0, 1))
        return adversarial

    def _expected_outcome(self, scenario: str) -> dict:
        """What the pipeline should do for this scenario."""
        outcomes = {
            "head_on":        {"action": "EXECUTE",          "colregs_critical": True},
            "crossing":       {"action": "EXECUTE",          "colregs_critical": True},
            "overtaking":     {"action": "EXECUTE",          "colregs_critical": False},
            "multi_vessel":  {"action": "EXECUTE",          "colregs_critical": True},
            "ood_stress":     {"action": "REJECT_HUMAN",     "colregs_critical": False},
            "sensor_failure": {"action": "EMERGENCY_FALLBACK", "colregs_critical": False},
            "adversarial":   {"action": "REJECT_HUMAN",     "colregs_critical": False},
        }
        return outcomes.get(scenario, {"action": "EXECUTE", "colregs_critical": False})
