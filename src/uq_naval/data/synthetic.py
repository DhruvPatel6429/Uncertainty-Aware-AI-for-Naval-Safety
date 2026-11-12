"""Synthetic maritime data generators.

Used for testing, demos, and bootstrapping when real maritime data is
unavailable. The generators produce realistic distributions (Gaussian
trajectories, multi-modal encounters) that exercise the UQ methods
without requiring fielded data.
"""
from __future__ import annotations

from typing import Iterator, Optional

import numpy as np


class SyntheticAISDataset:
    """Synthetic AIS-like vessel trajectory dataset.

    Generates (history, future) trajectory pairs with realistic kinematics:
    - 4 features per timestep: lat_rate, lon_rate, heading_delta, speed
    - Future trajectories include multi-modal encounters (turn left/right/hold).
    """

    def __init__(
        self,
        n_samples: int = 1000,
        history_len: int = 10,
        future_len: int = 30,
        feature_dim: int = 4,
        n_vessel_classes: int = 5,
        seed: int = 0,
    ):
        self.n_samples = n_samples
        self.history_len = history_len
        self.future_len = future_len
        self.feature_dim = feature_dim
        self.n_vessel_classes = n_vessel_classes
        self.rng = np.random.default_rng(seed)
        self._data = self._generate()

    def _generate(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        histories = np.zeros((self.n_samples, self.history_len, self.feature_dim))
        futures = np.zeros((self.n_samples, self.future_len, 2))
        labels = np.zeros(self.n_samples, dtype=int)

        for i in range(self.n_samples):
            # Random vessel class
            vessel_class = self.rng.integers(0, self.n_vessel_classes)
            labels[i] = vessel_class

            # Base kinematics depend on class (cargo = slow turns, speedboat = fast)
            base_speed = 5.0 + vessel_class * 3.0
            turn_rate = 0.5 + vessel_class * 0.5

            # Generate history
            for t in range(self.history_len):
                histories[i, t, 0] = self.rng.normal(0, 0.001)  # lat rate
                histories[i, t, 1] = self.rng.normal(0, 0.001)  # lon rate
                histories[i, t, 2] = self.rng.normal(0, turn_rate)  # heading delta
                histories[i, t, 3] = base_speed + self.rng.normal(0, 1.0)  # speed

            # Generate future — multi-modal (random turn left/right/hold)
            mode = self.rng.choice([0, 1, 2])  # 0 = left, 1 = hold, 2 = right
            for t in range(self.future_len):
                turn = {-1: -turn_rate, 0: 0.0, 1: turn_rate}[mode - 1]
                futures[i, t, 0] = turn
                futures[i, t, 1] = base_speed + self.rng.normal(0, 0.5)

        return histories, futures, labels

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, int]:
        h, f, l = self._data
        return h[idx], f[idx], int(l[idx])

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray, int]]:
        for i in range(self.n_samples):
            yield self[i]


class SyntheticDetectionDataset:
    """Synthetic maritime object detection dataset.

    Generates (image, label) pairs where images are small synthetic radar/EO
    scenes containing a single vessel class. Used to test the perception
    pipeline without real maritime data.
    """

    def __init__(
        self,
        n_samples: int = 500,
        image_size: tuple[int, int, int] = (3, 32, 32),
        n_classes: int = 5,
        seed: int = 0,
        inject_ood: float = 0.1,
    ):
        self.n_samples = n_samples
        self.image_size = image_size
        self.n_classes = n_classes
        self.rng = np.random.default_rng(seed)
        self.inject_ood = inject_ood
        self._images, self._labels, self._is_ood = self._generate()

    def _generate(self):
        images = np.zeros((self.n_samples, *self.image_size), dtype=np.float32)
        labels = np.zeros(self.n_samples, dtype=int)
        is_ood = np.zeros(self.n_samples, dtype=bool)

        for i in range(self.n_samples):
            label = self.rng.integers(0, self.n_classes)
            labels[i] = label
            # Generate a class-dependent pattern in the image
            base = self.rng.normal(0, 0.1, self.image_size)
            # Add a "vessel" blob whose intensity depends on class
            cx, cy = self.rng.integers(8, 24, 2)
            base[:, cx, cy] += 0.5 + 0.1 * label
            images[i] = base
            # Inject OOD samples
            if self.rng.random() < self.inject_ood:
                # OOD: noise pattern not seen in training
                images[i] = self.rng.normal(0.5, 0.5, self.image_size)
                is_ood[i] = True

        return images, labels, is_ood

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        return self._images[idx], int(self._labels[idx])

    def __iter__(self) -> Iterator[tuple[np.ndarray, int]]:
        for i in range(self.n_samples):
            yield self[i]


class SyntheticScenarioGenerator:
    """Generates scripted maritime encounter scenarios for stress-testing.

    Scenarios include:
    - ``head_on``: two vessels approaching head-on (COLREGs Rule 14)
    - ``crossing``: vessels on crossing courses (Rule 15)
    - ``overtaking``: one vessel overtaking another (Rule 13)
    - ``multi_vessel``: 3+ vessels in dense traffic
    - ``ood_stress``: distribution-shifted sensor conditions
    """

    SCENARIOS = ["head_on", "crossing", "overtaking", "multi_vessel", "ood_stress"]

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def generate(self, scenario: str, n_frames: int = 30) -> list[np.ndarray]:
        if scenario not in self.SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario}. Available: {self.SCENARIOS}")

        frames = []
        for t in range(n_frames):
            frame = self._generate_frame(scenario, t, n_frames)
            frames.append(frame)
        return frames

    def _generate_frame(self, scenario: str, t: int, n_frames: int) -> np.ndarray:
        # Synthetic frame: (3, 64, 64) image with vessel blobs
        frame = np.zeros((3, 64, 64), dtype=np.float32)
        progress = t / n_frames

        if scenario == "head_on":
            # Two blobs approaching from opposite sides
            x1 = int(10 + 20 * progress)
            x2 = int(50 - 20 * progress)
            frame[:, 30, x1] = 1.0
            frame[:, 30, x2] = 1.0
        elif scenario == "crossing":
            # Two blobs on crossing courses
            x1 = int(20 * progress)
            y1 = int(10 + 30 * progress)
            x2 = int(50 - 20 * progress)
            y2 = int(40 - 20 * progress)
            if 0 <= y1 < 64:
                frame[:, y1, x1] = 1.0
            if 0 <= y2 < 64:
                frame[:, y2, x2] = 1.0
        elif scenario == "overtaking":
            # One blob catching up to another
            x_slow = int(20 + 5 * progress)
            x_fast = int(10 + 30 * progress)
            frame[:, 30, x_slow] = 0.7
            frame[:, 30, x_fast] = 1.0
        elif scenario == "multi_vessel":
            # 3+ vessels
            for i in range(4):
                x = int(10 + i * 15 + 5 * progress)
                y = int(20 + i * 8)
                if 0 <= x < 64 and 0 <= y < 64:
                    frame[:, y, x] = 1.0
        elif scenario == "ood_stress":
            # Inject heavy noise (simulates fog/jamming)
            frame = self.rng.normal(0.5, 0.3, (3, 64, 64))
        return frame


def generate_calibration_data(
    n_samples: int = 200,
    n_classes: int = 5,
    feature_dim: int = 32,
    seed: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate (features, label) pairs suitable for UQ calibration.

    Returns a list of (x, y) batches where each batch is a single example.
    """
    rng = np.random.default_rng(seed)
    data = []
    for _ in range(n_samples):
        x = rng.normal(0, 1, feature_dim)
        y = int(rng.integers(0, n_classes))
        data.append((x.astype(np.float32), y))
    return data
