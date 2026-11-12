"""Dataset splitting protocols for ship classification.

Implements the four-way split (train / validation / calibration / test) with
two critical anti-leakage safeguards:

1. **Video-aware / vessel-aware splitting** — many ship datasets are frames
   extracted from video, so a random split puts near-identical frames in both
   train and test, inflating results. We split by video ID or vessel ID
   instead, ensuring all frames from the same vessel go to the same split.

2. **Hold-out-class OOD protocol** — we hold out whole ship classes during
   training and show them only at test time. This tests whether the model
   flags things it has never seen.

The four splits serve distinct purposes:
- **Train**: used to fit model weights
- **Validation**: used for hyperparameter tuning and early stopping
- **Calibration**: held out — never used for training; used to fit temperature
  scaling and conformal quantiles
- **Test**: held out — never used for training or calibration; used for final
  evaluation
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


# ───────────────────────────────────────────────────────────
# Split configuration
# ───────────────────────────────────────────────────────────

@dataclass
class SplitConfig:
    """Configuration for dataset splitting."""
    # Split proportions (must sum to 1.0)
    train_frac: float = 0.60
    val_frac: float = 0.10
    calibration_frac: float = 0.10
    test_frac: float = 0.20

    # Split strategy
    split_by: str = "vessel"  # "random" | "vessel" | "video"
    hold_out_classes: list[str] = field(default_factory=list)  # OOD classes
    seed: int = 0

    def __post_init__(self):
        total = self.train_frac + self.val_frac + self.calibration_frac + self.test_frac
        if not np.isclose(total, 1.0):
            raise ValueError(f"Split fractions must sum to 1.0, got {total}")

    @property
    def split_ratios(self) -> dict[str, float]:
        return {
            "train": self.train_frac,
            "val": self.val_frac,
            "calibration": self.calibration_frac,
            "test": self.test_frac,
        }


# ───────────────────────────────────────────────────────────
# Sample / metadata dataclasses
# ───────────────────────────────────────────────────────────

@dataclass
class ShipSample:
    """A single ship image sample with metadata."""
    image_path: str
    label: str                  # ship class name
    vessel_id: str               # unique vessel identifier (for vessel-aware split)
    video_id: Optional[str] = None  # video identifier (for video-aware split)
    class_id: int = 0            # numeric class index


@dataclass
class SplitResult:
    """Result of a dataset split."""
    train: list[ShipSample]
    val: list[ShipSample]
    calibration: list[ShipSample]
    test: list[ShipSample]
    held_out_classes: list[str] = field(default_factory=list)
    split_by: str = "random"

    @property
    def n_samples(self) -> int:
        return len(self.train) + len(self.val) + len(self.calibration) + len(self.test)

    def summary(self) -> dict:
        from collections import Counter
        return {
            "split_by": self.split_by,
            "n_train": len(self.train),
            "n_val": len(self.val),
            "n_calibration": len(self.calibration),
            "n_test": len(self.test),
            "total": self.n_samples,
            "held_out_classes": self.held_out_classes,
            "train_class_distribution": dict(Counter(s.label for s in self.train)),
            "test_class_distribution": dict(Counter(s.label for s in self.test)),
        }


# ───────────────────────────────────────────────────────────
# Splitter
# ───────────────────────────────────────────────────────────

class DatasetSplitter:
    """Split a ship dataset into train / val / calibration / test.

    Supports three split strategies:

    - ``random`` — random split (NOT recommended — risks leakage)
    - ``vessel`` — split by vessel ID (recommended for single-frame datasets)
    - ``video`` — split by video ID (recommended for video-derived datasets)

    Also supports hold-out-class OOD protocol: specified classes are excluded
    from training and used only in the test split.
    """

    def __init__(self, config: SplitConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    def split(self, samples: list[ShipSample]) -> SplitResult:
        """Split the sample list into four non-overlapping subsets.

        If ``hold_out_classes`` is set, all samples of those classes go to the
        test split (OOD evaluation), and the remaining classes are split
        across train/val/calibration/test.
        """
        # Separate held-out-class samples from the rest
        held_out_samples: list[ShipSample] = []
        remaining_samples: list[ShipSample] = []
        for s in samples:
            if s.label in self.config.hold_out_classes:
                held_out_samples.append(s)
            else:
                remaining_samples.append(s)

        # Split remaining samples by the configured strategy
        if self.config.split_by == "random":
            train, val, cal, test = self._split_random(remaining_samples)
        elif self.config.split_by == "vessel":
            train, val, cal, test = self._split_by_vessel(remaining_samples)
        elif self.config.split_by == "video":
            train, val, cal, test = self._split_by_video(remaining_samples)
        else:
            raise ValueError(f"Unknown split_by: {self.config.split_by!r}")

        # Append held-out-class samples to test
        test.extend(held_out_samples)

        return SplitResult(
            train=train,
            val=val,
            calibration=cal,
            test=test,
            held_out_classes=list(self.config.hold_out_classes),
            split_by=self.config.split_by,
        )

    def _split_random(self, samples: list[ShipSample]) -> tuple[list, list, list, list]:
        """Random split — NOT recommended (risks leakage)."""
        indices = np.arange(len(samples))
        self.rng.shuffle(indices)

        n_train = int(len(samples) * self.config.train_frac)
        n_val = int(len(samples) * self.config.val_frac)
        n_cal = int(len(samples) * self.config.calibration_frac)

        train = [samples[i] for i in indices[:n_train]]
        val = [samples[i] for i in indices[n_train:n_train+n_val]]
        cal = [samples[i] for i in indices[n_train+n_val:n_train+n_val+n_cal]]
        test = [samples[i] for i in indices[n_train+n_val+n_cal:]]
        return train, val, cal, test

    def _split_by_vessel(self, samples: list[ShipSample]) -> tuple[list, list, list, list]:
        """Split by vessel ID — all frames from the same vessel go to the same split.

        This is the recommended split strategy for single-frame ship datasets
        because it prevents near-identical frames of the same vessel from
        appearing in both train and test.
        """
        # Group samples by vessel_id
        vessel_groups: dict[str, list[ShipSample]] = {}
        for s in samples:
            vessel_groups.setdefault(s.vessel_id, []).append(s)

        # Shuffle vessel IDs
        vessel_ids = list(vessel_groups.keys())
        self.rng.shuffle(vessel_ids)

        n_vessels = len(vessel_ids)
        n_train = int(n_vessels * self.config.train_frac)
        n_val = int(n_vessels * self.config.val_frac)
        n_cal = int(n_vessels * self.config.calibration_frac)

        train_vessels = set(vessel_ids[:n_train])
        val_vessels = set(vessel_ids[n_train:n_train+n_val])
        cal_vessels = set(vessel_ids[n_train+n_val:n_train+n_val+n_cal])
        test_vessels = set(vessel_ids[n_train+n_val+n_cal:])

        train = [s for vid in train_vessels for s in vessel_groups[vid]]
        val = [s for vid in val_vessels for s in vessel_groups[vid]]
        cal = [s for vid in cal_vessels for s in vessel_groups[vid]]
        test = [s for vid in test_vessels for s in vessel_groups[vid]]
        return train, val, cal, test

    def _split_by_video(self, samples: list[ShipSample]) -> tuple[list, list, list, list]:
        """Split by video ID — all frames from the same video go to the same split.

        This is the recommended split strategy for video-derived datasets
        (e.g., SeaShips, OpenSARShip) because adjacent video frames are
        near-identical and would inflate test accuracy if split randomly.
        """
        # Group samples by video_id (fall back to vessel_id if missing)
        video_groups: dict[str, list[ShipSample]] = {}
        for s in samples:
            vid = s.video_id or s.vessel_id
            video_groups.setdefault(vid, []).append(s)

        video_ids = list(video_groups.keys())
        self.rng.shuffle(video_ids)

        n_videos = len(video_ids)
        n_train = int(n_videos * self.config.train_frac)
        n_val = int(n_videos * self.config.val_frac)
        n_cal = int(n_videos * self.config.calibration_frac)

        train_videos = set(video_ids[:n_train])
        val_videos = set(video_ids[n_train:n_train+n_val])
        cal_videos = set(video_ids[n_train+n_val:n_train+n_val+n_cal])
        test_videos = set(video_ids[n_train+n_val+n_cal:])

        train = [s for vid in train_videos for s in video_groups[vid]]
        val = [s for vid in val_videos for s in video_groups[vid]]
        cal = [s for vid in cal_videos for s in video_groups[vid]]
        test = [s for vid in test_videos for s in video_groups[vid]]
        return train, val, cal, test


# ───────────────────────────────────────────────────────────
# Helper: generate synthetic ship samples for testing
# ───────────────────────────────────────────────────────────

def generate_synthetic_ship_samples(
    n_samples: int = 1000,
    n_classes: int = 6,
    n_vessels_per_class: int = 10,
    n_frames_per_vessel: int = 10,
    seed: int = 0,
) -> list[ShipSample]:
    """Generate synthetic ship sample metadata for testing the splitter.

    Each "vessel" produces n_frames_per_vessel samples, simulating a video
    derived dataset.
    """
    rng = np.random.default_rng(seed)
    samples: list[ShipSample] = []

    class_names = SHIP_CLASSES[:n_classes] if n_classes <= len(SHIP_CLASSES) else [f"class_{i}" for i in range(n_classes)]

    for class_id, class_name in enumerate(class_names):
        for vessel_idx in range(n_vessels_per_class):
            vessel_id = f"{class_name}_v{vessel_idx:03d}"
            # Each vessel has a few videos
            n_videos = max(1, n_vessels_per_class // 3)
            for vid_idx in range(n_videos):
                video_id = f"{vessel_id}_vid{vid_idx}"
                n_frames = max(1, n_frames_per_vessel // n_videos)
                for frame_idx in range(n_frames):
                    samples.append(ShipSample(
                        image_path=f"synthetic/{vessel_id}/{video_id}/frame_{frame_idx:04d}.jpg",
                        label=class_name,
                        vessel_id=vessel_id,
                        video_id=video_id,
                        class_id=class_id,
                    ))

    # Shuffle and truncate to n_samples if needed
    rng.shuffle(samples)
    return samples[:n_samples] if len(samples) > n_samples else samples


# Late import to avoid circular dependency
from ..models.ship_models.resnet import SHIP_CLASSES
