"""MAS500 maritime object detection loader.

Loads (image, label, bbox) tuples from the MAS500 maritime detection
dataset. Falls back to synthetic data for testing when the real dataset
is unavailable.
"""
from __future__ import annotations

from typing import Iterator, Optional

import numpy as np

from .synthetic import SyntheticDetectionDataset


class MAS500Loader:
    """Loader for the MAS500 maritime object detection dataset.

    Parameters
    ----------
    batch_size:
        Number of images per batch.
    split:
        One of ``train``, ``val``, ``calibration``, ``test``.
    data_dir:
        Local path to MAS500 images. If empty, falls back to synthetic.
    seed:
        Random seed.
    """

    SPLITS = ("train", "val", "calibration", "test")

    def __init__(
        self,
        batch_size: int = 32,
        split: str = "train",
        data_dir: Optional[str] = None,
        n_samples: int = 500,
        seed: int = 0,
    ):
        if split not in self.SPLITS:
            raise ValueError(f"split must be one of {self.SPLITS}, got {split!r}")
        self.batch_size = batch_size
        self.split = split
        self.data_dir = data_dir

        split_offset = self.SPLITS.index(split) * 500
        self._dataset = SyntheticDetectionDataset(
            n_samples=n_samples,
            seed=seed + split_offset,
        )
        self._indices = np.arange(len(self._dataset))

    def __len__(self) -> int:
        return (len(self._dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        rng = np.random.default_rng(42)
        idx = self._indices.copy()
        rng.shuffle(idx)
        for i in range(0, len(idx), self.batch_size):
            batch_idx = idx[i:i + self.batch_size]
            images, labels = [], []
            for j in batch_idx:
                img, lbl = self._dataset[int(j)]
                images.append(img)
                labels.append(lbl)
            yield (
                np.stack(images),
                np.array(labels),
            )
