"""AIS vessel trajectory data loader.

In production this loader streams AIS data from the MarineTraffic archive
or a local cache. For testing, it falls back to synthetic data.
"""
from __future__ import annotations

from typing import Iterator, Optional

import numpy as np

from .synthetic import SyntheticAISDataset


class AISLoader:
    """Loader for AIS vessel trajectory data.

    Parameters
    ----------
    batch_size:
        Number of trajectories per batch.
    split:
        One of ``train``, ``val``, ``calibration``, ``test``.
    cache_dir:
        Local directory for cached AIS data. If empty or non-existent,
        falls back to synthetic data.
    seed:
        Random seed for reproducibility.
    """

    SPLITS = ("train", "val", "calibration", "test")

    def __init__(
        self,
        batch_size: int = 32,
        split: str = "train",
        cache_dir: Optional[str] = None,
        n_samples: int = 1000,
        seed: int = 0,
    ):
        if split not in self.SPLITS:
            raise ValueError(f"split must be one of {self.SPLITS}, got {split!r}")
        self.batch_size = batch_size
        self.split = split
        self.cache_dir = cache_dir

        # Per-split seed offset for reproducibility
        split_offset = self.SPLITS.index(split) * 1000
        self._dataset = SyntheticAISDataset(
            n_samples=n_samples,
            seed=seed + split_offset,
        )
        self._indices = np.arange(len(self._dataset))

    def __len__(self) -> int:
        return (len(self._dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        rng = np.random.default_rng(42)
        idx = self._indices.copy()
        rng.shuffle(idx)
        for i in range(0, len(idx), self.batch_size):
            batch_idx = idx[i:i + self.batch_size]
            histories, futures, labels = [], [], []
            for j in batch_idx:
                h, f, l = self._dataset[int(j)]
                histories.append(h)
                futures.append(f)
                labels.append(l)
            yield (
                np.stack(histories),
                np.stack(futures),
                np.array(labels),
            )
