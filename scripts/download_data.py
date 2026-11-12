"""CLI script — download / prepare datasets (synthetic for now).

Usage:
    python scripts/download_data.py --output data/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from uq_naval.data.synthetic import (
    SyntheticAISDataset,
    SyntheticDetectionDataset,
)
from uq_naval.utils.logging import configure_logging, get_logger


def main():
    parser = argparse.ArgumentParser(description="Prepare UQ-AI Naval datasets")
    parser.add_argument("--output", type=str, default="data/")
    parser.add_argument("--n-samples", type=int, default=500)
    args = parser.parse_args()

    configure_logging()
    log = get_logger("uq_naval.download_data")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Generating synthetic AIS dataset ({args.n_samples} samples)...")
    ais = SyntheticAISDataset(n_samples=args.n_samples, seed=0)
    log.info(f"  Done. histories shape: {ais._data[0].shape}")

    log.info(f"Generating synthetic detection dataset ({args.n_samples} samples)...")
    det = SyntheticDetectionDataset(n_samples=args.n_samples, seed=0)
    log.info(f"  Done. images shape: {det._images.shape}")

    # Save as .npz
    npz_path = out_dir / "synthetic_maritime.npz"
    import numpy as np
    np.savez_compressed(
        npz_path,
        ais_histories=ais._data[0],
        ais_futures=ais._data[1],
        ais_labels=ais._data[2],
        detection_images=det._images,
        detection_labels=det._labels,
        detection_is_ood=det._is_ood,
    )
    log.info(f"Saved synthetic dataset to: {npz_path}")
    log.info("Note: In production, replace with real MAS500 / MarineTraffic data.")


if __name__ == "__main__":
    main()
