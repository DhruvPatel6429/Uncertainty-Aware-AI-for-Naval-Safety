"""Data loaders for maritime datasets.

Provides unified interfaces to:
- AIS vessel trajectory traces (synthetic generator for testing)
- MAS500 maritime object detection dataset (synthetic generator)
- Synthetic maritime scenario generator for stress-testing
"""
from .synthetic import (
    SyntheticAISDataset,
    SyntheticDetectionDataset,
    SyntheticScenarioGenerator,
    generate_calibration_data,
)
from .ais import AISLoader
from .mas500 import MAS500Loader

__all__ = [
    "SyntheticAISDataset",
    "SyntheticDetectionDataset",
    "SyntheticScenarioGenerator",
    "generate_calibration_data",
    "AISLoader",
    "MAS500Loader",
]
