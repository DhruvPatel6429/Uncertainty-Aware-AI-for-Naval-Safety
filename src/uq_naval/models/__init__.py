"""Neural network architectures for naval autonomy.

This package provides model classes that the UQ methods wrap. Each model is
a standard PyTorch ``nn.Module``; UQ is applied externally by the methods in
``uq_naval.uq``.
"""
from .backbones import SmallCNN, BayesLinear, SimpleMLP
from .detectors import BayesianDetector, SimpleDetector
from .forecasters import TrajectoryLSTM, TrajectoryTransformer, EvidentialForecaster

__all__ = [
    "SmallCNN",
    "BayesLinear",
    "SimpleMLP",
    "BayesianDetector",
    "SimpleDetector",
    "TrajectoryLSTM",
    "TrajectoryTransformer",
    "EvidentialForecaster",
]
