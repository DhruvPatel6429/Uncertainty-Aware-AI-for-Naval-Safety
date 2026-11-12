"""Ship classification models.

Provides ResNet-50 pretrained on ImageNet and fine-tuned on ship datasets —
the standard, proven backbone. The novelty is in the UQ methods, calibration,
and deferral added on top.

Also provides a small CNN for fast testing and environments where the full
ResNet-50 is unavailable.
"""
from .resnet import (
    ShipResNet50,
    ShipClassifierHead,
    SmallShipCNN,
    load_model,
    save_model,
    SHIP_CLASSES,
)
from .trainer import ShipClassifierTrainer, TrainingConfig

__all__ = [
    "ShipResNet50",
    "ShipClassifierHead",
    "SmallShipCNN",
    "load_model",
    "save_model",
    "SHIP_CLASSES",
    "ShipClassifierTrainer",
    "TrainingConfig",
]
