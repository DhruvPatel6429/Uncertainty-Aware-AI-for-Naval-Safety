"""ResNet-50 ship classifier with ImageNet pretraining.

This is the primary model for ship-type classification. ResNet-50 pretrained
on ImageNet is the standard, proven backbone — the novelty of the project is
in what we add on top (UQ methods, calibration, deferral), not in the backbone.

Two model classes are provided:

- :class:`ShipResNet50` — full ResNet-50 with a fine-tunable classification head.
  Requires torchvision. Suitable for real experiments.
- :class:`SmallShipCNN` — a small 3-conv CNN for fast testing and environments
  where the full ResNet-50 is unavailable.

Both implement the same interface: ``forward(x) -> logits``, so the UQ methods
can wrap either one transparently.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import models, transforms
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ───────────────────────────────────────────────────────────
# Ship classes
# ───────────────────────────────────────────────────────────

SHIP_CLASSES: list[str] = [
    "cargo",
    "tanker",
    "passenger",
    "fishing",
    "naval",
    "small_craft",
]


# ───────────────────────────────────────────────────────────
# ImageNet normalisation (used for pretrained ResNet)
# ───────────────────────────────────────────────────────────

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ───────────────────────────────────────────────────────────
# ResNet-50 ship classifier
# ───────────────────────────────────────────────────────────

if HAS_TORCH:

    class ShipClassifierHead(nn.Module):
        """Fine-tunable classification head for ship classification.

        Replaces the final FC layer of ResNet-50 with:
        - Dropout (configurable, used for MC Dropout at inference)
        - Linear → n_classes
        """

        def __init__(self, in_features: int = 2048, n_classes: int = 6, dropout: float = 0.1):
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(in_features, n_classes)

        def forward(self, x):
            x = self.dropout(x)
            return self.fc(x)


    class ShipResNet50(nn.Module):
        """ResNet-50 pretrained on ImageNet, fine-tuned for ship classification.

        The backbone is ResNet-50 with the final fully-connected layer replaced
        by a fine-tunable head. By default, all backbone layers are frozen
        except the head — set ``fine_tune_all=True`` to fine-tune the entire
        network (recommended for the last few epochs).

        Parameters
        ----------
        n_classes : int
            Number of ship classes.
        pretrained : bool
            If True, load ImageNet-pretrained weights.
        fine_tune_all : bool
            If True, fine-tune all backbone layers. If False (default), only
            the head is trainable.
        dropout : float
            Dropout rate in the classification head. Used for MC Dropout
            at inference.
        """

        def __init__(
            self,
            n_classes: int = 6,
            pretrained: bool = True,
            fine_tune_all: bool = False,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.n_classes = n_classes

            # Load ResNet-50 with or without ImageNet pretraining
            try:
                # Newer torchvision API (>=0.13)
                weights = "IMAGENET1K_V1" if pretrained else None
                backbone = models.resnet50(weights=weights)
            except (TypeError, AttributeError):
                # Older torchvision API
                backbone = models.resnet50(pretrained=pretrained)

            # Remove the final FC layer — we replace it with our head
            self.features = nn.Sequential(*list(backbone.children())[:-1])
            self.head = ShipClassifierHead(
                in_features=2048,
                n_classes=n_classes,
                dropout=dropout,
            )

            # Freeze backbone unless fine_tune_all is set
            if not fine_tune_all:
                for param in self.features.parameters():
                    param.requires_grad = False

        def forward(self, x):
            # x: (B, 3, 224, 224)
            features = self.features(x)        # (B, 2048, 1, 1)
            features = features.flatten(1)     # (B, 2048)
            return self.head(features)         # (B, n_classes)

        def enable_mc_dropout(self) -> None:
            """Enable dropout layers for MC Dropout inference."""
            for m in self.modules():
                if isinstance(m, nn.Dropout):
                    m.train()

        def disable_mc_dropout(self) -> None:
            """Disable dropout layers for deterministic inference."""
            for m in self.modules():
                if isinstance(m, nn.Dropout):
                    m.eval()

        def get_transform(self):
            """Return the standard ImageNet transform for this model."""
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])


    class SmallShipCNN(nn.Module):
        """Small 3-conv CNN for fast testing.

        Used in test environments where loading ResNet-50 is impractical.
        Implements the same ``forward(x) -> logits`` interface as ShipResNet50.
        """

        def __init__(self, n_classes: int = 6, dropout: float = 0.1):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(64 * 8 * 8, 128)
            self.fc2 = nn.Linear(128, n_classes)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            # Accept various input sizes — resize to 64x64
            if x.shape[-1] != 64 or x.shape[-2] != 64:
                x = F.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False)
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = x.view(x.size(0), -1)
            x = self.dropout(F.relu(self.fc1(x)))
            return self.fc2(x)

        def enable_mc_dropout(self) -> None:
            for m in self.modules():
                if isinstance(m, nn.Dropout):
                    m.train()

        def disable_mc_dropout(self) -> None:
            for m in self.modules():
                if isinstance(m, nn.Dropout):
                    m.eval()


else:
    # Fallback when torch is unavailable — numpy placeholder
    class ShipResNet50:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError("ShipResNet50 requires PyTorch + torchvision")
    class ShipClassifierHead:  # type: ignore[no-redef]
        pass
    class SmallShipCNN:  # type: ignore[no-redef]
        pass


# ───────────────────────────────────────────────────────────
# Model persistence
# ───────────────────────────────────────────────────────────

def save_model(model: "ShipResNet50 | SmallShipCNN", path: str | Path) -> None:
    """Save model weights to disk."""
    if not HAS_TORCH:
        raise RuntimeError("save_model requires PyTorch")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(
    path: str | Path,
    model_class: type = None,
    **kwargs,
) -> "ShipResNet50 | SmallShipCNN":
    """Load model weights from disk.

    Parameters
    ----------
    path : str | Path
        Path to the saved model weights.
    model_class : type
        Model class to instantiate (ShipResNet50 or SmallShipCNN). If None,
        auto-detects from the weights file.
    """
    if not HAS_TORCH:
        raise RuntimeError("load_model requires PyTorch")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model weights not found: {path}")

    if model_class is None:
        # Auto-detect based on parameter count
        state_dict = torch.load(path, map_location="cpu")
        if any("features.0.weight" in k for k in state_dict):
            model_class = ShipResNet50
        else:
            model_class = SmallShipCNN

    model = model_class(**kwargs)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model
