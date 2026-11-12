"""Perception module — object detectors.

The detector architectures wrap a backbone and produce per-detection
class probabilities + bounding boxes. The UQ method (Bayesian DL, Conformal)
is applied externally to calibrate detection scores.
"""
from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .backbones import NumpyMLP


if HAS_TORCH:

    class SimpleDetector(nn.Module):
        """Tiny single-stage detector for tests.

        Input: image tensor (B, 3, H, W)
        Output: per-anchor class logits (B, A, C) and bbox deltas (B, A, 4).
        """

        def __init__(self, n_classes: int = 5, n_anchors: int = 9, dropout: float = 0.1):
            super().__init__()
            self.n_classes = n_classes
            self.n_anchors = n_anchors
            # Tiny CNN backbone
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
            # Detection heads
            self.cls_head = nn.Conv2d(64, n_anchors * n_classes, 1)
            self.box_head = nn.Conv2d(64, n_anchors * 4, 1)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            x = F.max_pool2d(F.relu(self.conv1(x)), 2)
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
            x = F.relu(self.conv3(x))
            x = self.dropout(x)
            cls = self.cls_head(x)
            box = self.box_head(x)
            # Reshape: (B, A, C, H, W) -> (B, A, C) via global average pooling
            b = cls.size(0)
            cls = cls.view(b, self.n_anchors, self.n_classes, -1).mean(dim=-1)
            box = box.view(b, self.n_anchors, 4, -1).mean(dim=-1)
            return cls, box

    class BayesianDetector(SimpleDetector):
        """Detector with MC Dropout enabled for Bayesian inference."""

        def __init__(self, n_classes: int = 5, n_anchors: int = 9, dropout: float = 0.2):
            super().__init__(n_classes=n_classes, n_anchors=n_anchors, dropout=dropout)

        def enable_mc_dropout(self):
            """Activate dropout at inference time."""
            for m in self.modules():
                if isinstance(m, nn.Dropout):
                    m.train()

else:
    SimpleDetector = NumpyMLP
    BayesianDetector = NumpyMLP
