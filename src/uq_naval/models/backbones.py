"""Backbone architectures — small, self-contained, dependency-light.

These are intentionally simple architectures suitable for tests, demos, and
small-scale experiments. Production deployments would swap in larger
pre-trained backbones (YOLOv8, EfficientNet) — the UQ methods are
backbone-agnostic.
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


# ---------------------------------------------------------------------------
# Pure-numpy fallback (when PyTorch is unavailable — tests only)
# ---------------------------------------------------------------------------

class NumpyMLP:
    """Tiny MLP in pure NumPy — used as a fallback when torch is unavailable.

    Forward pass: linear → ReLU → linear → softmax.
    """

    def __init__(self, input_dim: int = 32, hidden_dim: int = 16, n_classes: int = 5, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.1, (input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.normal(0, 0.1, (hidden_dim, n_classes))
        self.b2 = np.zeros(n_classes)
        self.input_dim = input_dim

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        h = np.maximum(0, x @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        return logits


# ---------------------------------------------------------------------------
# PyTorch backbones
# ---------------------------------------------------------------------------

if HAS_TORCH:

    class SimpleMLP(nn.Module):
        """3-layer MLP with optional dropout (used for MC Dropout experiments)."""

        def __init__(
            self,
            input_dim: int = 32,
            hidden_dim: int = 64,
            n_classes: int = 5,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, n_classes)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            x = self.dropout(F.relu(self.fc1(x)))
            x = self.dropout(F.relu(self.fc2(x)))
            return self.fc3(x)

    class SmallCNN(nn.Module):
        """Small CNN for image classification — suitable for tests."""

        def __init__(self, n_classes: int = 5, dropout: float = 0.1):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.fc1 = nn.Linear(32 * 8 * 8, 64)
            self.fc2 = nn.Linear(64, n_classes)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            x = F.max_pool2d(F.relu(self.conv1(x)), 2)
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
            x = x.view(x.size(0), -1)
            x = self.dropout(F.relu(self.fc1(x)))
            return self.fc2(x)

    class BayesLinear(nn.Module):
        """Bayes-by-Backprop linear layer (Blundell et al. 2015).

        Maintains a Gaussian variational posterior over each weight,
        parameterised by mean μ and (softplus) std σ. Each forward pass
        samples weights via the reparameterisation trick.
        """

        def __init__(self, in_features: int, out_features: int, prior_std: float = 1.0):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.prior_std = prior_std
            # Variational parameters
            self.W_mu = nn.Parameter(torch.zeros(out_features, in_features))
            self.W_rho = nn.Parameter(torch.zeros(out_features, in_features))
            self.b_mu = nn.Parameter(torch.zeros(out_features))
            self.b_rho = nn.Parameter(torch.zeros(out_features))
            # Initialise
            nn.init.normal_(self.W_mu, 0, 0.1)
            nn.init.normal_(self.b_mu, 0, 0.1)

        def forward(self, x):
            W_sigma = F.softplus(self.W_rho)
            b_sigma = F.softplus(self.b_rho)
            eps_W = torch.randn_like(self.W_mu)
            eps_b = torch.randn_like(self.b_mu)
            W = self.W_mu + W_sigma * eps_W
            b = self.b_mu + b_sigma * eps_b
            return F.linear(x, W, b)

else:
    # When torch is unavailable, alias to numpy fallbacks so imports don't fail
    SimpleMLP = NumpyMLP
    SmallCNN = NumpyMLP
    BayesLinear = NumpyMLP
