"""Trajectory prediction models.

Predicts the future course of detected obstacles. Two architectures are
provided; both wrap naturally in a UQ method (Deep Ensembles or Evidential DL).
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

    class TrajectoryLSTM(nn.Module):
        """LSTM-based trajectory forecaster.

        Input:  (B, T_in, feature_dim)   — historical trajectory features
        Output: (B, T_out, 2)            — (heading_delta, speed) per timestep
        """

        def __init__(
            self,
            feature_dim: int = 4,
            hidden_dim: int = 64,
            n_layers: int = 2,
            horizon: int = 30,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=feature_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                batch_first=True,
                dropout=dropout if n_layers > 1 else 0.0,
            )
            self.head = nn.Linear(hidden_dim, 2)
            self.horizon = horizon
            self.hidden_dim = hidden_dim

        def forward(self, x):
            # x: (B, T_in, F)
            h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.hidden_dim, device=x.device)
            c0 = torch.zeros_like(h0)
            out, _ = self.lstm(x, (h0, c0))
            # Take the last hidden state and project to horizon * 2
            last = out[:, -1, :]  # (B, H)
            preds = self.head(last).unsqueeze(1).repeat(1, self.horizon, 1)
            return preds

    class TrajectoryTransformer(nn.Module):
        """Transformer-based trajectory forecaster.

        A small encoder-only transformer for trajectory prediction.
        """

        def __init__(
            self,
            feature_dim: int = 4,
            d_model: int = 64,
            n_heads: int = 4,
            n_layers: int = 2,
            horizon: int = 30,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.embed = nn.Linear(feature_dim, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=128,
                dropout=dropout, batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.head = nn.Linear(d_model, 2)
            self.horizon = horizon

        def forward(self, x):
            # x: (B, T_in, F)
            x = self.embed(x)
            x = self.encoder(x)
            last = x[:, -1, :]
            preds = self.head(last).unsqueeze(1).repeat(1, self.horizon, 1)
            return preds

    class EvidentialForecaster(nn.Module):
        """Forecasting head producing NIG parameters for evidential regression.

        Output: (B, T_out, 4) — (gamma, nu, alpha, beta) per timestep.
        """

        def __init__(
            self,
            feature_dim: int = 4,
            hidden_dim: int = 64,
            horizon: int = 30,
        ):
            super().__init__()
            self.fc1 = nn.Linear(feature_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.head = nn.Linear(hidden_dim, 4 * horizon)
            self.horizon = horizon

        def forward(self, x):
            # x: (B, T_in, F) — pool over T_in
            x = x.mean(dim=1)  # (B, F)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            out = self.head(x)  # (B, 4 * horizon)
            return out.view(-1, self.horizon, 4)

else:
    TrajectoryLSTM = NumpyMLP
    TrajectoryTransformer = NumpyMLP
    EvidentialForecaster = NumpyMLP
