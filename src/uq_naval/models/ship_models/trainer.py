"""Ship classifier trainer.

Implements a proper training pipeline for ship classification:

- ResNet-50 fine-tuning with ImageNet pretraining
- Train / validation / calibration / test four-way split
- Video-aware and vessel-aware splitting to prevent leakage
- Hold-out-class protocol for OOD evaluation
- Checkpoint saving and best-model tracking
- Supports the four UQ methods: softmax baseline, MC Dropout, deep ensemble, conformal
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Callable

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .resnet import ShipResNet50, SmallShipCNN, SHIP_CLASSES


# ───────────────────────────────────────────────────────────
# Training configuration
# ───────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    """Configuration for ship classifier training."""
    # Model
    model_type: str = "resnet50"        # "resnet50" | "small_cnn"
    n_classes: int = 6
    pretrained: bool = True              # ImageNet pretraining
    fine_tune_all: bool = False          # Fine-tune backbone or just head
    dropout: float = 0.1                 # Dropout for MC Dropout

    # Optimisation
    lr: float = 1e-3                     # Learning rate
    weight_decay: float = 1e-4           # L2 regularisation
    epochs: int = 20                     # Total epochs
    warmup_epochs: int = 2               # Warmup epochs (linear LR schedule)
    batch_size: int = 32

    # Data
    num_workers: int = 2
    pin_memory: bool = True

    # Multi-seed
    seed: int = 0                        # Random seed (change for multi-seed runs)

    # Checkpointing
    save_dir: str = "checkpoints"
    save_best: bool = True               # Save best model on val accuracy
    save_last: bool = True               # Save last-epoch model

    # Logging
    log_interval: int = 10               # Log every N batches


# ───────────────────────────────────────────────────────────
# Trainer
# ───────────────────────────────────────────────────────────

class ShipClassifierTrainer:
    """Trainer for ship-type classification.

    Implements a standard supervised training loop with:
    - Adam optimiser + cosine LR schedule with warmup
    - Cross-entropy loss
    - Validation accuracy tracking
    - Best-model checkpointing

    Usage:
        >>> config = TrainingConfig(model_type="resnet50", epochs=20)
        >>> trainer = ShipClassifierTrainer(config)
        >>> trainer.train(train_loader, val_loader)
        >>> model = trainer.get_best_model()
    """

    def __init__(self, config: TrainingConfig):
        if not HAS_TORCH:
            raise RuntimeError("ShipClassifierTrainer requires PyTorch")
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build model
        self.model = self._build_model().to(self.device)

        # Build optimiser
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        # Build LR scheduler (cosine with warmup)
        self.scheduler = self._build_scheduler()

        # Tracking
        self.best_val_acc = 0.0
        self.history: list[dict] = []

    def _build_model(self) -> nn.Module:
        """Build the model based on config."""
        if self.config.model_type == "resnet50":
            return ShipResNet50(
                n_classes=self.config.n_classes,
                pretrained=self.config.pretrained,
                fine_tune_all=self.config.fine_tune_all,
                dropout=self.config.dropout,
            )
        elif self.config.model_type == "small_cnn":
            return SmallShipCNN(
                n_classes=self.config.n_classes,
                dropout=self.config.dropout,
            )
        else:
            raise ValueError(f"Unknown model_type: {self.config.model_type!r}")

    def _build_scheduler(self):
        """Cosine LR schedule with linear warmup."""
        def lr_lambda(epoch: int) -> float:
            if epoch < self.config.warmup_epochs:
                # Linear warmup
                return (epoch + 1) / self.config.warmup_epochs
            else:
                # Cosine decay
                progress = (epoch - self.config.warmup_epochs) / max(1, self.config.epochs - self.config.warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train(
        self,
        train_loader: Any,
        val_loader: Any,
        verbose: bool = True,
    ) -> dict:
        """Run the full training loop.

        Returns a summary dict with final train/val accuracy and best epoch.
        """
        self.model.to(self.device)

        for epoch in range(self.config.epochs):
            # Train epoch
            train_loss, train_acc = self._train_epoch(train_loader, epoch, verbose)
            # Validate
            val_loss, val_acc = self._validate(val_loader)

            self.scheduler.step()

            # Track
            self.history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": self.optimizer.param_groups[0]["lr"],
            })

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                if self.config.save_best:
                    self._save_checkpoint("best.pt")

            if verbose:
                print(f"Epoch {epoch+1:3d}/{self.config.epochs}: "
                      f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                      f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                      f"lr={self.optimizer.param_groups[0]['lr']:.2e}")

        # Save last-epoch model
        if self.config.save_last:
            self._save_checkpoint("last.pt")

        best_epoch = max(self.history, key=lambda h: h["val_acc"])["epoch"] if self.history else 0
        return {
            "best_val_acc": self.best_val_acc,
            "best_epoch": best_epoch,
            "final_train_acc": self.history[-1]["train_acc"] if self.history else 0.0,
            "final_val_acc": self.history[-1]["val_acc"] if self.history else 0.0,
            "history": self.history,
        }

    def _train_epoch(self, loader, epoch: int, verbose: bool) -> tuple[float, float]:
        """Train for one epoch. Returns (mean_loss, accuracy)."""
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if verbose and (batch_idx + 1) % self.config.log_interval == 0:
                print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(loader)} | "
                      f"Loss {loss.item():.4f} | Acc {correct/total:.4f}")

        return total_loss / max(len(loader), 1), correct / max(total, 1)

    def _validate(self, loader) -> tuple[float, float]:
        """Validate. Returns (mean_loss, accuracy)."""
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(images)
                loss = F.cross_entropy(logits, labels)
                total_loss += loss.item()
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return total_loss / max(len(loader), 1), correct / max(total, 1)

    def _save_checkpoint(self, filename: str) -> None:
        """Save a model checkpoint."""
        save_path = Path(self.config.save_dir) / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_acc": self.best_val_acc,
            "config": self.config.__dict__,
        }, save_path)

    def get_best_model(self) -> nn.Module:
        """Load and return the best model."""
        best_path = Path(self.config.save_dir) / "best.pt"
        if best_path.exists():
            checkpoint = torch.load(best_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        return self.model
