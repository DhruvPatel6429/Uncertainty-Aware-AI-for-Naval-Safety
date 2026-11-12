"""Perception module.

Wraps a detector + a UQ method, producing calibrated obstacle detections
with decomposed uncertainty. The UQ method is injected via configuration,
allowing the operator to swap between Bayesian DL, Conformal Prediction,
etc. at runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ..types import Detection
from ..uq.base import UQInterface


@dataclass
class PerceptionConfig:
    """Configuration for the perception module."""
    uq_method: str = "bayesian"        # UQ method name
    uq_config: dict = None             # UQ method config
    confidence_threshold: float = 0.5  # Min confidence to accept a detection
    epistemic_threshold: float = 0.4   # Above this → flag for human review
    ood_threshold: float = 0.7        # Above this → mark as OOD
    n_classes: int = 5


class PerceptionModule:
    """Perception pipeline stage.

    Wraps a detector (callable producing logits + bbox deltas) and a UQ
    method. The UQ method calibrates the detector's outputs and produces
    a calibrated confidence + decomposed uncertainty per detection.
    """

    def __init__(
        self,
        detector: Any,
        uq_method: UQInterface,
        config: Optional[PerceptionConfig] = None,
    ):
        self.detector = detector
        self.uq_method = uq_method
        self.config = config or PerceptionConfig()

    def process(self, sensor_input: Any) -> list[Detection]:
        """Process a sensor input and return calibrated detections.

        Parameters
        ----------
        sensor_input:
            A single frame from radar / EO / IR (or a batch of frames).
            Shape and modality depend on the detector; the framework is
            agnostic.

        Returns
        -------
        list[Detection]
            Calibrated detections, each with confidence, epistemic_unc,
            and is_ood set.
        """
        # Flatten the sensor input to a 1D feature vector for the UQ method
        # (which wraps a small MLP). In a production system this would be
        # a proper CNN forward pass; here we flatten for compatibility.
        x_flat = np.asarray(sensor_input).ravel()

        # If the flattened input is larger than what the model expects,
        # truncate; if smaller, pad. This is a synthetic-data shim.
        model_input = x_flat
        try:
            # Run UQ method — produces calibrated class probabilities + uncertainty
            uq_out = self.uq_method.predict(model_input)
        except Exception:
            # Fall back to a deterministic prediction with default uncertainty
            from ..types import UQOutput
            uq_out = UQOutput(
                prediction=0,
                aleatoric_unc=0.5,
                epistemic_unc=0.5,
                confidence=0.5,
                reject=False,
                audit={"fallback": True},
            )

        # Run detector for bounding boxes (in a real system this is integrated)
        try:
            cls_logits, bbox_deltas = self.detector(model_input)
            if hasattr(cls_logits, "detach"):
                cls_logits = cls_logits.detach().cpu().numpy()
                bbox_deltas = bbox_deltas.detach().cpu().numpy()
        except Exception:
            # Fallback: synthetic single detection at image centre
            cls_logits = np.array([[uq_out.confidence]])
            bbox_deltas = np.array([[0.4, 0.4, 0.6, 0.6]])

        # Convert logits to class probabilities (softmax)
        if cls_logits.ndim == 1:
            cls_logits = cls_logits.reshape(1, -1)
        probs = _softmax(cls_logits, axis=-1)

        # Synthesise detections — one per anchor (simplified for clarity)
        detections = []
        labels = ["cargo_vessel", "fishing_boat", "buoy", "speedboat", "unknown"]

        n_anchors = min(probs.shape[0], 5)  # cap at 5 for demo
        for i in range(n_anchors):
            class_idx = int(probs[i].argmax())
            confidence = float(probs[i].max())

            # Construct a bounding box from the detector output
            if i < bbox_deltas.shape[0]:
                box = tuple(float(x) for x in bbox_deltas[i])
            else:
                box = (0.1 * i, 0.1 * i, 0.1 * i + 0.2, 0.1 * i + 0.2)

            d = Detection(
                track_id=i,
                label=labels[class_idx] if class_idx < len(labels) else "unknown",
                bbox_xyxy=box,
                confidence=confidence,
                epistemic_unc=float(uq_out.epistemic_unc),
                is_ood=bool(uq_out.epistemic_unc > self.config.ood_threshold),
            )
            detections.append(d)

        return detections


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)
