"""ImageNet-C style corruption suite for ship classification.

Implements 5 corruption types at 5 severity levels, following the ImageNet-C
methodology (Hendrycks & Dietterich, 2019). Corruptions are applied at test
time to evaluate model robustness under degraded conditions.

The 5 corruption types are chosen to be operationally relevant to maritime
environments:
- **fog** — visibility reduction (maritime fog)
- **rain** — precipitation noise
- **low_light** — night / dusk conditions
- **glare** — sun glint on the camera sensor
- **noise** — generic sensor noise (Gaussian)

Each corruption is parameterised by a severity level 1-5, where 1 is the
mildest and 5 is the most severe.

References
----------
Hendrycks, D. & Dietterich, T. (2019). Benchmarking Neural Network Robustness
to Common Corruptions and Perturbations. ICLR.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class CorruptionType(str, Enum):
    """The 5 corruption types relevant to maritime conditions."""
    FOG = "fog"
    RAIN = "rain"
    LOW_LIGHT = "low_light"
    GLARE = "glare"
    NOISE = "noise"


SEVERITY_LEVELS = [1, 2, 3, 4, 5]


# ───────────────────────────────────────────────────────────
# Individual corruption functions
# ───────────────────────────────────────────────────────────

def fog_corruption(image: np.ndarray, severity: int) -> np.ndarray:
    """Add maritime fog — reduces contrast and adds a white veil.

    Severity 1-5 controls fog density (alpha blending factor).
    """
    if image.dtype == np.float32 or image.dtype == np.float64:
        img = np.clip(image, 0.0, 1.0).astype(np.float64)
    else:
        img = image.astype(np.float64) / 255.0

    # Severity → fog density (0.1 to 0.6)
    density = 0.1 + (severity - 1) * 0.125

    # Blend with white veil
    fog_veil = np.ones_like(img) * 0.85
    corrupted = (1 - density) * img + density * fog_veil

    if image.dtype == np.uint8:
        return np.clip(corrupted * 255, 0, 255).astype(np.uint8)
    return corrupted.astype(np.float32)


def rain_corruption(image: np.ndarray, severity: int) -> np.ndarray:
    """Add rain — vertical streaks + brightness reduction.

    Severity controls rain density and brightness reduction.
    """
    if image.dtype == np.float32 or image.dtype == np.float64:
        img = np.clip(image, 0.0, 1.0).astype(np.float64)
    else:
        img = image.astype(np.float64) / 255.0

    rng = np.random.default_rng(seed=42 + severity)
    h, w = img.shape[:2]

    # Number of raindrops scales with severity
    n_drops = severity * 50
    corrupted = img.copy()

    # Darken image slightly (rain reduces brightness)
    brightness_reduction = 0.05 * severity
    corrupted = corrupted * (1 - brightness_reduction)

    # Add rain streaks (vertical lines of varying length)
    for _ in range(n_drops):
        x = rng.integers(0, w)
        y_start = rng.integers(0, h - severity * 2)
        length = severity * 2 + rng.integers(-2, 3)
        y_end = min(h, y_start + length)
        # Raindrop is brighter than background
        for c in range(img.shape[2]) if img.ndim == 3 else [0]:
            corrupted[y_start:y_end, x, c] = np.minimum(
                corrupted[y_start:y_end, x, c] + 0.2,
                1.0,
            )

    if image.dtype == np.uint8:
        return np.clip(corrupted * 255, 0, 255).astype(np.uint8)
    return corrupted.astype(np.float32)


def low_light_corruption(image: np.ndarray, severity: int) -> np.ndarray:
    """Simulate low-light / night conditions — darkens and adds Poisson noise.

    Severity controls darkness level and noise intensity.
    """
    if image.dtype == np.float32 or image.dtype == np.float64:
        img = np.clip(image, 0.0, 1.0).astype(np.float64)
    else:
        img = image.astype(np.float64) / 255.0

    # Darken image — severity 1 keeps 70% brightness, severity 5 keeps 15%
    brightness_factor = max(0.15, 1.0 - severity * 0.17)
    darkened = img * brightness_factor

    # Add Poisson noise (simulates photon shot noise in low light)
    noise_scale = severity * 0.01
    noisy = darkened + np.random.poisson(darkened * 100) / 100 * noise_scale
    noisy = np.clip(noisy, 0.0, 1.0)

    if image.dtype == np.uint8:
        return np.clip(noisy * 255, 0, 255).astype(np.uint8)
    return noisy.astype(np.float32)


def glare_corruption(image: np.ndarray, severity: int) -> np.ndarray:
    """Add sun glare — bright spot with halo, simulating sun glint on the sensor.

    Severity controls spot brightness and size.
    """
    if image.dtype == np.float32 or image.dtype == np.float64:
        img = np.clip(image, 0.0, 1.0).astype(np.float64)
    else:
        img = image.astype(np.float64) / 255.0

    rng = np.random.default_rng(seed=100 + severity)
    h, w = img.shape[:2]

    # Glare spot location — randomised but reproducible
    cx = rng.integers(w // 4, 3 * w // 4)
    cy = rng.integers(h // 4, 3 * h // 4)

    # Spot radius scales with severity
    radius = severity * 5 + 10

    # Spot brightness scales with severity
    brightness = min(1.0, 0.5 + severity * 0.1)

    # Create glare mask
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    # Gaussian-like spot
    mask = np.exp(-(dist ** 2) / (2 * (radius ** 2)))
    if img.ndim == 3:
        mask = mask[:, :, np.newaxis]

    corrupted = img * (1 - mask * brightness) + mask * brightness
    corrupted = np.clip(corrupted, 0.0, 1.0)

    if image.dtype == np.uint8:
        return np.clip(corrupted * 255, 0, 255).astype(np.uint8)
    return corrupted.astype(np.float32)


def noise_corruption(image: np.ndarray, severity: int) -> np.ndarray:
    """Add Gaussian sensor noise.

    Severity controls noise standard deviation.
    """
    if image.dtype == np.float32 or image.dtype == np.float64:
        img = np.clip(image, 0.0, 1.0).astype(np.float64)
    else:
        img = image.astype(np.float64) / 255.0

    # Noise std scales with severity (0.02 to 0.20)
    noise_std = 0.02 + (severity - 1) * 0.045

    rng = np.random.default_rng(seed=200 + severity)
    noise = rng.normal(0, noise_std, img.shape)
    corrupted = np.clip(img + noise, 0.0, 1.0)

    if image.dtype == np.uint8:
        return np.clip(corrupted * 255, 0, 255).astype(np.uint8)
    return corrupted.astype(np.float32)


# ───────────────────────────────────────────────────────────
# Corruption registry
# ───────────────────────────────────────────────────────────

CORRUPTION_FUNCTIONS: dict[CorruptionType, Callable[[np.ndarray, int], np.ndarray]] = {
    CorruptionType.FOG: fog_corruption,
    CorruptionType.RAIN: rain_corruption,
    CorruptionType.LOW_LIGHT: low_light_corruption,
    CorruptionType.GLARE: glare_corruption,
    CorruptionType.NOISE: noise_corruption,
}


# ───────────────────────────────────────────────────────────
# Corruption suite
# ───────────────────────────────────────────────────────────

@dataclass
class CorruptionResult:
    """A corrupted image with metadata."""
    image: np.ndarray
    corruption_type: CorruptionType
    severity: int


class CorruptionSuite:
    """Apply ImageNet-C style corruptions to ship images.

    Usage:
        >>> suite = CorruptionSuite()
        >>> # Single corruption
        >>> corrupted = suite.apply(image, CorruptionType.FOG, severity=3)
        >>> # All corruptions at all severities
        >>> results = suite.apply_all(image)
    """

    def __init__(self, seed: int = 0):
        self.seed = seed

    @property
    def corruption_types(self) -> list[CorruptionType]:
        return list(CORRUPTION_FUNCTIONS.keys())

    @property
    def severity_levels(self) -> list[int]:
        return SEVERITY_LEVELS

    def apply(
        self,
        image: np.ndarray,
        corruption_type: CorruptionType,
        severity: int,
    ) -> np.ndarray:
        """Apply a single corruption at a given severity."""
        if corruption_type not in CORRUPTION_FUNCTIONS:
            raise ValueError(f"Unknown corruption: {corruption_type!r}")
        if severity not in SEVERITY_LEVELS:
            raise ValueError(f"Severity must be in {SEVERITY_LEVELS}, got {severity}")
        return CORRUPTION_FUNCTIONS[corruption_type](image, severity)

    def apply_all(
        self,
        image: np.ndarray,
    ) -> list[CorruptionResult]:
        """Apply all 5 corruptions × 5 severities = 25 corrupted variants."""
        results = []
        for ctype in self.corruption_types:
            for severity in self.severity_levels:
                corrupted = self.apply(image, ctype, severity)
                results.append(CorruptionResult(
                    image=corrupted,
                    corruption_type=ctype,
                    severity=severity,
                ))
        return results

    def corruption_grid(self) -> dict[str, list[int]]:
        """Return the corruption × severity grid as a dict."""
        return {
            ctype.value: list(SEVERITY_LEVELS)
            for ctype in self.corruption_types
        }

    def summary(self) -> dict:
        """Return a summary of the corruption suite."""
        return {
            "n_corruptions": len(self.corruption_types),
            "n_severities": len(self.severity_levels),
            "total_variants": len(self.corruption_types) * len(self.severity_levels),
            "corruption_types": [c.value for c in self.corruption_types],
            "severity_levels": list(self.severity_levels),
        }
