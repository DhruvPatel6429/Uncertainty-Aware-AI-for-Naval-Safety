"""Unified UQ method interface.

Every UQ method in the framework — Bayesian DL, Conformal Prediction, Deep
Ensembles, Evidential DL — implements this interface. Pipeline stages select
methods at runtime via a YAML configuration change.

Design Principles
-----------------
1. **Plug-in architecture**: a method is registered via ``register_uq_method``
   and instantiated by name from config.
2. **Stateless prediction**: ``predict`` does not mutate the method; the
   method's state is captured entirely by ``model`` + ``calibration_state``.
3. **Audit-first**: every prediction returns a ``UQOutput`` whose ``audit``
   field carries method-specific details for the audit trail.
4. **No engine lock-in**: methods work with any nn.Module-compatible model;
   no assumption of specific training loops.
"""
from __future__ import annotations

import abc
from typing import Any, Callable, Optional

import numpy as np

from ..types import CalibrationResult, UQOutput


class UQInterface(abc.ABC):
    """Abstract interface every UQ method must implement.

    Subclasses are constructed with a model (any callable) and a config dict,
    and expose two methods:

    - ``calibrate(calibration_data)`` → ``CalibrationResult``
    - ``predict(input)`` → ``UQOutput``
    """

    method_name: str = "abstract"

    def __init__(self, model: Any, config: Optional[dict] = None) -> None:
        self.model = model
        self.config = config or {}
        self._calibrated = False
        self._calibration_state: dict[str, Any] = {}

    # ---- API -------------------------------------------------------------

    @abc.abstractmethod
    def calibrate(self, calibration_data: Any) -> CalibrationResult:
        """Use a held-out calibration set to fit any post-hoc parameters
        (conformal quantile, temperature, etc.)."""
        ...

    @abc.abstractmethod
    def predict(self, x: Any) -> UQOutput:
        """Produce a calibrated, uncertainty-decomposed prediction for ``x``."""
        ...

    # ---- Convenience -----------------------------------------------------

    def predict_batch(self, xs: list[Any]) -> list[UQOutput]:
        """Default batched prediction — overridable for efficiency."""
        return [self.predict(x) for x in xs]

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(method={self.method_name!r}, calibrated={self._calibrated})"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_UQ_REGISTRY: dict[str, type[UQInterface]] = {}


def register_uq_method(name: str) -> Callable[[type[UQInterface]], type[UQInterface]]:
    """Class decorator registering a UQ method under ``name``."""
    def decorator(cls: type[UQInterface]) -> type[UQInterface]:
        cls.method_name = name
        _UQ_REGISTRY[name] = cls
        return cls
    return decorator


def list_uq_methods() -> list[str]:
    """List registered UQ method names."""
    return sorted(_UQ_REGISTRY.keys())


def load_uq_method(
    name: str,
    model: Any,
    config: Optional[dict] = None,
) -> UQInterface:
    """Instantiate a registered UQ method by name."""
    if name not in _UQ_REGISTRY:
        raise KeyError(
            f"Unknown UQ method {name!r}. Available: {list_uq_methods()}"
        )
    return _UQ_REGISTRY[name](model=model, config=config)


__all__ = [
    "UQInterface",
    "register_uq_method",
    "list_uq_methods",
    "load_uq_method",
]
