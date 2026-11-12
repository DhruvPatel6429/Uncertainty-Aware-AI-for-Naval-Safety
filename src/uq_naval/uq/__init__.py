"""Uncertainty Quantification methods.

All methods implement :class:`uq_naval.uq.base.UQInterface` and are
registered via :func:`uq_naval.uq.base.register_uq_method`.

Available methods
-----------------
- ``bayesian``       — MC Dropout + Laplace approximation
- ``bayesian_bb``    — Bayes-by-Backprop
- ``conformal``      — Inductive conformal prediction
- ``conformal_mondrian`` — Class-conditional conformal
- ``conformal_rc``   — Conformal Risk Control
- ``ensemble``       — Deep ensemble
- ``ensemble_snapshot`` — Snapshot ensemble
- ``ensemble_batch`` — Batch-ensembled network
- ``evidential_clf`` — Evidential classification (Dirichlet)
- ``evidential_reg`` — Evidential regression (NIG)
"""
from .base import (
    UQInterface,
    register_uq_method,
    list_uq_methods,
    load_uq_method,
)

# Importing the sub-packages triggers @register_uq_method on all classes.
from . import bayesian  # noqa: F401
from . import conformal  # noqa: F401
from . import ensembles   # noqa: F401
from . import evidential  # noqa: F401

__all__ = [
    "UQInterface",
    "register_uq_method",
    "list_uq_methods",
    "load_uq_method",
]
