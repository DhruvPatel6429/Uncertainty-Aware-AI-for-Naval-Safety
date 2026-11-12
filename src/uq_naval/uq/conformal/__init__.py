"""Conformal Prediction methods.

Implements the Vovk framework with three operational variants:

- :class:`InductiveConformal` — split conformal, single forward pass at inference.
- :class:`MondrianConformal` — class-conditional prediction sets.
- :class:`ConformalRiskControl` — bounds arbitrary monotone risk functionals
  (Angelopoulos et al., 2024).

References
----------
Vovk, V., Gammerman, A., Shafer, G. (2005). Algorithmic Learning in a Random World.
Angelopoulos, A.N. & Bates, S. (2021). A Gentle Introduction to Conformal Prediction.
Tibshirani, R.J. et al. (2019). Conformal Prediction Under Covariate Shift.
Angelopoulos, A.N. et al. (2024). Conformal Risk Control.
"""
from .inductive import InductiveConformal
from .mondrian import MondrianConformal
from .risk_control import ConformalRiskControl

__all__ = [
    "InductiveConformal",
    "MondrianConformal",
    "ConformalRiskControl",
]
