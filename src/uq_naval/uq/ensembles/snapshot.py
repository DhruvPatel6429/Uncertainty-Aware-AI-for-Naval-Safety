"""Snapshot Ensemble (Huang et al., 2017).

Collects multiple models from a single training run by exploiting the
cyclic learning-rate trajectory. Halves training cost vs. a standard
ensemble.
"""
from __future__ import annotations

from typing import Any, Optional

from .deep_ensemble import DeepEnsemble
from ..base import register_uq_method


@register_uq_method("ensemble_snapshot")
class SnapshotEnsemble(DeepEnsemble):
    """Snapshot ensemble — same inference interface as DeepEnsemble.

    Snapshot ensembles differ from standard ensembles only in training:
    a cyclic learning rate collects M snapshots from one training run.
    At inference, the per-snapshot weights are stacked into a list and
    passed to this class exactly as a DeepEnsemble would consume them.

    This class therefore inherits DeepEnsemble's behaviour wholesale and
    exists primarily for registry / audit-trail purposes.
    """

    def __init__(self, model: Any, config: Optional[dict] = None) -> None:
        super().__init__(model=model, config=config)
        # Override method_name set by decorator
        self.method_name = "ensemble_snapshot"
