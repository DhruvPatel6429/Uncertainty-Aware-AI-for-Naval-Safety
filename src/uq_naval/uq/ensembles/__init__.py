"""Deep Ensemble methods.

- :class:`DeepEnsemble` (Lakshminarayanan et al., 2017)
- :class:`SnapshotEnsemble` (Huang et al., 2017)
- :class:`BatchEnsemble` (Wen et al., 2020)
"""
from .deep_ensemble import DeepEnsemble
from .snapshot import SnapshotEnsemble
from .batch_ensemble import BatchEnsemble

__all__ = ["DeepEnsemble", "SnapshotEnsemble", "BatchEnsemble"]
