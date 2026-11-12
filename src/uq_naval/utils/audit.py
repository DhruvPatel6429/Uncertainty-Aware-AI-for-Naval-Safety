"""Audit logger — persists every decision for safety-assurance evidence."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from .logging import get_logger


class AuditLogger:
    """Persists pipeline decisions as JSONL for audit-trail purposes.

    Each entry includes the full uncertainty decomposition, the conformal
    risk calculation, the reject-option threshold check, and the resulting
    action. This is the principal artefact for future safety-assurance
    evidence (STANAG-4671-style certification).
    """

    def __init__(self, log_path: str | Path | None = None):
        self.log_path = Path(log_path) if log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._buffer: list[dict[str, Any]] = []
        self._logger = get_logger("uq_naval.audit")

    def log(self, decision: dict[str, Any]) -> None:
        """Append a decision to the audit log."""
        self._buffer.append(decision)
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(decision, default=str) + "\n")
        self._logger.info(f"Decision logged: action={decision.get('action')!r}")

    def flush(self) -> list[dict[str, Any]]:
        """Return the in-memory buffer and clear it."""
        buf = self._buffer
        self._buffer = []
        return buf

    def __len__(self) -> int:
        return len(self._buffer)

    def summary(self) -> dict[str, Any]:
        """Return summary statistics of the audit log."""
        if not self._buffer:
            return {"n_decisions": 0}
        actions = [d.get("action") for d in self._buffer]
        from collections import Counter
        action_counts = Counter(actions)
        return {
            "n_decisions": len(self._buffer),
            "action_counts": dict(action_counts),
            "mean_collision_bound": float(
                sum(d.get("collision_prob_bound", 0) for d in self._buffer) / len(self._buffer)
            ),
        }
