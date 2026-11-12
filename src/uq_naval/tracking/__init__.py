"""MLflow tracking helpers — wraps experiment + run management."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Optional


@contextmanager
def start_run(experiment_name: str = "uq-naval", run_name: Optional[str] = None):
    """Context manager wrapping mlflow.start_run.

    Falls back to a no-op if MLflow is unavailable or unconfigured.
    """
    try:
        import mlflow
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name) as run:
            yield run
    except ImportError:
        # MLflow not installed — yield a no-op context
        yield None


def log_metrics(metrics: dict[str, float], step: Optional[int] = None) -> None:
    """Log metrics to MLflow (no-op if unavailable)."""
    try:
        import mlflow
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)
    except ImportError:
        pass


def log_params(params: dict[str, Any]) -> None:
    """Log params to MLflow (no-op if unavailable)."""
    try:
        import mlflow
        mlflow.log_params({k: str(v) for k, v in params.items()})
    except ImportError:
        pass


def log_artifact(path: str) -> None:
    """Log a file artifact to MLflow."""
    try:
        import mlflow
        mlflow.log_artifact(path)
    except ImportError:
        pass
