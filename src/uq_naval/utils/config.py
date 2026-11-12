"""Hydra / OmegaConf configuration loader."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file. Returns a plain dict (no OmegaConf dependency required)."""
    import yaml
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(config: dict[str, Any], path: str | Path) -> None:
    """Save a config dict as YAML."""
    import yaml
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)


def merge_configs(*configs: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge multiple config dicts. Later configs override earlier."""
    out: dict[str, Any] = {}
    for cfg in configs:
        for k, v in cfg.items():
            if k in out and isinstance(out[k], dict) and isinstance(v, dict):
                out[k] = merge_configs(out[k], v)
            else:
                out[k] = v
    return out
