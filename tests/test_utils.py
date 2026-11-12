"""Tests for utility helpers."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from uq_naval.utils.seeds import set_global_seed
from uq_naval.utils.config import load_config, save_config, merge_configs
from uq_naval.utils.audit import AuditLogger


class TestSeeds:

    def test_set_global_seed(self):
        # Should not raise
        set_global_seed(42)

    def test_reproducibility(self):
        import random
        set_global_seed(123)
        a1 = [random.random() for _ in range(5)]
        set_global_seed(123)
        a2 = [random.random() for _ in range(5)]
        assert a1 == a2


class TestConfig:

    def test_save_and_load(self, tmp_path):
        cfg = {"uq_method": "bayesian", "n_mc_samples": 10, "nested": {"key": "value"}}
        path = tmp_path / "test_config.yaml"
        save_config(cfg, path)
        loaded = load_config(path)
        assert loaded == cfg

    def test_merge(self):
        a = {"x": 1, "y": {"z": 2}}
        b = {"y": {"w": 3}, "k": 4}
        merged = merge_configs(a, b)
        assert merged == {"x": 1, "y": {"z": 2, "w": 3}, "k": 4}

    def test_load_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/to/config.yaml")


class TestAuditLogger:

    def test_log_to_buffer(self):
        logger = AuditLogger()
        logger.log({"action": "EXECUTE", "collision_prob_bound": 1e-5})
        assert len(logger) == 1

    def test_log_to_file(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        logger = AuditLogger(log_path=path)
        logger.log({"action": "EXECUTE", "collision_prob_bound": 1e-5})
        logger.log({"action": "REJECT_HUMAN", "collision_prob_bound": 0.0})
        # Each line is JSON
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["action"] == "EXECUTE"

    def test_summary(self):
        logger = AuditLogger()
        logger.log({"action": "EXECUTE", "collision_prob_bound": 0.0})
        logger.log({"action": "REJECT_HUMAN", "collision_prob_bound": 0.0})
        logger.log({"action": "EXECUTE", "collision_prob_bound": 0.0})
        summary = logger.summary()
        assert summary["n_decisions"] == 3
        assert summary["action_counts"]["EXECUTE"] == 2
        assert summary["action_counts"]["REJECT_HUMAN"] == 1

    def test_flush(self):
        logger = AuditLogger()
        logger.log({"action": "EXECUTE"})
        buf = logger.flush()
        assert len(buf) == 1
        assert len(logger) == 0
