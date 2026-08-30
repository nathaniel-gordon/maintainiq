"""Configuration validation and the checkpoint semantic contract."""
from __future__ import annotations

from dataclasses import replace

import pytest

from pme.core.costs import CostModel
from pme.errors import ConfigError
from pme.runtime.config import OutputPaths, StreamConfig


def test_defaults_are_valid():
    cfg = StreamConfig()
    assert cfg.warning_risk < cfg.critical_risk
    assert cfg.costs.breakeven_risk > 0


@pytest.mark.parametrize(
    "overrides",
    [
        {"window_size_h": 0},
        {"max_out_of_orderness_h": -1},
        {"allowed_lateness_h": -1},
        {"min_window_samples": 0},
        {"min_window_samples": 999},
        {"scoring_batch_size": 0},
        {"alert_cooldown_h": -1},
        {"critical_risk": 0.1, "warning_risk": 0.5},
        {"critical_risk": 1.5},
        {"crews_per_day": 0},
        {"horizon_days": 0},
        {"max_drop_rate": 2.0},
        {"max_runtime_s": 0.0},
    ],
)
def test_impossible_settings_are_rejected(overrides):
    with pytest.raises(ConfigError):
        StreamConfig(**overrides)


def test_semantics_carry_thresholds_and_economics():
    semantics = StreamConfig().semantics()
    assert semantics["critical_risk"] == 0.5
    assert semantics["anomaly_priority_bonus"] == 500.0
    assert semantics["costs"]["unplanned_failure"] == 8500.0
    # Runtime-only knobs must stay out, or a resume with a different checkpoint
    # interval would be refused for no good reason.
    assert "checkpoint_every" not in semantics
    assert "max_runtime_s" not in semantics


def test_changing_economics_changes_the_semantic_contract():
    base = StreamConfig()
    dearer = replace(base, costs=CostModel(unplanned_failure=20_000.0))
    assert base.semantics() != dearer.semantics()


def test_output_paths_are_derived_from_one_directory(tmp_path):
    paths = OutputPaths.under(tmp_path, prefix="replay_")
    assert paths.alerts.name == "replay_alerts.jsonl"
    assert paths.late.name == "replay_late.jsonl"
    assert paths.database.name == "replay_maintainiq.db"
    assert paths.checkpoint.parent == tmp_path
