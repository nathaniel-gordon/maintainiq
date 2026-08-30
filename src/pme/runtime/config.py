"""Daemon configuration.

Every knob that changes pipeline *semantics* lives here and is written into each
checkpoint, so a restore refuses to continue under a different lateness policy —
or different alert thresholds, or different plant economics — than the run it is
resuming. Runtime-only knobs (how often to checkpoint, how long to allow) are
deliberately excluded: they change how a run is operated, not what it means.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..core.costs import CostModel
from ..core.schema import MIN_WINDOW_SAMPLES, WINDOW_SIZE_H
from ..errors import ConfigError


@dataclass(frozen=True, slots=True)
class StreamConfig:
    """Windowing, watermark, lateness, alerting and scheduling parameters."""

    window_size_h: int = WINDOW_SIZE_H
    window_step_h: int = 1
    tumbling_size_h: int = 168  # weekly fleet-health roll-up
    max_out_of_orderness_h: int = 6
    allowed_lateness_h: int = 2
    min_window_samples: int = MIN_WINDOW_SAMPLES
    scoring_batch_size: int = 512
    alert_cooldown_h: int = 24

    #: Risk bands. Checked highest-first by the alerting operator.
    critical_risk: float = 0.5
    warning_risk: float = 0.2

    #: Scheduling. ``anomaly_priority_bonus`` lets a novel failure signature earn a
    #: crew slot while its modelled risk is still low.
    crews_per_day: int = 3
    horizon_days: int = 7
    anomaly_priority_bonus: float = 500.0
    costs: CostModel = field(default_factory=CostModel)

    #: Fail the run when more than this share of the stream is dropped as too-late.
    #: A maintenance pipeline that silently discards a quarter of its telemetry is
    #: not a successful run, whatever its exit status used to say.
    max_drop_rate: float = 0.01

    #: Take a checkpoint every N consumed events (0 disables checkpointing).
    checkpoint_every: int = 0
    #: Emit a runtime heartbeat every N consumed events (0 disables).
    heartbeat_every: int = 5_000
    #: Stop after N consumed events, having just checkpointed (crash simulation).
    stop_after_events: int | None = None
    #: Hard wall-clock guard so the daemon can never hang a CI run.
    max_runtime_s: float = 900.0

    #: Semantic fields a restore must agree on; runtime-only fields are excluded.
    SEMANTIC_FIELDS = (
        "window_size_h",
        "window_step_h",
        "tumbling_size_h",
        "max_out_of_orderness_h",
        "allowed_lateness_h",
        "min_window_samples",
        "scoring_batch_size",
        "alert_cooldown_h",
        "critical_risk",
        "warning_risk",
        "crews_per_day",
        "horizon_days",
        "anomaly_priority_bonus",
        "costs",
    )

    def __post_init__(self) -> None:
        if self.window_size_h < 1 or self.window_step_h < 1 or self.tumbling_size_h < 1:
            raise ConfigError("window sizes must be >= 1", window_size_h=self.window_size_h,
                              window_step_h=self.window_step_h, tumbling_size_h=self.tumbling_size_h)
        if self.max_out_of_orderness_h < 0 or self.allowed_lateness_h < 0:
            raise ConfigError("watermark budgets must be >= 0",
                              max_out_of_orderness_h=self.max_out_of_orderness_h,
                              allowed_lateness_h=self.allowed_lateness_h)
        if not 1 <= self.min_window_samples <= self.window_size_h:
            raise ConfigError("min_window_samples must lie in [1, window_size_h]",
                              min_window_samples=self.min_window_samples, window_size_h=self.window_size_h)
        if self.scoring_batch_size < 1:
            raise ConfigError("scoring_batch_size must be >= 1", scoring_batch_size=self.scoring_batch_size)
        if self.alert_cooldown_h < 0:
            raise ConfigError("alert_cooldown_h must be >= 0", alert_cooldown_h=self.alert_cooldown_h)
        if not 0.0 <= self.warning_risk <= self.critical_risk <= 1.0:
            raise ConfigError("risk bands must satisfy 0 <= warning <= critical <= 1",
                              warning_risk=self.warning_risk, critical_risk=self.critical_risk)
        if self.crews_per_day < 1 or self.horizon_days < 1:
            raise ConfigError("crews_per_day and horizon_days must be >= 1",
                              crews_per_day=self.crews_per_day, horizon_days=self.horizon_days)
        if not 0.0 <= self.max_drop_rate <= 1.0:
            raise ConfigError("max_drop_rate must lie in [0, 1]", max_drop_rate=self.max_drop_rate)
        if self.max_runtime_s <= 0:
            raise ConfigError("max_runtime_s must be > 0", max_runtime_s=self.max_runtime_s)

    def semantics(self) -> dict[str, Any]:
        """The subset a checkpoint must agree on, in JSON-comparable form."""
        out: dict[str, Any] = {}
        for name in self.SEMANTIC_FIELDS:
            value = getattr(self, name)
            out[name] = value.to_dict() if isinstance(value, CostModel) else value
        return out

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class OutputPaths:
    """Where each sink writes. Derived from one directory by default."""

    root: Path
    alerts: Path = field(default=Path())
    schedule: Path = field(default=Path())
    metrics: Path = field(default=Path())
    checkpoint: Path = field(default=Path())
    late: Path = field(default=Path())
    database: Path = field(default=Path())

    @staticmethod
    def under(root: str | Path, prefix: str = "") -> OutputPaths:
        root = Path(root)
        return OutputPaths(
            root=root,
            alerts=root / f"{prefix}alerts.jsonl",
            schedule=root / f"{prefix}schedule.csv",
            metrics=root / f"{prefix}stream_metrics.json",
            checkpoint=root / f"{prefix}checkpoint.json",
            late=root / f"{prefix}late.jsonl",
            database=root / f"{prefix}maintainiq.db",
        )


__all__ = ["CostModel", "OutputPaths", "StreamConfig"]
