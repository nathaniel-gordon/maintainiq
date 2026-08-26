"""Daemon configuration.

Every knob that changes pipeline *semantics* lives here and is written into each
checkpoint, so a restore can refuse to continue under a different lateness policy
than the run it is resuming.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..core.schema import MIN_WINDOW_SAMPLES, WINDOW_SIZE_H


@dataclass(frozen=True, slots=True)
class StreamConfig:
    """Windowing, watermark, lateness and scheduling parameters."""

    window_size_h: int = WINDOW_SIZE_H
    window_step_h: int = 1
    tumbling_size_h: int = 168  # weekly fleet-health roll-up
    max_out_of_orderness_h: int = 6
    allowed_lateness_h: int = 2
    min_window_samples: int = MIN_WINDOW_SAMPLES
    scoring_batch_size: int = 512
    alert_cooldown_h: int = 24
    crews_per_day: int = 3
    horizon_days: int = 7

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
    )

    def semantics(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.SEMANTIC_FIELDS}

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

    @staticmethod
    def under(root: str | Path, prefix: str = "") -> "OutputPaths":
        root = Path(root)
        return OutputPaths(
            root=root,
            alerts=root / f"{prefix}alerts.jsonl",
            schedule=root / f"{prefix}schedule.csv",
            metrics=root / f"{prefix}stream_metrics.json",
            checkpoint=root / f"{prefix}checkpoint.json",
        )
