"""Offline training by *replaying the stream*.

The design matrix is produced by running a historian backfill through the very
same windowing engine and feature operator the daemon uses at serving time. There
is no second feature implementation to drift out of sync — train/serve skew is
eliminated structurally rather than monitored.

Labels are joined afterwards from the settled history (see
:mod:`pme.learning.labels`), which is the only place the future is allowed to be
consulted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..core.assigners import SlidingWindows, TumblingWindows
from ..core.engine import WindowingEngine
from ..core.events import SlidingTrigger
from ..core.schema import FEATURE_NAMES
from ..core.watermark import BoundedOutOfOrdernessWatermark
from ..operators.features import FeatureOperator
from ..runtime.config import StreamConfig
from ..sources.base import TelemetrySource
from .bundle import MaintenanceModels
from .labels import build_labels


@dataclass(slots=True)
class FeatureReplay:
    """Design matrix recovered from a stream replay."""

    machine_ids: np.ndarray
    event_times: np.ndarray
    x: np.ndarray

    def to_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(self.x, columns=list(FEATURE_NAMES))
        frame.insert(0, "hour", self.event_times)
        frame.insert(0, "machine_id", self.machine_ids)
        return frame


def replay_features(source: TelemetrySource, config: StreamConfig | None = None,
                    min_samples: int | None = None) -> FeatureReplay:
    """Stream ``source`` through engine + feature operator and collect every window.

    ``min_samples`` defaults to the *serving* gate, not the full window. Training
    on complete windows only while the daemon emits from ``min_window_samples``
    means every partially filled window is scored by a model that never saw one —
    the exact train/serve skew this replay design exists to avoid.
    """
    cfg = config or StreamConfig(max_out_of_orderness_h=0, allowed_lateness_h=0)
    engine = WindowingEngine(
        sliding=SlidingWindows(cfg.window_size_h, cfg.window_step_h),
        tumbling=TumblingWindows(cfg.tumbling_size_h),
        watermark=BoundedOutOfOrdernessWatermark(cfg.max_out_of_orderness_h),
        allowed_lateness=cfg.allowed_lateness_h,
    )
    operator = FeatureOperator(min_samples=cfg.min_window_samples if min_samples is None else min_samples)

    keys: list[str] = []
    times: list[int] = []
    rows: list[tuple[float, ...]] = []

    def absorb(outputs: list[Any]) -> None:
        for out in outputs:
            if not isinstance(out, SlidingTrigger):
                continue
            for feature in operator.process(out):
                keys.append(feature.key)
                times.append(feature.event_time)
                rows.append(feature.values)

    for event in source.emit():
        absorb(engine.push(event))
    absorb(engine.close())

    return FeatureReplay(
        machine_ids=np.array(keys, dtype=object),
        event_times=np.array(times, dtype=np.int64),
        x=np.array(rows, dtype=float) if rows else np.empty((0, len(FEATURE_NAMES))),
    )


@dataclass(slots=True)
class TrainingResult:
    models: MaintenanceModels
    metrics: dict[str, Any]
    replay: FeatureReplay
    rul_h: np.ndarray
    fail_within_h: np.ndarray


def train_from_stream(source: TelemetrySource, seed: int = 42,
                      config: StreamConfig | None = None) -> TrainingResult:
    """Replay the historian, join labels, fit the bundle on a machine-level holdout."""
    cfg = config or StreamConfig(max_out_of_orderness_h=0, allowed_lateness_h=0)
    replay = replay_features(source, cfg)
    if len(replay.x) == 0:
        raise ValueError("stream replay produced no complete windows; is the history long enough?")

    labels = build_labels(source.history())
    index = {(str(m), int(h)): i for i, (m, h) in enumerate(zip(labels["machine_id"], labels["hour"], strict=True))}
    positions = np.array(
        [index[(str(m), int(t))] for m, t in zip(replay.machine_ids, replay.event_times, strict=True)], dtype=np.int64
    )
    rul = labels["rul_h"].to_numpy(dtype=float)[positions]
    fail = labels["fail_within_h"].to_numpy(dtype=int)[positions]

    models = MaintenanceModels(seed=seed, min_window_samples=cfg.min_window_samples)
    metrics = models.fit(replay.x, replay.machine_ids, rul, fail)
    metrics["feature_rows"] = len(replay.x)
    metrics["feature_columns"] = len(FEATURE_NAMES)
    metrics["min_window_samples"] = cfg.min_window_samples
    return TrainingResult(models=models, metrics=metrics, replay=replay, rul_h=rul, fail_within_h=fail)
