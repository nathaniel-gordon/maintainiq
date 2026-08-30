"""Metrics sink: fleet-health tumbling roll-ups plus daemon runtime observability.

This sink is the one the batch engine could not have had. It records, per closed
tumbling window, how each machine actually behaved, and per heartbeat, how far the
watermark trailed the newest event and how deep the reorder buffer got. Watermark
lag and late-drop rate are the two numbers that tell an operator whether the
lateness budget is set correctly for their broker.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core.events import Heartbeat, TumblingResult
from .base import Sink


class MetricsSink(Sink):
    """Persists tumbling fleet-health windows and daemon heartbeats as JSON."""

    name = "metrics"
    accepts = (TumblingResult, Heartbeat)

    def __init__(self, path: str | Path, keep_heartbeats: int = 512) -> None:
        self.path = Path(path)
        self.keep_heartbeats = keep_heartbeats
        #: Every Nth heartbeat is kept once the series is decimated. Recorded in
        #: the artifact, because a series whose sampling interval silently
        #: changes half-way through is worse than no series.
        self.windows: list[dict[str, Any]] = []
        self.heartbeats: list[dict[str, Any]] = []
        self.heartbeat_stride = 1
        self.runtime: dict[str, Any] = {}

    def write(self, record: TumblingResult | Heartbeat) -> None:
        if isinstance(record, TumblingResult):
            self.windows.append(
                {
                    "machine_id": record.key,
                    "window_start": record.window.start,
                    "window_end": record.window.end,
                    "samples": record.samples,
                    "failures": record.failures,
                    "mean_load": round(record.mean_load, 4),
                    "mean_vibration": round(record.mean_vibration, 4),
                    "max_vibration": round(record.max_vibration, 4),
                }
            )
        else:
            self.heartbeats.append(
                {
                    "events_consumed": record.events_consumed,
                    "watermark": record.watermark,
                    "max_event_time": record.max_event_time,
                    "watermark_lag": (
                        None if record.watermark is None or record.max_event_time is None
                        else record.max_event_time - record.watermark
                    ),
                    "reorder_buffer": record.reorder_buffer,
                    "dropped_late": record.dropped_late,
                    "alerts_emitted": record.alerts_emitted,
                }
            )
            if len(self.heartbeats) > self.keep_heartbeats:
                # Keep every other sample: a long run stays bounded without losing shape.
                self.heartbeats = self.heartbeats[::2]
                self.heartbeat_stride *= 2

    def set_runtime(self, stats: dict[str, Any]) -> None:
        """Attach the engine's end-of-run lateness/watermark statistics."""
        self.runtime = stats

    def close(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "runtime": self.runtime,
            "fleet_health_windows": self.windows,
            "heartbeat_stride": self.heartbeat_stride,
            "heartbeats": self.heartbeats,
        }
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def failure_windows(self) -> list[dict[str, Any]]:
        """Tumbling windows in which at least one machine actually failed."""
        return [w for w in self.windows if w["failures"] > 0]

    def summary(self) -> dict[str, Any]:
        return {
            "tumbling_windows": len(self.windows),
            "windows_with_failures": len(self.failure_windows()),
            "heartbeats": len(self.heartbeats),
            "heartbeat_stride": self.heartbeat_stride,
            "path": str(self.path),
        }

    def state(self) -> dict[str, Any]:
        return {"windows": self.windows, "heartbeats": self.heartbeats,
                "heartbeat_stride": self.heartbeat_stride}

    def restore(self, state: dict[str, Any]) -> None:
        self.windows = list(state["windows"])
        self.heartbeats = list(state["heartbeats"])
        self.heartbeat_stride = int(state["heartbeat_stride"])
