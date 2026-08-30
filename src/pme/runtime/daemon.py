"""The daemon: a scheduler loop that consumes a telemetry stream until it drains.

    source ──► watermark ──► windowing engine ──► operators ──► sink fan-out
                                   │                  │
                                   │                  ├─ features  (incremental, per machine)
                                   │                  ├─ scoring   (micro-batched inference)
                                   │                  └─ alerting  (event-time cooldown)
                                   └─ reorder buffer, lateness policy, tumbling roll-ups

The loop is self-terminating: the source is bounded, the drain phase closes every
outstanding window, and a wall-clock guard aborts rather than hanging. Periodic
tasks — heartbeats and checkpoint barriers — are driven off the consumed-event
counter rather than wall time, so a run is reproducible regardless of how fast
the machine is.
"""
from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.assigners import SlidingWindows, TumblingWindows
from ..core.engine import WindowingEngine
from ..core.events import SENSORS, Alert, FeatureRecord, Heartbeat, ScoredRecord, SlidingTrigger, TumblingResult
from ..core.watermark import BoundedOutOfOrdernessWatermark
from ..learning.bundle import MaintenanceModels
from ..operators.alerting import AlertOperator
from ..operators.features import FeatureOperator
from ..operators.scoring import ScoringOperator
from ..sinks.alerts import AlertSink
from ..sinks.base import Sink
from ..sinks.metrics import MetricsSink
from ..sinks.schedule import CostModel, MaintenanceScheduleSink, ServiceOrder
from ..sources.base import TelemetrySource
from .checkpoint import CheckpointStore
from .config import OutputPaths, StreamConfig


@dataclass(slots=True)
class RunReport:
    """Everything a run produced, for the CLI, the report writer and the tests."""

    events_consumed: int
    features_emitted: int
    records_scored: int
    alerts: list[Alert]
    orders: list[ServiceOrder]
    engine_stats: dict[str, Any]
    sink_summaries: dict[str, dict[str, Any]]
    checkpoints_written: int
    stopped_early: bool
    wall_seconds: float
    alert_bytes: bytes = field(default=b"", repr=False)


class MaintenanceDaemon:
    """Wires the streaming topology and runs it to completion."""

    def __init__(
        self,
        source: TelemetrySource,
        models: MaintenanceModels,
        config: StreamConfig | None = None,
        paths: OutputPaths | None = None,
        costs: CostModel | None = None,
        extra_sinks: Sequence[Sink] = (),
    ) -> None:
        self.source = source
        self.models = models
        self.config = config or StreamConfig()
        self.paths = paths or OutputPaths.under(Path.cwd() / "output")
        cfg = self.config

        self.engine = WindowingEngine(
            sliding=SlidingWindows(cfg.window_size_h, cfg.window_step_h),
            tumbling=TumblingWindows(cfg.tumbling_size_h),
            watermark=BoundedOutOfOrdernessWatermark(cfg.max_out_of_orderness_h),
            allowed_lateness=cfg.allowed_lateness_h,
            vibration_index=SENSORS.index("vibration_mm_s"),
        )
        self.features = FeatureOperator(min_samples=cfg.min_window_samples)
        self.scoring = ScoringOperator(models, batch_size=cfg.scoring_batch_size)
        self.alerting = AlertOperator(cooldown_h=cfg.alert_cooldown_h)

        self.alert_sink = AlertSink(self.paths.alerts)
        self.schedule_sink = MaintenanceScheduleSink(
            self.paths.schedule, costs=costs, crews_per_day=cfg.crews_per_day,
            horizon_days=cfg.horizon_days,
        )
        self.metrics_sink = MetricsSink(self.paths.metrics)
        self.sinks: list[Sink] = [self.alert_sink, self.schedule_sink, self.metrics_sink, *extra_sinks]

        self.checkpoints = CheckpointStore(self.paths.checkpoint)
        self.events_consumed = 0
        self._restored = False

    # ------------------------------------------------------------------- state

    def snapshot(self) -> dict[str, Any]:
        """Full pipeline state. Callers must have flushed operators first."""
        return {
            "config": self.config.semantics(),
            "source": {"name": self.source.name, "cursor": self.source.cursor, "length": len(self.source)},
            "events_consumed": self.events_consumed,
            "engine": self.engine.state(),
            "operators": {
                self.features.name: self.features.state(),
                self.scoring.name: self.scoring.state(),
                self.alerting.name: self.alerting.state(),
            },
            "sinks": {sink.name: sink.state() for sink in self.sinks},
        }

    def restore(self, payload: dict[str, Any]) -> None:
        """Reinstate a checkpoint. Must be called before :meth:`run`."""
        stored = payload["config"]
        current = self.config.semantics()
        if stored != current:
            differing = {k: (stored.get(k), current.get(k)) for k in current if stored.get(k) != current.get(k)}
            raise ValueError(f"checkpoint semantics differ from current config: {differing}")
        if payload["source"]["length"] != len(self.source):
            raise ValueError("checkpoint was taken against a different-length source")
        self.source.seek(int(payload["source"]["cursor"]))
        self.events_consumed = int(payload["events_consumed"])
        self.engine.restore(payload["engine"])
        self.features.restore(payload["operators"][self.features.name])
        self.scoring.restore(payload["operators"][self.scoring.name])
        self.alerting.restore(payload["operators"][self.alerting.name])
        for sink in self.sinks:
            if sink.name in payload["sinks"]:
                sink.restore(payload["sinks"][sink.name])
        self._restored = True

    def restore_latest(self) -> bool:
        """Restore from the configured checkpoint file if one exists."""
        if not self.checkpoints.exists():
            return False
        self.restore(self.checkpoints.read())
        return True

    # --------------------------------------------------------------------- run

    def run(self) -> RunReport:
        """Consume the stream until drained (or until ``stop_after_events``)."""
        cfg = self.config
        started = time.monotonic()
        for sink in self.sinks:
            sink.open()

        stopped_early = False
        deadline = started + cfg.max_runtime_s
        for event in self.source.emit():
            self.events_consumed += 1
            for out in self.engine.push(event):
                self._route_engine_output(out)

            if cfg.heartbeat_every and self.events_consumed % cfg.heartbeat_every == 0:
                self._heartbeat()
            if cfg.checkpoint_every and self.events_consumed % cfg.checkpoint_every == 0:
                self._barrier()
            if cfg.stop_after_events is not None and self.events_consumed >= cfg.stop_after_events:
                if not (cfg.checkpoint_every and self.events_consumed % cfg.checkpoint_every == 0):
                    self._barrier()
                stopped_early = True
                break
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"daemon exceeded max_runtime_s={cfg.max_runtime_s} after {self.events_consumed} events"
                )

        if not stopped_early:
            for out in self.engine.close():
                self._route_engine_output(out)
        self._flush_operators()
        self._heartbeat()

        stats = self.engine.stats()
        self.metrics_sink.set_runtime(
            {
                **stats,
                "events_consumed": self.events_consumed,
                "features_emitted": self.features.emitted,
                "records_scored": self.scoring.scored,
                "scoring_batches": self.scoring.batches,
                "alerts_emitted": self.alerting.emitted,
                "alerts_suppressed": self.alerting.suppressed,
                "checkpoints_written": self.checkpoints.writes,
                "stopped_early": stopped_early,
            }
        )
        alert_bytes = self.alert_sink.digest()
        for sink in self.sinks:
            sink.close()

        return RunReport(
            events_consumed=self.events_consumed,
            features_emitted=self.features.emitted,
            records_scored=self.scoring.scored,
            alerts=list(self.alert_sink.alerts),
            orders=list(self.schedule_sink.orders),
            engine_stats=stats,
            sink_summaries={sink.name: sink.summary() for sink in self.sinks},
            checkpoints_written=self.checkpoints.writes,
            stopped_early=stopped_early,
            wall_seconds=round(time.monotonic() - started, 3),
            alert_bytes=alert_bytes,
        )

    # ------------------------------------------------------------------ wiring

    def _route_engine_output(self, out: SlidingTrigger | TumblingResult) -> None:
        if isinstance(out, TumblingResult):
            self.metrics_sink.write(out)
            return
        for feature in self.features.process(out):
            self._route_feature(feature)

    def _route_feature(self, feature: FeatureRecord) -> None:
        for scored in self.scoring.process(feature):
            self._route_scored(scored)

    def _route_scored(self, scored: ScoredRecord) -> None:
        self.schedule_sink.write(scored)
        for alert in self.alerting.process(scored):
            self.alert_sink.write(alert)

    def _flush_operators(self) -> None:
        """Drain buffered inference. Idempotent, so it is safe at every barrier."""
        for scored in self.scoring.flush():
            self._route_scored(scored)

    def _barrier(self) -> None:
        """Checkpoint barrier: flush every buffer, then persist the whole topology.

        Flushing first is what makes a restored run bit-identical: no operator
        carries work across the barrier, so the resumed pipeline starts from the
        same clean edge the uninterrupted run passed through.
        """
        self._flush_operators()
        self.checkpoints.write(self.snapshot())

    def _heartbeat(self) -> None:
        clock = self.engine.watermark
        # After drain the watermark is +infinity; report it as caught up rather
        # than as an absurd negative lag.
        watermark = min(clock.current, clock.max_seen)
        self.metrics_sink.write(
            Heartbeat(
                events_consumed=self.events_consumed,
                watermark=watermark,
                max_event_time=clock.max_seen,
                reorder_buffer=self.engine.reorder_buffer_size,
                dropped_late=self.engine.late_dropped,
                alerts_emitted=self.alerting.emitted,
            )
        )
