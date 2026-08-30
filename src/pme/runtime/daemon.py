"""The daemon: a scheduler loop that consumes a telemetry stream until it drains.

    source ──► watermark ──► windowing engine ──► operators ──► sink fan-out
                                   │                  │
                                   │                  ├─ features  (incremental, per machine)
                                   │                  ├─ scoring   (micro-batched inference)
                                   │                  └─ alerting  (event-time cooldown)
                                   └─ reorder buffer, lateness policy, tumbling roll-ups,
                                      late side output

Periodic tasks — heartbeats and checkpoint barriers — are driven off the
consumed-event counter rather than wall time, so a run is reproducible regardless
of how fast the machine is. The loop terminates on its own: the source is
bounded, the drain phase closes every outstanding window, a wall-clock guard
aborts rather than hanging, and SIGINT/SIGTERM take a checkpoint and stop.

Sink dispatch is table-driven off each sink's ``accepts``, so an extra sink is
wired in by declaring the records it wants.
"""
from __future__ import annotations

import signal
import time
import types
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.assigners import SlidingWindows, TumblingWindows
from ..core.engine import WindowingEngine
from ..core.events import (
    SENSORS,
    Alert,
    FeatureRecord,
    Heartbeat,
    LateEvent,
    ScoredRecord,
    SlidingTrigger,
    TumblingResult,
)
from ..core.watermark import BoundedOutOfOrdernessWatermark
from ..errors import (
    CheckpointError,
    DropRateExceededError,
    RuntimeGuardError,
    StreamMismatchError,
    TrainServeSkewError,
)
from ..learning.bundle import MaintenanceModels
from ..logging_setup import correlation_id, get_logger
from ..operators.alerting import AlertOperator
from ..operators.features import FeatureOperator
from ..operators.scoring import ScoringOperator
from ..sinks.alerts import AlertSink
from ..sinks.base import Sink
from ..sinks.database import DatabaseSink
from ..sinks.late import LateEventSink
from ..sinks.metrics import MetricsSink
from ..sinks.schedule import MaintenanceScheduleSink, ServiceOrder
from ..sources.base import TelemetrySource
from ..store.repository import MetadataStore
from .checkpoint import CheckpointStore
from .config import CostModel, OutputPaths, StreamConfig

log = get_logger(__name__)


@dataclass(slots=True)
class RunReport:
    """Everything a run produced, for the CLI and the report writer."""

    run_id: str
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

    @property
    def drop_rate(self) -> float:
        return float(self.engine_stats["drop_rate"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "events_consumed": self.events_consumed,
            "features_emitted": self.features_emitted,
            "records_scored": self.records_scored,
            "alerts": len(self.alerts),
            "service_orders": len(self.orders),
            "stopped_early": self.stopped_early,
            "checkpoints_written": self.checkpoints_written,
            "wall_seconds": self.wall_seconds,
            "engine": self.engine_stats,
            "sinks": self.sink_summaries,
        }


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
        store: MetadataStore | None = None,
        run_id: str | None = None,
    ) -> None:
        self.source = source
        self.models = models
        self.config = config or StreamConfig()
        self.paths = paths or OutputPaths.under(Path.cwd() / "output")
        self.store = store
        self.run_id = run_id or f"run-{uuid.uuid4().hex[:12]}"
        cfg = self.config

        self.engine = WindowingEngine(
            sliding=SlidingWindows(cfg.window_size_h, cfg.window_step_h),
            tumbling=TumblingWindows(cfg.tumbling_size_h),
            watermark=BoundedOutOfOrdernessWatermark(cfg.max_out_of_orderness_h),
            allowed_lateness=cfg.allowed_lateness_h,
            vibration_index=SENSORS.index("vibration_mm_s"),
        )
        # The feature maths is shared between training and serving by construction,
        # but the emission gate is not: a bundle trained on full windows would be
        # asked to score partial ones. Compare and refuse.
        trained_gate = getattr(models, "min_window_samples", cfg.min_window_samples)
        if trained_gate != cfg.min_window_samples:
            raise TrainServeSkewError(
                "the model bundle was trained under a different window gate",
                trained_min_window_samples=trained_gate,
                serving_min_window_samples=cfg.min_window_samples,
            )
        self.features = FeatureOperator(min_samples=cfg.min_window_samples)
        self.scoring = ScoringOperator(models, batch_size=cfg.scoring_batch_size)
        self.alerting = AlertOperator(
            cooldown_h=cfg.alert_cooldown_h,
            critical_risk=cfg.critical_risk,
            warning_risk=cfg.warning_risk,
        )

        self.alert_sink = AlertSink(self.paths.alerts)
        self.schedule_sink = MaintenanceScheduleSink(
            self.paths.schedule, costs=costs or cfg.costs, crews_per_day=cfg.crews_per_day,
            horizon_days=cfg.horizon_days, anomaly_bonus=cfg.anomaly_priority_bonus,
        )
        self.metrics_sink = MetricsSink(self.paths.metrics)
        self.late_sink = LateEventSink(self.paths.late)
        self.sinks: list[Sink] = [self.alert_sink, self.schedule_sink, self.metrics_sink, self.late_sink]
        if store is not None:
            self.sinks.append(DatabaseSink(store, self.run_id))
        self.sinks.extend(extra_sinks)
        self._routes = self._build_routes()

        self.checkpoints = CheckpointStore(self.paths.checkpoint)
        self.events_consumed = 0
        self._restored = False
        self._stop_requested = ""

    def _build_routes(self) -> dict[type, list[Sink]]:
        """Dispatch table: record class -> every sink that declared it."""
        routes: dict[type, list[Sink]] = {}
        for sink in self.sinks:
            for record_type in sink.accepts:
                routes.setdefault(record_type, []).append(sink)
        return routes

    # ------------------------------------------------------------------- state

    def snapshot(self) -> dict[str, Any]:
        """Full pipeline state. Callers must have flushed operators first."""
        return {
            "run_id": self.run_id,
            "config": self.config.semantics(),
            "source": {
                "name": self.source.name,
                "cursor": self.source.cursor,
                "length": len(self.source),
                "fingerprint": self.source.fingerprint(),
            },
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
        """Reinstate a checkpoint, or refuse it. Must be called before :meth:`run`.

        A checkpoint is only meaningful against the exact stream it was taken
        from. Comparing the config and the source length was not enough: the same
        file re-delivered in a different order passed both checks and then
        discarded most of the run. The fingerprint closes that.
        """
        self._require_keys(payload, ("config", "source", "events_consumed", "engine", "operators", "sinks"))
        self._require_keys(payload["source"], ("name", "cursor", "length", "fingerprint"))

        stored, current = payload["config"], self.config.semantics()
        if stored != current:
            differing = {k: {"checkpoint": stored.get(k), "current": current.get(k)}
                         for k in set(stored) | set(current) if stored.get(k) != current.get(k)}
            raise CheckpointError("checkpoint was taken under different semantics", differing=differing)

        source = payload["source"]
        if source["name"] != self.source.name:
            raise StreamMismatchError("checkpoint belongs to a different source",
                                      checkpoint=source["name"], current=self.source.name)
        if int(source["length"]) != len(self.source):
            raise StreamMismatchError("checkpoint source has a different length",
                                      checkpoint=int(source["length"]), current=len(self.source))
        if source["fingerprint"] != self.source.fingerprint():
            raise StreamMismatchError(
                "checkpoint source fingerprint does not match; the same events delivered in a "
                "different order are a different stream",
                checkpoint=str(source["fingerprint"])[:16], current=self.source.fingerprint()[:16],
            )
        cursor = int(source["cursor"])
        if not 0 <= cursor <= len(self.source):
            raise CheckpointError("checkpoint cursor is out of range", cursor=cursor, length=len(self.source))
        consumed = int(payload["events_consumed"])
        if consumed < 0:
            raise CheckpointError("checkpoint event count is negative", events_consumed=consumed)

        # The checkpoint owns the run id, so results from before and after the
        # restore group under one run rather than orphaning the sink's inserts.
        self.run_id = str(payload.get("run_id") or self.run_id)
        for sink in self.sinks:
            if isinstance(sink, DatabaseSink):
                sink.run_id = self.run_id
        self.source.seek(cursor)
        self.events_consumed = consumed
        try:
            self.engine.restore(payload["engine"])
            self.features.restore(payload["operators"][self.features.name])
            self.scoring.restore(payload["operators"][self.scoring.name])
            self.alerting.restore(payload["operators"][self.alerting.name])
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointError("checkpoint state is structurally invalid", detail=str(exc)) from exc
        for sink in self.sinks:
            if sink.name in payload["sinks"]:
                sink.restore(payload["sinks"][sink.name])
        self._restored = True
        log.info("checkpoint restored", run_id=self.run_id, cursor=cursor, events_consumed=consumed)

    @staticmethod
    def _require_keys(payload: Any, keys: tuple[str, ...]) -> None:
        if not isinstance(payload, dict):
            raise CheckpointError("checkpoint payload is not an object", got=type(payload).__name__)
        missing = [k for k in keys if k not in payload]
        if missing:
            raise CheckpointError("checkpoint is missing required fields", missing=missing)

    def restore_latest(self) -> bool:
        """Restore from the configured checkpoint file if one exists."""
        if not self.checkpoints.exists():
            return False
        self.restore(self.checkpoints.read())
        return True

    # ---------------------------------------------------------------- shutdown

    def request_stop(self, reason: str = "stop requested") -> None:
        """Ask the loop to checkpoint and stop at the next event boundary."""
        self._stop_requested = reason

    def install_signal_handlers(self) -> None:
        """Turn SIGINT/SIGTERM into a clean checkpointed shutdown.

        Only safe on the main thread, so the CLI installs them and the library
        does not. Without this a container stop loses the in-flight window.
        """
        def handler(signum: int, _frame: types.FrameType | None) -> None:
            self.request_stop(f"signal {signal.Signals(signum).name}")

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):  # pragma: no cover - not the main thread
                log.warning("could not install signal handler", signal=sig)

    # --------------------------------------------------------------------- run

    def run(self) -> RunReport:
        """Consume the stream until drained, stopped or over the wall-clock guard."""
        cfg = self.config
        started = time.monotonic()
        self._start_run_record()
        for sink in self.sinks:
            sink.open()
        log.info("run started", run_id=self.run_id, source=self.source.name,
                 length=len(self.source), cursor=self.source.cursor, restored=self._restored)

        stopped_early = False
        deadline = started + cfg.max_runtime_s
        try:
            for event in self.source.emit():
                self.events_consumed += 1
                for out in self.engine.push(event):
                    self._route_engine_output(out)

                if cfg.heartbeat_every and self.events_consumed % cfg.heartbeat_every == 0:
                    self._heartbeat()
                if cfg.checkpoint_every and self.events_consumed % cfg.checkpoint_every == 0:
                    self._barrier()
                if self._stop_requested:
                    log.info("stopping early", run_id=self.run_id, reason=self._stop_requested,
                             events_consumed=self.events_consumed)
                    self._barrier()
                    stopped_early = True
                    break
                if cfg.stop_after_events is not None and self.events_consumed >= cfg.stop_after_events:
                    if not (cfg.checkpoint_every and self.events_consumed % cfg.checkpoint_every == 0):
                        self._barrier()
                    stopped_early = True
                    break
                if time.monotonic() > deadline:
                    raise RuntimeGuardError("daemon exceeded its wall-clock budget",
                                            max_runtime_s=cfg.max_runtime_s,
                                            events_consumed=self.events_consumed)
        except Exception:
            self._finish_run_record("failed")
            raise

        if not stopped_early:
            for out in self.engine.close():
                self._route_engine_output(out)
        self._flush_operators()
        self._heartbeat()

        stats = self.engine.stats()
        self.metrics_sink.set_runtime(
            {
                **stats,
                "run_id": self.run_id,
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
        for sink in self.sinks:
            sink.flush()
            sink.close()

        report = RunReport(
            run_id=self.run_id,
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
        )
        self._finish_run_record("stopped" if stopped_early else "completed")
        log.info("run finished", run_id=self.run_id, events_consumed=self.events_consumed,
                 dropped_late=stats["events_dropped_late"], drop_rate=stats["drop_rate"],
                 alerts=len(report.alerts), wall_seconds=report.wall_seconds)
        self._check_drop_rate(report)
        return report

    def _check_drop_rate(self, report: RunReport) -> None:
        """A run that threw away too much of its input is not a successful run."""
        rate = report.drop_rate
        if rate <= self.config.max_drop_rate:
            return
        log.error("late-drop rate over budget", run_id=self.run_id, drop_rate=rate,
                  limit=self.config.max_drop_rate,
                  dropped=report.engine_stats["events_dropped_late"])
        self._finish_run_record("failed")
        raise DropRateExceededError(
            "too much of the stream was dropped as too-late",
            run_id=self.run_id,
            drop_rate=round(rate, 6),
            limit=self.config.max_drop_rate,
            dropped=report.engine_stats["events_dropped_late"],
            released=report.engine_stats["events_released"],
        )

    # ------------------------------------------------------------ run records

    def _start_run_record(self) -> None:
        if self.store is None:
            return
        self.store.start_run(
            run_id=self.run_id,
            source_name=self.source.name,
            source_length=len(self.source),
            source_fingerprint=self.source.fingerprint(),
            config=self.config.semantics(),
            correlation_id=correlation_id(),
        )

    def _finish_run_record(self, status: str) -> None:
        if self.store is None:
            return
        self.store.finish_run(
            run_id=self.run_id,
            status=status,
            events_consumed=self.events_consumed,
            events_dropped_late=self.engine.late_dropped,
            alerts_emitted=self.alerting.emitted,
        )

    # ------------------------------------------------------------------ wiring

    def _emit(self, record: Any) -> None:
        """Fan one terminal record out to every sink that declared its type."""
        for sink in self._routes.get(type(record), ()):
            sink.write(record)

    def _route_engine_output(self, out: SlidingTrigger | TumblingResult | LateEvent) -> None:
        if isinstance(out, SlidingTrigger):
            for feature in self.features.process(out):
                self._route_feature(feature)
        else:
            self._emit(out)

    def _route_feature(self, feature: FeatureRecord) -> None:
        for scored in self.scoring.process(feature):
            self._route_scored(scored)

    def _route_scored(self, scored: ScoredRecord) -> None:
        self._emit(scored)
        for alert in self.alerting.process(scored):
            self._emit(alert)

    def _flush_operators(self) -> None:
        """Drain buffered inference. Idempotent, so it is safe at every barrier."""
        for scored in self.scoring.flush():
            self._route_scored(scored)

    def _barrier(self) -> None:
        """Checkpoint barrier: flush every buffer, then persist the whole topology.

        Operators are flushed first so no un-checkpointed work is carried across
        the barrier. Scoring is row-independent and strictly FIFO, so flushing
        early changes how records are grouped into batches but not their values or
        their order — which is why a resumed run reproduces the alert feed byte
        for byte even though the uninterrupted run never flushed at this point.
        """
        self._flush_operators()
        for sink in self.sinks:
            sink.flush()
        payload = self.snapshot()
        self.checkpoints.write(payload)
        if self.store is not None:
            self.store.record_checkpoint(
                run_id=self.run_id,
                cursor=self.source.cursor,
                events_consumed=self.events_consumed,
                version=self.checkpoints.version,
                sha256=self.checkpoints.last_sha256,
                path=str(self.checkpoints.path),
            )
        log.debug("checkpoint written", run_id=self.run_id, cursor=self.source.cursor,
                  events_consumed=self.events_consumed)

    def _heartbeat(self) -> None:
        clock = self.engine.watermark
        self._emit(
            Heartbeat(
                events_consumed=self.events_consumed,
                watermark=clock.effective,
                max_event_time=None if clock.effective is None else clock.max_seen,
                reorder_buffer=self.engine.reorder_buffer_size,
                dropped_late=self.engine.late_dropped,
                alerts_emitted=self.alerting.emitted,
            )
        )
