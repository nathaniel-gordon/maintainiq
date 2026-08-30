"""Database sink: the run's results, queryable after the fact.

The JSONL feeds are the durable artifact; this sink is what makes them
answerable — "which machines alerted CRITICAL last run", "how late were the
events we dropped", "what is every machine's current risk" — without re-reading
and re-parsing files. It is also what the HTTP service reads.

Writes are batched and flushed at checkpoint barriers and at close, so a run
performs one transaction per batch rather than one per record. Alerts insert with
``OR IGNORE`` on their natural key, which makes replaying a window after a
restore a no-op rather than a duplicate.
"""
from __future__ import annotations

from typing import Any

from ..core.events import Alert, LateEvent, ScoredRecord
from ..logging_setup import correlation_id, get_logger
from ..store.repository import MetadataStore
from .base import Sink

log = get_logger(__name__)

#: Rows buffered per table before an automatic flush.
DEFAULT_BATCH = 500


class DatabaseSink(Sink):
    """Persists alerts, machine state and late events into the metadata store."""

    name = "database"
    accepts = (Alert, ScoredRecord, LateEvent)

    def __init__(self, store: MetadataStore, run_id: str, batch_size: int = DEFAULT_BATCH) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.store = store
        self.run_id = run_id
        self.batch_size = batch_size
        self._alerts: list[dict[str, Any]] = []
        self._state: dict[str, dict[str, Any]] = {}
        self._late: list[dict[str, Any]] = []
        self.alerts_written = 0
        self.late_written = 0
        self.state_written = 0

    def write(self, record: Alert | ScoredRecord | LateEvent) -> None:
        if isinstance(record, Alert):
            self._alerts.append(record.to_json_obj())
        elif isinstance(record, LateEvent):
            self._late.append(
                {"machine_id": record.key, "event_time": record.event_time,
                 "watermark": record.watermark, "lateness_h": record.lateness_h}
            )
        else:
            current = self._state.get(record.key)
            if current is None or record.event_time >= int(current["event_time"]):
                self._state[record.key] = {
                    "machine_id": record.key, "event_time": record.event_time,
                    "risk_prob": record.risk_prob, "rul_est_h": record.rul_est_h,
                    "anomaly": record.anomaly,
                }
        if len(self._alerts) >= self.batch_size or len(self._late) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        """Drain every buffer into the database. Safe to call at any barrier."""
        if self._alerts:
            self.alerts_written += self.store.record_alerts(self.run_id, self._alerts, correlation_id())
            self._alerts.clear()
        if self._late:
            self.store.record_late_events(self.run_id, self._late)
            self.late_written += len(self._late)
            self._late.clear()
        if self._state:
            self.store.upsert_machine_state(self.run_id, list(self._state.values()))
            self.state_written += len(self._state)
            self._state.clear()

    def close(self) -> None:
        self.flush()
        log.debug("database sink closed", run_id=self.run_id, alerts=self.alerts_written,
                  late_events=self.late_written)

    def summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "alerts_written": self.alerts_written,
            "late_events_written": self.late_written,
            "machine_state_writes": self.state_written,
        }


__all__ = ["DEFAULT_BATCH", "DatabaseSink"]
