"""Immutable records that travel through the streaming pipeline.

Every record carries an explicit **event time** (the hour the reading was taken on
the machine) rather than a processing time. Event time is the only clock the
pipeline trusts: it is what makes replay, out-of-order delivery and
checkpoint/restore produce identical results regardless of when the daemon runs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: Sensor channels, in the fixed order used by every readings tuple in the system.
#: The order is part of the wire format (checkpoints store bare tuples), so it is
#: defined once here and never re-derived.
SENSORS: tuple[str, ...] = (
    "vibration_mm_s",
    "temp_c",
    "pressure_bar",
    "rpm",
    "oil_quality_pct",
)

#: Sentinel used as "the stream is drained": a watermark of +infinity closes every
#: outstanding window. An int (not math.inf) keeps every timestamp comparison integral.
MAX_EVENT_TIME: int = 1 << 62

#: Sentinel used before any event has been seen.
MIN_EVENT_TIME: int = -(1 << 62)


@dataclass(frozen=True, slots=True)
class TelemetryEvent:
    """One sensor sweep from one machine at one event-time hour.

    Frozen because events are shared between the reorder buffer, the operator
    state and the checkpoint writer; accidental mutation anywhere would silently
    break replay determinism.
    """

    machine_id: str
    event_time: int
    readings: tuple[float, ...]
    load_factor: float
    failure: int = 0

    def to_row(self) -> list[Any]:
        """Serialise to a plain JSON list (checkpoints must not depend on pickle)."""
        return [self.machine_id, self.event_time, list(self.readings), self.load_factor, self.failure]

    @staticmethod
    def from_row(row: list[Any]) -> "TelemetryEvent":
        return TelemetryEvent(row[0], int(row[1]), tuple(float(v) for v in row[2]), float(row[3]), int(row[4]))


@dataclass(frozen=True, slots=True)
class Window:
    """A half-open event-time interval ``[start, end)``.

    ``kind`` distinguishes the two assigners that produce windows so downstream
    stages can dispatch without re-deriving the geometry.
    """

    start: int
    end: int
    kind: str

    @property
    def max_timestamp(self) -> int:
        """Largest event time that still belongs to this window."""
        return self.end - 1


@dataclass(frozen=True, slots=True)
class SlidingTrigger:
    """The engine released ``event`` into the sliding window ``window`` for ``key``.

    Emitted in strict event-time order per key. The stateful feature operator
    folds the event into its incremental aggregate and evicts anything older than
    ``window.start``.
    """

    key: str
    window: Window
    event: TelemetryEvent
    watermark: int


@dataclass(frozen=True, slots=True)
class TumblingResult:
    """A completed tumbling window of raw telemetry health for one machine."""

    key: str
    window: Window
    watermark: int
    samples: int
    failures: int
    mean_load: float
    mean_vibration: float
    max_vibration: float


@dataclass(frozen=True, slots=True)
class FeatureRecord:
    """Model-ready feature vector for one machine at one event time.

    ``values`` is aligned with :func:`pme.core.schema.feature_names`. It is built
    only from events whose event time lies inside ``window`` — never from the
    future — which is what the no-leakage property test verifies.
    """

    key: str
    event_time: int
    window: Window
    values: tuple[float, ...]
    samples: int


@dataclass(frozen=True, slots=True)
class ScoredRecord:
    """Feature record plus the three model outputs."""

    key: str
    event_time: int
    risk_prob: float
    rul_est_h: float
    anomaly: int


@dataclass(frozen=True, slots=True)
class Heartbeat:
    """Periodic runtime sample emitted by the daemon into the same fan-out.

    Routing daemon health through the pipeline rather than around it means the
    metrics sink is the single place that knows how to persist observability, and
    the daemon keeps no reporting logic of its own.
    """

    events_consumed: int
    watermark: int
    max_event_time: int
    reorder_buffer: int
    dropped_late: int
    alerts_emitted: int


@dataclass(frozen=True, slots=True)
class Alert:
    """A maintenance alert emitted the moment its window closed."""

    key: str
    event_time: int
    severity: str
    risk_prob: float
    rul_est_h: float
    anomaly: int
    reason: str

    def to_json_obj(self) -> dict[str, Any]:
        """Ordered plain dict; the alert sink writes these as canonical JSON lines."""
        return {
            "machine_id": self.key,
            "event_time": self.event_time,
            "severity": self.severity,
            "risk_prob": self.risk_prob,
            "rul_est_h": self.rul_est_h,
            "anomaly": self.anomaly,
            "reason": self.reason,
        }
