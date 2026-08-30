"""The windowing engine: watermarks, reorder buffer, lateness policy, firing.

Responsibilities are deliberately narrow. The engine owns *time* — it decides
when an out-of-order event may be released, which window closes at that instant,
which events are hopelessly late, and when a tumbling roll-up is complete. It
owns no model, no feature maths and no IO.

Firing rule (single-fire, lateness-delayed)::

    sliding  window closing at t   fires when  watermark >= t + 1 + allowed_lateness
    tumbling window [s, s+size)   fires when  watermark >= s + size + allowed_lateness

Because a window is held open for the whole allowed-lateness budget and fires
exactly once, *any* delivery order that keeps every event inside that budget
produces identical output to a perfectly ordered stream. That equivalence is the
point of the design and is asserted directly by the property suite.
"""
from __future__ import annotations

import heapq
from typing import Any

from .assigners import SlidingWindows, TumblingWindows
from .events import MAX_EVENT_TIME, SlidingTrigger, TelemetryEvent, TumblingResult, Window
from .watermark import BoundedOutOfOrdernessWatermark

#: Upper edges (in event-time units) of the lateness histogram buckets.
LATENESS_BUCKETS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)


class _TumblingAccumulator:
    """Running roll-up for one (machine, tumbling window) pair."""

    __slots__ = ("failures", "load_sum", "samples", "vib_max", "vib_sum")

    def __init__(self) -> None:
        self.samples = 0
        self.failures = 0
        self.load_sum = 0.0
        self.vib_sum = 0.0
        self.vib_max = float("-inf")

    def add(self, vibration: float, load: float, failure: int) -> None:
        self.samples += 1
        self.failures += failure
        self.load_sum += load
        self.vib_sum += vibration
        self.vib_max = max(self.vib_max, vibration)

    def state(self) -> list[Any]:
        return [self.samples, self.failures, self.load_sum, self.vib_sum, self.vib_max]

    @staticmethod
    def from_state(row: list[Any]) -> _TumblingAccumulator:
        acc = _TumblingAccumulator()
        acc.samples, acc.failures = int(row[0]), int(row[1])
        acc.load_sum, acc.vib_sum, acc.vib_max = float(row[2]), float(row[3]), float(row[4])
        return acc


class WindowingEngine:
    """Event-time windowing with watermarks and an allowed-lateness policy."""

    def __init__(
        self,
        sliding: SlidingWindows,
        tumbling: TumblingWindows,
        watermark: BoundedOutOfOrdernessWatermark,
        allowed_lateness: int = 0,
        vibration_index: int = 0,
    ) -> None:
        if allowed_lateness < 0:
            raise ValueError("allowed_lateness must be >= 0")
        self.sliding = sliding
        self.tumbling = tumbling
        self.watermark = watermark
        self.allowed_lateness = allowed_lateness
        self.vibration_index = vibration_index

        self._buffer: list[tuple[int, int, TelemetryEvent]] = []
        self._seq = 0
        self._acc: dict[tuple[str, int], _TumblingAccumulator] = {}
        self._pending: list[tuple[int, str, int]] = []  # (window_end, key, window_start)

        self.released = 0
        self.late_dropped = 0
        self.lateness_sum = 0
        self.lateness_max = 0
        self.lateness_hist = [0] * (len(LATENESS_BUCKETS) + 1)

    # ------------------------------------------------------------------ timing

    @property
    def reorder_buffer_size(self) -> int:
        """Events admitted but not yet released, i.e. waiting on the watermark."""
        return len(self._buffer)

    @property
    def release_frontier(self) -> int:
        """Largest event time whose sliding window has already fired."""
        return self.watermark.current - self.allowed_lateness - 1

    # ------------------------------------------------------------------ ingest

    def push(self, event: TelemetryEvent) -> list[SlidingTrigger | TumblingResult]:
        """Admit one (possibly out-of-order) event and return whatever fired."""
        frontier = self.release_frontier
        if event.event_time <= frontier:
            self._record_late(event, frontier)
            return []
        heapq.heappush(self._buffer, (event.event_time, self._seq, event))
        self._seq += 1
        self.watermark.observe(event.event_time)
        return self._advance()

    def close(self) -> list[SlidingTrigger | TumblingResult]:
        """Drain: the source is exhausted, so every remaining window may fire."""
        self.watermark.close()
        return self._advance()

    # ------------------------------------------------------------------ firing

    def _advance(self) -> list[SlidingTrigger | TumblingResult]:
        out: list[SlidingTrigger | TumblingResult] = []
        wm = self.watermark.current
        frontier = self.release_frontier
        while self._buffer and self._buffer[0][0] <= frontier:
            _, _, ev = heapq.heappop(self._buffer)
            self.released += 1
            self._accumulate_tumbling(ev)
            out.append(SlidingTrigger(ev.machine_id, self.sliding.closing_window(ev.event_time), ev, wm))
        self._fire_tumbling(out, wm)
        return out

    def _accumulate_tumbling(self, ev: TelemetryEvent) -> None:
        window = self.tumbling.window_of(ev.event_time)
        key = (ev.machine_id, window.start)
        acc = self._acc.get(key)
        if acc is None:
            acc = _TumblingAccumulator()
            self._acc[key] = acc
            heapq.heappush(self._pending, (window.end, ev.machine_id, window.start))
        acc.add(ev.readings[self.vibration_index], ev.load_factor, ev.failure)

    def _fire_tumbling(self, out: list[SlidingTrigger | TumblingResult], wm: int) -> None:
        while self._pending and self._pending[0][0] + self.allowed_lateness <= wm:
            end, key, start = heapq.heappop(self._pending)
            acc = self._acc.pop((key, start))
            out.append(
                TumblingResult(
                    key=key,
                    window=Window(start, end, self.tumbling.kind),
                    watermark=wm,
                    samples=acc.samples,
                    failures=acc.failures,
                    mean_load=acc.load_sum / acc.samples,
                    mean_vibration=acc.vib_sum / acc.samples,
                    max_vibration=acc.vib_max,
                )
            )

    # ---------------------------------------------------------------- lateness

    def _record_late(self, event: TelemetryEvent, frontier: int) -> None:
        lateness = frontier - event.event_time + 1
        self.late_dropped += 1
        self.lateness_sum += lateness
        self.lateness_max = max(self.lateness_max, lateness)
        for i, edge in enumerate(LATENESS_BUCKETS):
            if lateness <= edge:
                self.lateness_hist[i] += 1
                return
        self.lateness_hist[-1] += 1

    def stats(self) -> dict[str, Any]:
        """Late-arrival and watermark telemetry — the observability the batch job never had."""
        total = self.released + self.late_dropped
        labels = [f"<={e}h" for e in LATENESS_BUCKETS] + [f">{LATENESS_BUCKETS[-1]}h"]
        return {
            "events_released": self.released,
            "events_dropped_late": self.late_dropped,
            "drop_rate": round(self.late_dropped / total, 6) if total else 0.0,
            "lateness_mean_h": round(self.lateness_sum / self.late_dropped, 3) if self.late_dropped else 0.0,
            "lateness_max_h": self.lateness_max,
            "lateness_histogram": dict(zip(labels, self.lateness_hist, strict=True)),
            "reorder_buffer_size": len(self._buffer),
            "watermark": self.watermark.current if self.watermark.current != MAX_EVENT_TIME else "closed",
            **self.watermark.lag_stats(),
        }

    # ------------------------------------------------------------- persistence

    def state(self) -> dict[str, Any]:
        return {
            "watermark": self.watermark.state(),
            "buffer": [[t, s, ev.to_row()] for t, s, ev in sorted(self._buffer)],
            "seq": self._seq,
            "acc": [[k[0], k[1], v.state()] for k, v in sorted(self._acc.items())],
            "pending": sorted(self._pending),
            "released": self.released,
            "late_dropped": self.late_dropped,
            "lateness_sum": self.lateness_sum,
            "lateness_max": self.lateness_max,
            "lateness_hist": list(self.lateness_hist),
        }

    def restore(self, state: dict[str, Any]) -> None:
        self.watermark.restore(state["watermark"])
        self._buffer = [(int(t), int(s), TelemetryEvent.from_row(row)) for t, s, row in state["buffer"]]
        heapq.heapify(self._buffer)
        self._seq = int(state["seq"])
        self._acc = {(k, int(ws)): _TumblingAccumulator.from_state(v) for k, ws, v in state["acc"]}
        self._pending = [(int(e), k, int(s)) for e, k, s in state["pending"]]
        heapq.heapify(self._pending)
        self.released = int(state["released"])
        self.late_dropped = int(state["late_dropped"])
        self.lateness_sum = int(state["lateness_sum"])
        self.lateness_max = int(state["lateness_max"])
        self.lateness_hist = [int(v) for v in state["lateness_hist"]]
