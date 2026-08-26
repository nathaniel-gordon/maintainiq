"""Event-time watermark generation.

A watermark is the pipeline's assertion that *no event with an earlier event time
will still arrive*. It is what lets an unbounded, out-of-order stream produce
complete window results at all.
"""
from __future__ import annotations

from typing import Any

from .events import MAX_EVENT_TIME, MIN_EVENT_TIME


class BoundedOutOfOrdernessWatermark:
    """Watermark = ``max event time seen - max_out_of_orderness``, clamped monotonic.

    The clamp is not cosmetic: a single very old straggler would otherwise drag
    the watermark backwards, re-opening windows that already fired and destroying
    replay determinism. Monotonicity is asserted by a dedicated property test.
    """

    __slots__ = ("max_out_of_orderness", "_max_seen", "_watermark", "_lag_sum", "_lag_max", "_ticks")

    def __init__(self, max_out_of_orderness: int) -> None:
        if max_out_of_orderness < 0:
            raise ValueError("max_out_of_orderness must be >= 0")
        self.max_out_of_orderness = max_out_of_orderness
        self._max_seen = MIN_EVENT_TIME
        self._watermark = MIN_EVENT_TIME
        self._lag_sum = 0
        self._lag_max = 0
        self._ticks = 0

    @property
    def current(self) -> int:
        return self._watermark

    @property
    def max_seen(self) -> int:
        return self._max_seen

    def observe(self, event_time: int) -> int:
        """Fold an event time in and return the (never-decreasing) watermark."""
        if event_time > self._max_seen:
            self._max_seen = event_time
        candidate = self._max_seen - self.max_out_of_orderness
        if candidate > self._watermark:
            self._watermark = candidate
        self._ticks += 1
        lag = self._max_seen - self._watermark
        self._lag_sum += lag
        if lag > self._lag_max:
            self._lag_max = lag
        return self._watermark

    def close(self) -> int:
        """Advance to +infinity: the source is drained, every window may fire."""
        self._watermark = MAX_EVENT_TIME
        return self._watermark

    def lag_stats(self) -> dict[str, float]:
        """Mean/max distance between the newest event seen and the watermark."""
        mean = self._lag_sum / self._ticks if self._ticks else 0.0
        return {"watermark_lag_mean": round(mean, 3), "watermark_lag_max": float(self._lag_max)}

    def state(self) -> dict[str, Any]:
        return {
            "max_out_of_orderness": self.max_out_of_orderness,
            "max_seen": self._max_seen,
            "watermark": self._watermark,
            "lag_sum": self._lag_sum,
            "lag_max": self._lag_max,
            "ticks": self._ticks,
        }

    def restore(self, state: dict[str, Any]) -> None:
        self.max_out_of_orderness = int(state["max_out_of_orderness"])
        self._max_seen = int(state["max_seen"])
        self._watermark = int(state["watermark"])
        self._lag_sum = int(state["lag_sum"])
        self._lag_max = int(state["lag_max"])
        self._ticks = int(state["ticks"])
