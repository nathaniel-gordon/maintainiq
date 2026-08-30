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
    replay determinism.
    """

    __slots__ = ("_lag_max", "_lag_sum", "_max_seen", "_ticks", "_watermark", "max_out_of_orderness")

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

    @property
    def is_closed(self) -> bool:
        """True once :meth:`close` has pushed the watermark past every event."""
        return self._watermark == MAX_EVENT_TIME

    @property
    def effective(self) -> int | None:
        """Watermark clamped to the newest event seen; ``None`` before the first.

        Reporting the raw value after drain means reporting 2**62, and reporting it
        before the first event means reporting -2**62. Neither is a time anyone
        wants in a metrics file.
        """
        if self._max_seen == MIN_EVENT_TIME:
            return None
        return min(self._watermark, self._max_seen)

    def observe(self, event_time: int) -> int:
        """Fold an event time in and return the (never-decreasing) watermark."""
        self._max_seen = max(self._max_seen, event_time)
        candidate = self._max_seen - self.max_out_of_orderness
        self._watermark = max(self._watermark, candidate)
        self._ticks += 1
        lag = self._max_seen - self._watermark
        self._lag_sum += lag
        self._lag_max = max(self._lag_max, lag)
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
