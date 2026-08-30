"""Window assigners: pure functions from an event time to the windows it belongs to.

Assigners hold no state and touch no clock, so window geometry is trivially
property-testable in isolation from the rest of the engine.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from .events import Window


@runtime_checkable
class WindowAssigner(Protocol):
    """Maps an event time onto the windows that must contain it."""

    kind: str

    def assign(self, event_time: int) -> tuple[Window, ...]:
        """Every window this event time belongs to, ordered by ``start``."""
        ...


class TumblingWindows:
    """Fixed-size, non-overlapping windows aligned to multiples of ``size``.

    Used for fleet-health roll-ups (samples, failures, load) that must partition
    the timeline exactly once — hence tumbling rather than sliding.
    """

    kind = "tumbling"

    def __init__(self, size: int, offset: int = 0) -> None:
        if size <= 0:
            raise ValueError("tumbling window size must be positive")
        self.size = size
        self.offset = offset

    def assign(self, event_time: int) -> tuple[Window, ...]:
        start = ((event_time - self.offset) // self.size) * self.size + self.offset
        return (Window(start, start + self.size, self.kind),)

    def window_of(self, event_time: int) -> Window:
        """Convenience accessor: tumbling assignment is always a single window."""
        return self.assign(event_time)[0]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"TumblingWindows(size={self.size}, offset={self.offset})"


class SlidingWindows:
    """Overlapping windows of ``size`` advancing every ``step`` time units.

    With ``step == 1`` this is the classic rolling-feature window. The engine
    never materialises the ``size/step`` overlapping panes: it uses the single
    window that *closes* at each event time and lets the stateful operator carry
    an evicting incremental aggregate. :meth:`assign` remains the honest full
    assignment.
    """

    kind = "sliding"

    def __init__(self, size: int, step: int = 1) -> None:
        if size <= 0 or step <= 0:
            raise ValueError("sliding window size and step must be positive")
        if size % step != 0:
            raise ValueError("sliding window size must be a whole multiple of step")
        self.size = size
        self.step = step

    def assign(self, event_time: int) -> tuple[Window, ...]:
        last_start = (event_time // self.step) * self.step
        starts = range(last_start - self.size + self.step, last_start + self.step, self.step)
        return tuple(Window(s, s + self.size, self.kind) for s in starts)

    def closing_window(self, event_time: int) -> Window:
        """The single window whose ``max_timestamp`` is exactly ``event_time``.

        This is the window the engine fires when it releases an event: all earlier
        windows containing the event are still open, all later ones have closed.
        """
        return Window(event_time - self.size + 1, event_time + 1, self.kind)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"SlidingWindows(size={self.size}, step={self.step})"
