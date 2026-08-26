"""The source port: where telemetry enters the pipeline.

A source yields events in *delivery order*, which may differ from event-time
order — that is the whole reason the engine needs watermarks. Sources are
cursor-addressable so the daemon can checkpoint its read position and resume.
"""
from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable

import pandas as pd

from ..core.events import TelemetryEvent


@runtime_checkable
class TelemetrySource(Protocol):
    """Replayable, cursor-addressable stream of telemetry events."""

    name: str

    def __len__(self) -> int:
        """Total events the source will ever deliver (sources here are bounded)."""
        ...

    @property
    def cursor(self) -> int:
        """Index of the next event to be delivered."""
        ...

    def seek(self, cursor: int) -> None:
        """Reposition the read cursor (used to resume from a checkpoint)."""
        ...

    def emit(self) -> Iterator[TelemetryEvent]:
        """Yield events from the current cursor to the end of the stream."""
        ...

    def history(self) -> pd.DataFrame:
        """Complete event-time-ordered ground truth, for offline label building.

        Labels (time-to-next-failure) are inherently backward-looking and can only
        be built from a settled historian export, never from the live stream.
        """
        ...


class MaterializedSource:
    """Base for sources backed by an in-memory list of events.

    Materialising is what makes the cursor semantics exact: ``seek(k)`` resumes at
    precisely the event the uninterrupted run saw at position ``k``, which is what
    byte-identical checkpoint restore requires.
    """

    name = "materialized"

    def __init__(self, events: list[TelemetryEvent], history: pd.DataFrame) -> None:
        self._events = events
        self._history = history
        self._cursor = 0

    def __len__(self) -> int:
        return len(self._events)

    @property
    def cursor(self) -> int:
        return self._cursor

    def seek(self, cursor: int) -> None:
        if not 0 <= cursor <= len(self._events):
            raise ValueError(f"cursor {cursor} out of range for {len(self._events)} events")
        self._cursor = cursor

    def emit(self) -> Iterator[TelemetryEvent]:
        while self._cursor < len(self._events):
            event = self._events[self._cursor]
            self._cursor += 1
            yield event

    def history(self) -> pd.DataFrame:
        return self._history
