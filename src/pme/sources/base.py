"""The source port: where telemetry enters the pipeline.

A source yields events in *delivery order*, which may differ from event-time
order — that is the whole reason the engine needs watermarks. Sources are
cursor-addressable so the daemon can checkpoint its read position and resume.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterator
from typing import Protocol, runtime_checkable

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

    def fingerprint(self) -> str:
        """Stable hash of the delivered event sequence.

        A checkpoint records this so a resume can refuse a stream that is not the
        one the checkpoint was taken against — same file re-delivered in a
        different order included, which used to be accepted silently.
        """
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
        self._fingerprint: str | None = None

    def __len__(self) -> int:
        return len(self._events)

    @property
    def cursor(self) -> int:
        return self._cursor

    def seek(self, cursor: int) -> None:
        if not 0 <= cursor <= len(self._events):
            raise ValueError(f"cursor {cursor} out of range for {len(self._events)} events")
        self._cursor = cursor

    def fingerprint(self) -> str:
        """SHA-256 over ``(machine_id, event_time, readings)`` in delivery order.

        Covers both content and order, which is the pair a restore has to agree
        on: the same events delivered in a different sequence produce a different
        stream and must not be resumed into.
        """
        if self._fingerprint is None:
            digest = hashlib.sha256()
            for event in self._events:
                digest.update(
                    f"{event.machine_id}|{event.event_time}|{event.readings!r}|"
                    f"{event.load_factor!r}|{event.failure}\n".encode()
                )
            self._fingerprint = digest.hexdigest()
        return self._fingerprint

    def emit(self) -> Iterator[TelemetryEvent]:
        while self._cursor < len(self._events):
            event = self._events[self._cursor]
            self._cursor += 1
            yield event

    def history(self) -> pd.DataFrame:
        return self._history
