"""The sink port: where results leave the pipeline.

Sinks declare which record types they accept so the daemon can fan a single
output stream out to all of them without any sink knowing another exists. Adding
a new destination (a message bus, a CMMS work-order API) means one new class and
one config entry — the pipeline itself does not change.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Sink(ABC):
    """A terminal stage. Sinks may buffer, but must be idempotent across restore."""

    name: str = "sink"

    #: Record classes this sink consumes; the daemon uses it for dispatch.
    accepts: tuple[type, ...] = ()

    def open(self) -> None:
        """Acquire resources (open files, connect). Called once before the run."""

    @abstractmethod
    def write(self, record: Any) -> None:
        """Consume one record of an accepted type."""

    def close(self) -> None:
        """Release resources and materialise any deferred output."""

    def summary(self) -> dict[str, Any]:
        """Counters worth printing at the end of a run."""
        return {}

    def state(self) -> dict[str, Any]:
        """Snapshot for checkpointing. Sinks with no resumable state return ``{}``."""
        return {}

    def restore(self, state: dict[str, Any]) -> None:
        """Reinstate a snapshot produced by :meth:`state`."""
