"""The operator port: a stateful, checkpointable stage of the streaming pipeline.

Operators are chained by the daemon and never call each other directly. Each one
takes records in, returns records out, and can hand its entire keyed state to the
checkpoint writer — the contract that makes mid-stream restore possible.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence


class Operator(ABC):
    """One stage of the pipeline."""

    #: Stable identifier used as the operator's key inside a checkpoint file.
    name: str = "operator"

    @abstractmethod
    def process(self, record: Any) -> Sequence[Any]:
        """Handle one input record; return zero or more output records."""

    def flush(self) -> Sequence[Any]:
        """Emit anything buffered. Called at a checkpoint barrier and at drain.

        Flushing at the barrier is what keeps a restored run aligned with an
        uninterrupted one: no operator carries un-checkpointed buffered work.
        """
        return ()

    @abstractmethod
    def state(self) -> dict[str, Any]:
        """JSON-serialisable snapshot of all keyed state."""

    @abstractmethod
    def restore(self, state: dict[str, Any]) -> None:
        """Reinstate a snapshot produced by :meth:`state`."""
