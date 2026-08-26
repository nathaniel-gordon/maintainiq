"""Stateful per-machine feature operator.

Holds one :class:`MachineWindowState` per machine and folds each released event
into it in O(1). The engine supplies the closing window; the operator evicts
everything before ``window.start`` and emits the feature vector for that window.

This replaces the baseline's ``pandas.rolling`` re-scan. Because the state only
ever contains events the engine has already released — and the engine only
releases an event once the watermark has passed it — a feature vector at event
time ``t`` provably cannot contain data from after ``t``.
"""
from __future__ import annotations

from typing import Any, Sequence

from ..core.events import FeatureRecord, SlidingTrigger
from ..core.rolling import MachineWindowState
from ..core.schema import MIN_WINDOW_SAMPLES
from .base import Operator


class FeatureOperator(Operator):
    """Sliding-window rolling statistics, computed incrementally."""

    name = "features"

    def __init__(self, min_samples: int = MIN_WINDOW_SAMPLES) -> None:
        self.min_samples = min_samples
        self._states: dict[str, MachineWindowState] = {}
        self.emitted = 0
        self.warmup_skipped = 0

    def process(self, record: SlidingTrigger) -> Sequence[FeatureRecord]:
        state = self._states.get(record.key)
        if state is None:
            state = MachineWindowState()
            self._states[record.key] = state
        event = record.event
        state.admit(event.event_time, event.readings, event.load_factor)
        state.evict_before(record.window.start)
        if state.samples < self.min_samples:
            self.warmup_skipped += 1
            return ()
        self.emitted += 1
        return (
            FeatureRecord(
                key=record.key,
                event_time=event.event_time,
                window=record.window,
                values=state.feature_vector(),
                samples=state.samples,
            ),
        )

    def window_state(self, key: str) -> MachineWindowState | None:
        """Expose a machine's aggregate for tests and diagnostics."""
        return self._states.get(key)

    def state(self) -> dict[str, Any]:
        return {
            "min_samples": self.min_samples,
            "emitted": self.emitted,
            "warmup_skipped": self.warmup_skipped,
            "machines": {k: v.state() for k, v in sorted(self._states.items())},
        }

    def restore(self, state: dict[str, Any]) -> None:
        self.min_samples = int(state["min_samples"])
        self.emitted = int(state["emitted"])
        self.warmup_skipped = int(state["warmup_skipped"])
        self._states = {k: MachineWindowState.from_state(v) for k, v in state["machines"].items()}
