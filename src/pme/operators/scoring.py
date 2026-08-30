"""Model-scoring operator with micro-batched inference.

Calling a gradient-boosting ensemble once per event would dominate the daemon's
runtime, so feature records are buffered and scored in fixed-size micro-batches —
the same trick every production streaming-inference stack uses.

Batching is safe here for a specific reason: all three estimators are
row-independent, so ``score_batch`` returns bit-identical values regardless of how
rows are grouped, and records leave the buffer strictly FIFO. Output order and
output values are therefore invariant to the batch size, which is what lets a
checkpoint barrier flush the buffer without perturbing the result.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..core.events import FeatureRecord, ScoredRecord, Window
from ..learning.bundle import MaintenanceModels
from .base import Operator


class ScoringOperator(Operator):
    """Applies the risk / RUL / anomaly bundle to windowed feature records."""

    name = "scoring"

    def __init__(self, models: MaintenanceModels, batch_size: int = 512) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.models = models
        self.batch_size = batch_size
        self._buffer: list[FeatureRecord] = []
        self.scored = 0
        self.batches = 0

    def process(self, record: FeatureRecord) -> Sequence[ScoredRecord]:
        self._buffer.append(record)
        if len(self._buffer) >= self.batch_size:
            return self._drain()
        return ()

    def flush(self) -> Sequence[ScoredRecord]:
        return self._drain() if self._buffer else ()

    def _drain(self) -> Sequence[ScoredRecord]:
        batch = self._buffer
        self._buffer = []
        x = np.array([r.values for r in batch], dtype=float)
        risk, rul, anomaly = self.models.score_batch(x)
        self.scored += len(batch)
        self.batches += 1
        return [
            ScoredRecord(
                key=rec.key,
                event_time=rec.event_time,
                risk_prob=float(risk[i]),
                rul_est_h=float(rul[i]),
                anomaly=int(anomaly[i]),
            )
            for i, rec in enumerate(batch)
        ]

    def state(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "scored": self.scored,
            "batches": self.batches,
            "pending": [
                [r.key, r.event_time, r.window.start, r.window.end, r.window.kind, list(r.values), r.samples]
                for r in self._buffer
            ],
        }

    def restore(self, state: dict[str, Any]) -> None:
        self.batch_size = int(state["batch_size"])
        self.scored = int(state["scored"])
        self.batches = int(state["batches"])
        self._buffer = [
            FeatureRecord(
                key=row[0],
                event_time=int(row[1]),
                window=Window(int(row[2]), int(row[3]), row[4]),
                values=tuple(float(v) for v in row[5]),
                samples=int(row[6]),
            )
            for row in state["pending"]
        ]
