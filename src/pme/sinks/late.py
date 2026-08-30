"""Late side output: the events the lateness policy refused.

A counter in a metrics file is not a record of lost data. This sink writes every
dropped event to its own JSONL feed so an operator can see exactly which machines
are arriving outside the watermark budget and re-ingest them if they matter.

Like the alert feed it truncates back to the checkpointed byte offset on restore,
so a resumed run neither duplicates nor loses a line.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core.events import LateEvent
from .base import Sink


class LateEventSink(Sink):
    """Append-only JSONL feed of dropped-late events, plus per-machine counts."""

    name = "late"
    accepts = (LateEvent,)

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.count = 0
        self.by_machine: dict[str, int] = {}
        self.max_lateness_h = 0
        self._handle: Any = None
        self._resume_bytes: int | None = None

    def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self._resume_bytes is not None and self.path.exists():
            with self.path.open("r+b") as fh:
                fh.truncate(self._resume_bytes)
            self._handle = self.path.open("a", encoding="utf-8", newline="\n")
        else:
            self._handle = self.path.open("w", encoding="utf-8", newline="\n")

    def write(self, record: LateEvent) -> None:
        self.count += 1
        self.by_machine[record.key] = self.by_machine.get(record.key, 0) + 1
        self.max_lateness_h = max(self.max_lateness_h, record.lateness_h)
        if self._handle is not None:
            self._handle.write(json.dumps(record.to_json_obj(), sort_keys=True) + "\n")

    def close(self) -> None:
        if self._handle is not None:
            self._handle.flush()
            self._handle.close()
            self._handle = None

    def summary(self) -> dict[str, Any]:
        return {
            "late_events": self.count,
            "max_lateness_h": self.max_lateness_h,
            "machines_affected": len(self.by_machine),
            "path": str(self.path),
        }

    def state(self) -> dict[str, Any]:
        if self._handle is not None:
            self._handle.flush()
        return {
            "bytes_written": self.path.stat().st_size if self.path.exists() else 0,
            "count": self.count,
            "max_lateness_h": self.max_lateness_h,
            "by_machine": dict(sorted(self.by_machine.items())),
        }

    def restore(self, state: dict[str, Any]) -> None:
        self._resume_bytes = int(state["bytes_written"])
        self.count = int(state["count"])
        self.max_lateness_h = int(state["max_lateness_h"])
        self.by_machine = {str(k): int(v) for k, v in state["by_machine"].items()}


__all__ = ["LateEventSink"]
