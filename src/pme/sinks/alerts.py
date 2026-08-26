"""Alert sink: an append-only JSONL feed of maintenance alerts.

The file is written with canonical, sorted-key JSON and no float rounding, so two
runs that computed the same numbers produce byte-identical files. That property
is what the checkpoint/restore test actually asserts on.

On restore the sink truncates the feed back to the checkpointed byte offset before
resuming, so a crash mid-batch cannot leave duplicated or half-written alerts.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core.events import Alert
from .base import Sink


class AlertSink(Sink):
    """Writes alerts to JSONL and keeps them in memory for the run report."""

    name = "alerts"
    accepts = (Alert,)

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.alerts: list[Alert] = []
        self.by_severity: dict[str, int] = {}
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

    def write(self, record: Alert) -> None:
        self.alerts.append(record)
        self.by_severity[record.severity] = self.by_severity.get(record.severity, 0) + 1
        self._handle.write(json.dumps(record.to_json_obj(), sort_keys=True) + "\n")

    def close(self) -> None:
        if self._handle is not None:
            self._handle.flush()
            self._handle.close()
            self._handle = None

    def digest(self) -> bytes:
        """Exact bytes of the feed written so far — the comparison unit for replay tests."""
        if self._handle is not None:
            self._handle.flush()
        return self.path.read_bytes() if self.path.exists() else b""

    def summary(self) -> dict[str, Any]:
        return {"alerts": len(self.alerts), "by_severity": dict(sorted(self.by_severity.items())),
                "path": str(self.path)}

    def state(self) -> dict[str, Any]:
        if self._handle is not None:
            self._handle.flush()
        return {
            "bytes_written": self.path.stat().st_size if self.path.exists() else 0,
            "by_severity": dict(sorted(self.by_severity.items())),
            "count": len(self.alerts),
        }

    def restore(self, state: dict[str, Any]) -> None:
        self._resume_bytes = int(state["bytes_written"])
        self.by_severity = {k: int(v) for k, v in state["by_severity"].items()}
        self.alerts = []
