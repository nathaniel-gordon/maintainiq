"""Checkpoint storage.

Checkpoints are plain JSON, not pickle, for two reasons. First, restore must
reproduce results bit-for-bit and ``json`` round-trips a float through its
shortest exact ``repr`` — no precision is lost, which matters because the
incremental window accumulators carry their own tiny drift and *that drift is
part of the state*. Recomputing the sums on restore would produce values a few
ULP away and break byte-identical replay. Second, a checkpoint you can read in a
text editor is a checkpoint you can debug at 3am.

Writes are atomic (temp file + ``os.replace``) so a crash during checkpointing
leaves the previous good checkpoint intact.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

#: Bumped whenever the checkpoint layout changes incompatibly.
CHECKPOINT_VERSION = 2


class CheckpointStore:
    """Atomic single-slot checkpoint file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.writes = 0

    def exists(self) -> bool:
        return self.path.exists()

    def write(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        body = dict(payload)
        body["version"] = CHECKPOINT_VERSION
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(body, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.path)
        self.writes += 1

    def read(self) -> dict[str, Any]:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        version = payload.get("version")
        if version != CHECKPOINT_VERSION:
            raise ValueError(f"checkpoint version {version} != expected {CHECKPOINT_VERSION}")
        return payload

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()
