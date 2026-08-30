"""Checkpoint storage.

Checkpoints are plain JSON, not pickle, for two reasons. First, restore must
reproduce results bit-for-bit and ``json`` round-trips a float through its
shortest exact ``repr`` — no precision is lost, which matters because the
incremental window accumulators carry their own tiny drift and *that drift is
part of the state*. Recomputing the sums on restore would produce values a few
ULP away and break byte-identical replay. Second, a checkpoint you can read in a
text editor is a checkpoint you can debug at 3am.

A checkpoint is still trusted state, so the file records the digest of what was
written and the reader rejects anything it cannot parse, anything with the wrong
version, and anything whose digest does not match. That does not make a
determined tamperer's job impossible — the digest sits next to the payload — but
it does catch truncation, partial writes and edits, which are the realistic
failure modes for a local state file.

Writes are atomic (temp file + ``os.replace``) so a crash during checkpointing
leaves the previous good checkpoint intact.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from ..errors import CheckpointError

#: Bumped whenever the checkpoint layout changes incompatibly.
#: 3: engine gained per-machine late counts; the dead ``load_sum`` accumulator
#: was dropped from the window state; the payload carries a digest.
CHECKPOINT_VERSION = 3


def digest_of(payload: dict[str, Any]) -> str:
    """SHA-256 of the canonical JSON encoding, digest field excluded."""
    body = {k: v for k, v in payload.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()


class CheckpointStore:
    """Atomic single-slot checkpoint file."""

    version = CHECKPOINT_VERSION

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.writes = 0
        self.last_sha256 = ""

    def exists(self) -> bool:
        return self.path.exists()

    def write(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        body = dict(payload)
        body["version"] = CHECKPOINT_VERSION
        body["sha256"] = digest_of(body)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(body, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.path)
        self.writes += 1
        self.last_sha256 = str(body["sha256"])

    def read(self) -> dict[str, Any]:
        try:
            raw = self.path.read_text(encoding="utf-8")
        except OSError as exc:
            raise CheckpointError("checkpoint could not be read", path=str(self.path),
                                  detail=str(exc)) from exc
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CheckpointError("checkpoint is not valid JSON", path=str(self.path),
                                  detail=str(exc)) from exc
        if not isinstance(payload, dict):
            raise CheckpointError("checkpoint is not an object", path=str(self.path))
        version = payload.get("version")
        if version != CHECKPOINT_VERSION:
            raise CheckpointError("checkpoint version is not supported", path=str(self.path),
                                  found=version, expected=CHECKPOINT_VERSION)
        recorded = payload.get("sha256")
        if not isinstance(recorded, str) or recorded != digest_of(payload):
            raise CheckpointError("checkpoint digest does not match its contents", path=str(self.path))
        result: dict[str, Any] = payload
        return result

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()


__all__ = ["CHECKPOINT_VERSION", "CheckpointStore", "digest_of"]
