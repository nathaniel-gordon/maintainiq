"""Checkpoint file format: atomic writes, versioning and digest checks."""
from __future__ import annotations

import json

import pytest

from pme.errors import CheckpointError
from pme.runtime.checkpoint import CHECKPOINT_VERSION, CheckpointStore, digest_of


@pytest.fixture
def store(tmp_path) -> CheckpointStore:
    return CheckpointStore(tmp_path / "checkpoint.json")


def test_round_trip_preserves_the_payload(store):
    payload = {"events_consumed": 7, "engine": {"released": 3, "drift": 0.1 + 0.2}}
    store.write(payload)
    read = store.read()
    assert read["events_consumed"] == 7
    assert read["engine"]["drift"] == 0.1 + 0.2
    assert read["version"] == CHECKPOINT_VERSION
    assert store.writes == 1
    assert store.last_sha256 == read["sha256"]


def test_write_leaves_no_temp_file_behind(store):
    store.write({"a": 1})
    assert list(store.path.parent.glob("*.tmp")) == []


def test_a_tampered_payload_is_refused(store):
    store.write({"events_consumed": 7})
    payload = json.loads(store.path.read_text(encoding="utf-8"))
    payload["events_consumed"] = 9_999
    store.path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(CheckpointError, match="digest"):
        store.read()


def test_a_truncated_file_is_refused(store):
    store.write({"events_consumed": 7})
    raw = store.path.read_text(encoding="utf-8")
    store.path.write_text(raw[: len(raw) // 2], encoding="utf-8")
    with pytest.raises(CheckpointError, match="valid JSON"):
        store.read()


def test_an_older_version_is_refused(store):
    payload = {"events_consumed": 1, "version": CHECKPOINT_VERSION - 1}
    payload["sha256"] = digest_of(payload)
    store.path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(CheckpointError, match="version"):
        store.read()


def test_a_json_scalar_is_not_a_checkpoint(store):
    store.path.write_text("42", encoding="utf-8")
    with pytest.raises(CheckpointError):
        store.read()


def test_clear_removes_the_slot(store):
    store.write({"a": 1})
    assert store.exists()
    store.clear()
    assert not store.exists()
