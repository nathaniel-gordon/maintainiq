"""Path normalisation and the input-size cap."""
from __future__ import annotations

import pytest

from pme.errors import PathError
from pme.paths import (
    DEFAULT_MAX_INPUT_BYTES,
    ensure_directory,
    ensure_readable_file,
    normalize,
)


def test_normalize_returns_an_absolute_path(tmp_path):
    assert normalize(tmp_path / "a" / ".." / "b").is_absolute()


def test_normalize_removes_dot_dot_segments(tmp_path):
    assert normalize(tmp_path / "a" / ".." / "b") == normalize(tmp_path / "b")


def test_ensure_directory_creates_missing_parents(tmp_path):
    target = ensure_directory(tmp_path / "a" / "b" / "c")
    assert target.is_dir()


def test_ensure_directory_is_idempotent(tmp_path):
    first = ensure_directory(tmp_path / "out")
    assert ensure_directory(tmp_path / "out") == first


def test_ensure_directory_refuses_an_existing_file(tmp_path):
    target = tmp_path / "not-a-dir"
    target.write_text("x", encoding="utf-8")
    with pytest.raises(PathError, match="not a directory"):
        ensure_directory(target)


def test_ensure_readable_file_accepts_a_real_file(tmp_path):
    target = tmp_path / "t.csv"
    target.write_text("machine_id,hour\n", encoding="utf-8")
    assert ensure_readable_file(target) == normalize(target)


def test_a_missing_file_is_refused(tmp_path):
    with pytest.raises(PathError, match="does not exist"):
        ensure_readable_file(tmp_path / "nope.csv")


def test_a_directory_is_not_a_readable_file(tmp_path):
    with pytest.raises(PathError, match="not a regular file"):
        ensure_readable_file(tmp_path)


def test_an_empty_file_is_refused(tmp_path):
    target = tmp_path / "empty.csv"
    target.touch()
    with pytest.raises(PathError, match="empty"):
        ensure_readable_file(target)


def test_a_file_over_the_cap_is_refused(tmp_path):
    target = tmp_path / "big.csv"
    target.write_bytes(b"x" * 64)
    with pytest.raises(PathError, match="size limit") as excinfo:
        ensure_readable_file(target, max_bytes=32)
    assert excinfo.value.details["limit"] == 32


def test_the_default_cap_is_large_but_finite():
    assert 0 < DEFAULT_MAX_INPUT_BYTES < 1 << 40
