"""Path normalisation and containment.

Every path the process takes from a caller — ``--data``, ``--output``,
``--models``, a database location — goes through here first. As a local CLI the
risk is modest, but the same code is reachable from the HTTP service's
configuration and from a container entrypoint, where an unnormalised path is a
traversal bug. Normalising once, in one place, keeps that from ever being a
question.
"""
from __future__ import annotations

from pathlib import Path

from .errors import PathError

#: Refuse telemetry files larger than this unless the caller raises it explicitly.
#: A materialised source holds the whole file in memory twice, so an unbounded
#: read is an availability problem rather than a slow one.
DEFAULT_MAX_INPUT_BYTES = 256 * 1024 * 1024


def normalize(path: str | Path) -> Path:
    """Absolute, symlink-free-where-possible, ``..``-free version of ``path``."""
    try:
        return Path(path).expanduser().resolve()
    except (OSError, RuntimeError) as exc:  # pragma: no cover - platform dependent
        raise PathError("path could not be resolved", path=str(path), reason=str(exc)) from exc


def resolve_under(root: str | Path, candidate: str | Path) -> Path:
    """Resolve ``candidate`` and require that it stays inside ``root``.

    Used wherever a path component comes from outside the process (a request
    parameter, a config file) rather than from the operator's own shell.
    """
    base = normalize(root)
    target = normalize(base / candidate) if not Path(candidate).is_absolute() else normalize(candidate)
    if target != base and base not in target.parents:
        raise PathError("path escapes its permitted root", root=str(base), path=str(target))
    return target


def ensure_directory(path: str | Path) -> Path:
    """Normalise, create if missing, and confirm it is a directory."""
    target = normalize(path)
    if target.exists() and not target.is_dir():
        raise PathError("output path exists and is not a directory", path=str(target))
    target.mkdir(parents=True, exist_ok=True)
    return target


def ensure_readable_file(path: str | Path, max_bytes: int = DEFAULT_MAX_INPUT_BYTES) -> Path:
    """Normalise and confirm the file exists, is a regular file and fits the cap."""
    target = normalize(path)
    if not target.exists():
        raise PathError("input file does not exist", path=str(target))
    if not target.is_file():
        raise PathError("input path is not a regular file", path=str(target))
    size = target.stat().st_size
    if size > max_bytes:
        raise PathError("input file exceeds the size limit", path=str(target), size=size, limit=max_bytes)
    if size == 0:
        raise PathError("input file is empty", path=str(target))
    return target


__all__ = [
    "DEFAULT_MAX_INPUT_BYTES",
    "ensure_directory",
    "ensure_readable_file",
    "normalize",
    "resolve_under",
]
