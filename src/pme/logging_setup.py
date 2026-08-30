"""Structured JSON logging with a correlation id on every record.

The daemon, the CLI and the HTTP service all write to the same handler. Each
record carries a ``correlation_id`` taken from a context variable, so a log line
can be tied back to the HTTP request or the run that produced it without
threading an id through every function signature.

Values are never logged blindly: :func:`redact` masks anything whose key looks
like a credential, and the service only ever logs the *presence* of an
``Authorization`` header, never its contents.
"""
from __future__ import annotations

import json
import logging
import os
import secrets
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

#: Set for the lifetime of a run or an HTTP request; empty outside one.
_correlation_id: ContextVar[str] = ContextVar("pme_correlation_id", default="")

#: Substrings that mark a value as a credential. Matching keys are never printed.
_SECRET_HINTS = ("token", "secret", "password", "passwd", "api_key", "apikey", "authorization", "bearer", "key")

_STANDARD_ATTRS = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {"message", "asctime"}


def new_correlation_id() -> str:
    """A short, unguessable id. Short because it is read by humans in logs."""
    return secrets.token_hex(8)


def correlation_id() -> str:
    return _correlation_id.get()


def set_correlation_id(value: str) -> None:
    _correlation_id.set(value)


@contextmanager
def correlation_scope(value: str | None = None) -> Iterator[str]:
    """Bind a correlation id for the duration of the block."""
    cid = value or new_correlation_id()
    token = _correlation_id.set(cid)
    try:
        yield cid
    finally:
        _correlation_id.reset(token)


def redact(fields: dict[str, Any]) -> dict[str, Any]:
    """Replace credential-looking values with ``"***"`` (keys are kept)."""
    out: dict[str, Any] = {}
    for key, value in fields.items():
        lowered = key.lower()
        if any(hint in lowered for hint in _SECRET_HINTS):
            out[key] = "***"
        elif isinstance(value, dict):
            out[key] = redact(value)
        else:
            out[key] = value
    return out


class JsonFormatter(logging.Formatter):
    """One JSON object per line, with any extra keyword fields merged in."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        cid = getattr(record, "correlation_id", "") or correlation_id()
        if cid:
            payload["correlation_id"] = cid
        extras = {k: v for k, v in record.__dict__.items() if k not in _STANDARD_ATTRS and k != "correlation_id"}
        if extras:
            payload.update(redact(extras))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, sort_keys=True)


class PlainFormatter(logging.Formatter):
    """Human-readable fallback for interactive use."""

    def format(self, record: logging.LogRecord) -> str:
        cid = getattr(record, "correlation_id", "") or correlation_id()
        prefix = f"[{cid}] " if cid else ""
        extras = {k: v for k, v in record.__dict__.items() if k not in _STANDARD_ATTRS and k != "correlation_id"}
        suffix = ""
        if extras:
            suffix = " " + " ".join(f"{k}={v}" for k, v in sorted(redact(extras).items()))
        return f"{record.levelname:<7} {prefix}{record.getMessage()}{suffix}"


def _safe_fields(fields: dict[str, Any]) -> dict[str, Any]:
    """Rename any field that would collide with a built-in ``LogRecord`` attribute."""
    return {(f"field_{k}" if k in _STANDARD_ATTRS else k): v for k, v in fields.items()}


class StructuredLogger:
    """Thin wrapper so call sites read ``log.info("thing happened", count=3)``.

    The stdlib logger only accepts extras through an ``extra=`` mapping, which is
    noisy at every call site and silently explodes on a key that collides with a
    ``LogRecord`` attribute. This normalises both.
    """

    __slots__ = ("_logger",)

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    @property
    def stdlib(self) -> logging.Logger:
        return self._logger

    def _emit(self, level: int, msg: str, fields: dict[str, Any], exc_info: bool = False) -> None:
        if self._logger.isEnabledFor(level):
            self._logger.log(level, msg, extra=_safe_fields(fields), exc_info=exc_info)

    def debug(self, msg: str, **fields: Any) -> None:
        self._emit(logging.DEBUG, msg, fields)

    def info(self, msg: str, **fields: Any) -> None:
        self._emit(logging.INFO, msg, fields)

    def warning(self, msg: str, **fields: Any) -> None:
        self._emit(logging.WARNING, msg, fields)

    def error(self, msg: str, **fields: Any) -> None:
        self._emit(logging.ERROR, msg, fields)

    def exception(self, msg: str, **fields: Any) -> None:
        self._emit(logging.ERROR, msg, fields, exc_info=True)


def configure_logging(level: str = "INFO", fmt: str = "json", stream: Any = None) -> None:
    """Install the root handler. Idempotent: repeated calls replace the handler.

    Logs go to stderr so that stdout stays a clean channel for CLI results.
    """
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonFormatter() if fmt == "json" else PlainFormatter())
    root = logging.getLogger("pme")
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.propagate = False


def get_logger(name: str) -> StructuredLogger:
    """Logger under the ``pme`` root, so one call configures every module."""
    return StructuredLogger(logging.getLogger(name if name.startswith("pme") else f"pme.{name}"))


def log_level_from_env(default: str = "INFO") -> str:
    return os.environ.get("PME_LOG_LEVEL", default)


__all__ = [
    "JsonFormatter",
    "PlainFormatter",
    "StructuredLogger",
    "configure_logging",
    "correlation_id",
    "correlation_scope",
    "get_logger",
    "log_level_from_env",
    "new_correlation_id",
    "redact",
    "set_correlation_id",
]
