"""The error taxonomy shared by the library, the CLI and the HTTP service.

Every failure this system raises on purpose is a :class:`PmeError` carrying a
stable machine-readable ``code``, a human ``message`` and a ``details`` mapping.
The three surfaces render the same object differently: the CLI maps ``exit_code``
onto the process status, the HTTP service maps ``http_status`` onto the response
and serialises :meth:`PmeError.to_dict`, and the logger records the code so a
failure can be counted without parsing prose.

Codes are part of the interface. Add new ones; do not rename existing ones.
"""
from __future__ import annotations

from typing import Any

#: Process exit statuses. 0/1 keep their usual meaning, everything else is a
#: category an operator can branch on in a shell script.
EXIT_OK = 0
EXIT_USAGE = 2
EXIT_DATA = 3
EXIT_INTEGRITY = 4
EXIT_INTERNAL = 5


class PmeError(Exception):
    """Base class for every deliberate failure."""

    code: str = "internal_error"
    http_status: int = 500
    exit_code: int = EXIT_INTERNAL

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details

    def to_dict(self) -> dict[str, Any]:
        """Wire form: ``{"error": {"code", "message", "details"}}`` is built from this."""
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details:
            payload["details"] = self.details
        return payload

    def __str__(self) -> str:
        if not self.details:
            return self.message
        rendered = ", ".join(f"{k}={v!r}" for k, v in sorted(self.details.items()))
        return f"{self.message} ({rendered})"


# --------------------------------------------------------------------- config


class ConfigError(PmeError):
    """A configuration value is impossible or two settings contradict each other."""

    code = "config_invalid"
    http_status = 500
    exit_code = EXIT_USAGE


class TrainServeSkewError(ConfigError):
    """The serving gate does not match the gate the bundle was trained under."""

    code = "train_serve_skew"


# ----------------------------------------------------------------- input data


class ValidationError(PmeError):
    """Caller-supplied input failed validation before any work was done."""

    code = "validation_failed"
    http_status = 400
    exit_code = EXIT_DATA


class PathError(ValidationError):
    """A path argument escaped its permitted root or is otherwise unusable."""

    code = "path_rejected"


class DataQualityError(ValidationError):
    """A telemetry file is too damaged to run against."""

    code = "data_quality"


# --------------------------------------------------------------------- stream


class SourceError(PmeError):
    """The telemetry source could not be opened or read."""

    code = "source_unavailable"
    exit_code = EXIT_DATA


class DropRateExceededError(PmeError):
    """More of the stream was discarded as too-late than the run permits."""

    code = "drop_rate_exceeded"
    exit_code = EXIT_DATA


class RuntimeGuardError(PmeError):
    """The wall-clock guard fired: the run took longer than it was allowed to."""

    code = "runtime_guard"


# ----------------------------------------------------------------- durability


class CheckpointError(PmeError):
    """A checkpoint is missing, unreadable, or structurally invalid."""

    code = "checkpoint_invalid"
    exit_code = EXIT_INTEGRITY


class StreamMismatchError(CheckpointError):
    """The checkpoint was taken against a different stream than the one supplied."""

    code = "stream_mismatch"
    http_status = 409


class StorageError(PmeError):
    """The metadata store rejected an operation."""

    code = "storage_failure"


class ModelIntegrityError(PmeError):
    """A model bundle failed its checksum or authenticity check."""

    code = "model_integrity"
    exit_code = EXIT_INTEGRITY


# ------------------------------------------------------------------ interface


class NotFoundError(PmeError):
    """No such resource."""

    code = "not_found"
    http_status = 404
    exit_code = EXIT_USAGE


class MethodNotAllowedError(PmeError):
    """The resource exists but not for this HTTP method."""

    code = "method_not_allowed"
    http_status = 405
    exit_code = EXIT_USAGE


class UnauthorizedError(PmeError):
    """A bearer token is required and was absent or wrong."""

    code = "unauthorized"
    http_status = 401
    exit_code = EXIT_USAGE


class RateLimitedError(PmeError):
    """The caller exceeded the per-client request budget."""

    code = "rate_limited"
    http_status = 429
    exit_code = EXIT_USAGE


__all__ = [
    "EXIT_DATA",
    "EXIT_INTEGRITY",
    "EXIT_INTERNAL",
    "EXIT_OK",
    "EXIT_USAGE",
    "CheckpointError",
    "ConfigError",
    "DataQualityError",
    "DropRateExceededError",
    "MethodNotAllowedError",
    "ModelIntegrityError",
    "NotFoundError",
    "PathError",
    "PmeError",
    "RateLimitedError",
    "RuntimeGuardError",
    "SourceError",
    "StorageError",
    "StreamMismatchError",
    "TrainServeSkewError",
    "UnauthorizedError",
    "ValidationError",
]
