"""Telemetry validation: the boundary between a file and the pipeline.

Everything that arrives from outside is suspect. A single NaN in one sensor cell
used to surface as a raw estimator traceback from inside the scoring operator,
half-way through a run, after output had already been written. That is the wrong
place to find out: the file is checkable up front, so it is checked up front.

Bad rows are *quarantined*, not fatal — a historian export with a handful of
dropped cells should still produce a run — but the fraction is bounded, and every
rejected row is recorded with its row number, a stable code and the payload that
failed, so nothing disappears silently.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..core.events import SENSORS
from ..errors import DataQualityError

#: Physically plausible envelopes for each channel. Deliberately wide: the job
#: here is to catch a mis-scaled column or a sentinel like -9999, not to
#: second-guess a plant's operating point.
SENSOR_RANGES: dict[str, tuple[float, float]] = {
    "vibration_mm_s": (0.0, 200.0),
    "temp_c": (-50.0, 400.0),
    "pressure_bar": (0.0, 400.0),
    "rpm": (0.0, 30_000.0),
    "oil_quality_pct": (0.0, 100.0),
}

LOAD_FACTOR_RANGE: tuple[float, float] = (0.0, 5.0)

REQUIRED_COLUMNS: tuple[str, ...] = ("machine_id", "hour", *SENSORS)

#: Above this share of unusable rows the file is treated as the wrong file rather
#: than a slightly damaged one.
DEFAULT_MAX_BAD_FRACTION = 0.05

#: What to do when the same (machine_id, hour) appears more than once.
DUPLICATE_POLICIES = ("last", "first", "reject")


@dataclass(frozen=True, slots=True)
class BadRow:
    """One rejected input row, ready for the quarantine table."""

    row_number: int
    code: str
    detail: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"row_number": self.row_number, "code": self.code, "detail": self.detail, "payload": self.payload}


@dataclass(slots=True)
class ValidationOutcome:
    """The clean frame plus everything that did not make it in."""

    frame: pd.DataFrame
    rejected: list[BadRow] = field(default_factory=list)
    duplicates_dropped: int = 0
    rows_in: int = 0

    @property
    def rows_out(self) -> int:
        return len(self.frame)

    def summary(self) -> dict[str, Any]:
        codes: dict[str, int] = {}
        for row in self.rejected:
            codes[row.code] = codes.get(row.code, 0) + 1
        return {
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "rejected": len(self.rejected),
            "duplicates_dropped": self.duplicates_dropped,
            "reject_codes": dict(sorted(codes.items())),
        }


def _payload(frame: pd.DataFrame, position: int) -> dict[str, Any]:
    """A JSON-safe copy of one row, for the quarantine record."""
    raw = frame.iloc[position].to_dict()
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, float) and not math.isfinite(value):
            out[str(key)] = str(value)
        elif isinstance(value, np.generic):
            out[str(key)] = value.item()
        else:
            out[str(key)] = value
    return out


def validate_frame(
    frame: pd.DataFrame,
    source: str,
    max_bad_fraction: float = DEFAULT_MAX_BAD_FRACTION,
    duplicate_policy: str = "last",
) -> ValidationOutcome:
    """Coerce, range-check and de-duplicate a telemetry frame.

    Returns the surviving rows plus a quarantine list. Raises
    :class:`~pme.errors.DataQualityError` only when the file as a whole cannot be
    used: missing columns, an empty result, or more damage than the caller allows.
    """
    if duplicate_policy not in DUPLICATE_POLICIES:
        raise DataQualityError("unknown duplicate policy", policy=duplicate_policy,
                               allowed=list(DUPLICATE_POLICIES))
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise DataQualityError("telemetry is missing required columns", source=source, missing=missing)

    work = frame.reset_index(drop=True).copy()
    work["_row_number"] = np.arange(len(work), dtype=np.int64)
    rows_in = len(work)
    rejected: list[BadRow] = []
    keep = np.ones(len(work), dtype=bool)

    def reject(mask: pd.Series[bool], code: str, detail: str) -> None:
        mask = mask.to_numpy(dtype=bool) & keep
        for position in np.flatnonzero(mask):
            rejected.append(BadRow(int(work.at[position, "_row_number"]), code, detail, _payload(work, int(position))))
        keep[mask] = False

    # machine_id: a non-empty label. Anything else makes the key meaningless.
    ids = work["machine_id"].astype("string")
    reject(ids.isna() | (ids.str.strip() == ""), "missing_machine_id", "machine_id")
    work["machine_id"] = ids.fillna("").str.strip()

    # hour: the event time. Must be a non-negative integer.
    hours = pd.to_numeric(work["hour"], errors="coerce")
    reject(hours.isna(), "hour_not_numeric", "hour")
    reject(hours.notna() & (hours % 1 != 0), "hour_not_integer", "hour")
    reject(hours.notna() & (hours < 0), "hour_negative", "hour")
    work["hour"] = hours.fillna(-1).astype("int64")

    # sensors: finite floats inside a plausible envelope.
    for sensor in SENSORS:
        values = pd.to_numeric(work[sensor], errors="coerce")
        reject(values.isna(), "not_numeric", sensor)
        finite = values.replace([np.inf, -np.inf], np.nan)
        reject(values.notna() & finite.isna(), "not_finite", sensor)
        low, high = SENSOR_RANGES[sensor]
        reject(finite.notna() & ((finite < low) | (finite > high)), "out_of_range", sensor)
        work[sensor] = finite.astype(float)

    # load_factor and failure are optional; absent means "assume nominal".
    if "load_factor" in work.columns:
        loads = pd.to_numeric(work["load_factor"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        reject(loads.isna(), "not_numeric", "load_factor")
        low, high = LOAD_FACTOR_RANGE
        reject(loads.notna() & ((loads < low) | (loads > high)), "out_of_range", "load_factor")
        work["load_factor"] = loads.astype(float)
    else:
        work["load_factor"] = 1.0

    if "failure" in work.columns:
        fails = pd.to_numeric(work["failure"], errors="coerce")
        reject(fails.isna(), "not_numeric", "failure")
        reject(fails.notna() & ~fails.isin([0, 1]), "not_boolean", "failure")
        work["failure"] = fails.fillna(0).astype("int64")
    else:
        work["failure"] = 0

    clean = work[keep].copy()
    clean["load_factor"] = clean["load_factor"].fillna(1.0)

    # Duplicate (machine_id, hour) pairs used to collapse into a lookup dict and
    # then be delivered once per file row, double-counting the same event.
    duplicated = clean.duplicated(subset=["machine_id", "hour"], keep=False)
    duplicates_dropped = 0
    if bool(duplicated.any()):
        if duplicate_policy == "reject":
            for position in np.flatnonzero(duplicated.to_numpy(dtype=bool)):
                row = clean.iloc[int(position)]
                rejected.append(
                    BadRow(int(row["_row_number"]), "duplicate_key",
                           f"{row['machine_id']}@{int(row['hour'])}", _payload(clean, int(position)))
                )
            clean = clean[~duplicated]
        else:
            keep_which = "last" if duplicate_policy == "last" else "first"
            losers = clean.duplicated(subset=["machine_id", "hour"], keep=keep_which)
            duplicates_dropped = int(losers.sum())
            for position in np.flatnonzero(losers.to_numpy(dtype=bool)):
                row = clean.iloc[int(position)]
                rejected.append(
                    BadRow(int(row["_row_number"]), "duplicate_key_dropped",
                           f"{row['machine_id']}@{int(row['hour'])}", _payload(clean, int(position)))
                )
            clean = clean[~losers]

    clean = clean.drop(columns=["_row_number"]).reset_index(drop=True)

    if rows_in and len(rejected) / rows_in > max_bad_fraction:
        raise DataQualityError(
            "too many unusable telemetry rows",
            source=source,
            rows=rows_in,
            rejected=len(rejected),
            limit=max_bad_fraction,
        )
    if clean.empty:
        raise DataQualityError("no usable telemetry rows", source=source, rows=rows_in)

    return ValidationOutcome(frame=clean, rejected=rejected, duplicates_dropped=duplicates_dropped, rows_in=rows_in)


__all__ = [
    "DEFAULT_MAX_BAD_FRACTION",
    "DUPLICATE_POLICIES",
    "LOAD_FACTOR_RANGE",
    "REQUIRED_COLUMNS",
    "SENSOR_RANGES",
    "BadRow",
    "ValidationOutcome",
    "validate_frame",
]
