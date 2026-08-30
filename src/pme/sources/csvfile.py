"""CSV / historian-export source — the adapter you point at real plant data.

Two things about a historian export bite immediately.

First, **row order is not event order**. Exports are commonly written
machine-major (every hour of M000, then every hour of M001), so reading the file
as a delivery order makes event time reset to zero at each machine boundary and
the watermark discards nearly everything behind it. ``order="event_time"`` is
therefore the default: a file is treated as a settled export unless the caller
says otherwise. ``order="file"`` keeps the file's own order for the case where it
genuinely reflects a broker's delivery sequence.

Second, **the file is untrusted**. Columns are coerced, ranged and de-duplicated
by :mod:`pme.sources.validation` before a single event is constructed, and the
rejects come back on the source for the caller to persist.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..errors import SourceError, ValidationError
from ..paths import DEFAULT_MAX_INPUT_BYTES, ensure_readable_file
from .base import MaterializedSource
from .synthetic import events_from_frame, shuffle_delivery
from .validation import DEFAULT_MAX_BAD_FRACTION, REQUIRED_COLUMNS, BadRow, ValidationOutcome, validate_frame

#: Delivery-order policies. ``event_time`` sorts the file into event-time order;
#: ``file`` trusts the file's own row order.
ORDERS = ("event_time", "file")


class CsvTelemetrySource(MaterializedSource):
    """Telemetry from a CSV with columns ``machine_id, hour, <sensors>[, load_factor, failure]``."""

    def __init__(
        self,
        path: str | Path,
        order: str = "event_time",
        out_of_order_delay: int = 0,
        late_fraction: float = 0.0,
        straggler_delay: tuple[int, int] = (24, 96),
        delivery_seed: int = 42,
        duplicate_policy: str = "last",
        max_bad_fraction: float = DEFAULT_MAX_BAD_FRACTION,
        max_bytes: int = DEFAULT_MAX_INPUT_BYTES,
    ) -> None:
        if order not in ORDERS:
            raise ValidationError("unknown delivery order", order=order, allowed=list(ORDERS))
        resolved = ensure_readable_file(path, max_bytes=max_bytes)
        try:
            frame = pd.read_csv(resolved)
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            raise SourceError("telemetry file could not be parsed", path=str(resolved), detail=str(exc)) from exc

        outcome = validate_frame(
            frame, source=resolved.name, max_bad_fraction=max_bad_fraction, duplicate_policy=duplicate_policy
        )
        clean = outcome.frame

        ordered = events_from_frame(clean)
        if order == "file":
            # Keys are unique after validation, so the file order maps one-to-one.
            index = {(e.machine_id, e.event_time): e for e in ordered}
            delivered = [index[(str(r.machine_id), int(r.hour))] for r in clean.itertuples()]
        else:
            delivered = ordered
        delivered = shuffle_delivery(delivered, out_of_order_delay, late_fraction, straggler_delay, delivery_seed)

        super().__init__(delivered, clean)
        self.path = resolved
        self.order = order
        self.validation = outcome
        self.name = f"csv[{resolved.name}]"

    @property
    def rejected_rows(self) -> list[BadRow]:
        """Rows quarantined during validation, for the caller to persist."""
        return list(self.validation.rejected)

    def validation_summary(self) -> dict[str, object]:
        return self.validation.summary()


__all__ = ["ORDERS", "REQUIRED_COLUMNS", "CsvTelemetrySource", "ValidationOutcome"]
