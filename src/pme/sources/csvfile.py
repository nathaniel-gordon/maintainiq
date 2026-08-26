"""CSV / historian-export source — the adapter you point at real plant data.

Delivery order is taken from the file as written. Historian exports are commonly
sorted by ingest time rather than event time, so this source does **not** sort:
letting genuinely out-of-order rows through is the honest behaviour, and the
engine's watermark and lateness policy are what make it safe.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..core.events import SENSORS
from .base import MaterializedSource
from .synthetic import events_from_frame, shuffle_delivery

REQUIRED_COLUMNS: tuple[str, ...] = ("machine_id", "hour", *SENSORS)


class CsvTelemetrySource(MaterializedSource):
    """Telemetry from a CSV with columns ``machine_id, hour, <sensors>[, load_factor, failure]``."""

    def __init__(
        self,
        path: str | Path,
        out_of_order_delay: int = 0,
        late_fraction: float = 0.0,
        straggler_delay: tuple[int, int] = (24, 96),
        delivery_seed: int = 42,
        preserve_file_order: bool = True,
    ) -> None:
        path = Path(path)
        frame = pd.read_csv(path)
        missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(f"{path.name} is missing required columns: {missing}")
        frame["hour"] = frame["hour"].astype(int)
        if "load_factor" not in frame:
            frame["load_factor"] = 1.0
        if "failure" not in frame:
            frame["failure"] = 0
        frame["failure"] = frame["failure"].astype(int)

        ordered = events_from_frame(frame)
        if preserve_file_order:
            # Rebuild delivery order from the file's own row order.
            index = {(e.machine_id, e.event_time): e for e in ordered}
            delivered = [index[(str(r.machine_id), int(r.hour))] for r in frame.itertuples()]
        else:
            delivered = ordered
        delivered = shuffle_delivery(delivered, out_of_order_delay, late_fraction, straggler_delay, delivery_seed)
        super().__init__(delivered, frame)
        self.name = f"csv[{path.name}]"
