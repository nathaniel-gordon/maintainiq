"""Synthetic fleet source with degradation physics and configurable delivery chaos.

The physics are deliberately simple: machines accumulate wear under a diurnal
load, sensors respond to wear, and a machine fails when wear crosses 1.0 and is
then repaired. A given seed always produces the same telemetry, which is what
makes every replay and restore comparison in the tests meaningful. It is a
simulator, not plant data, and the failure signal is far cleaner than a real
one.

What is new is *delivery*. A historian backfill delivers the same events in
perfect event-time order; a live broker delivers them jittered by up to
``out_of_order_delay`` hours, with a configurable trickle of stragglers that
arrive far too late to be admitted. One telemetry ground truth, two delivery
regimes — which is exactly the situation watermarks exist to handle.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.events import SENSORS, TelemetryEvent
from .base import MaterializedSource


def _simulate(n_machines: int, hours: int, seed: int) -> pd.DataFrame:
    """Hourly telemetry for a fleet: machine_id, hour, sensors, load_factor, failure."""
    rng = np.random.default_rng(seed)
    machine_ids: list[str] = []
    cols: dict[str, list[float]] = {c: [] for c in ("hour", *SENSORS, "load_factor", "failure")}
    for m in range(n_machines):
        wear = float(rng.uniform(0.0, 0.35))
        base_load = rng.uniform(0.5, 1.0)
        mid = f"M{m:03d}"
        for h in range(hours):
            load = float(np.clip(base_load + 0.25 * np.sin(2 * np.pi * h / 24) + rng.normal(0, 0.08), 0.2, 1.4))
            wear += float(rng.gamma(shape=1.6, scale=0.00062) * load**2)
            failed = 0
            if wear >= 1.0:
                failed = 1
                wear = float(rng.uniform(0.02, 0.08))  # repaired immediately after failure
            machine_ids.append(mid)
            cols["hour"].append(h)
            cols["vibration_mm_s"].append(1.8 + 6.5 * wear**1.5 + rng.normal(0, 0.35))
            cols["temp_c"].append(58 + 27 * wear + 6 * load + rng.normal(0, 1.6))
            cols["pressure_bar"].append(5.2 - 1.4 * wear + rng.normal(0, 0.22))
            cols["rpm"].append(1800 - 230 * wear + rng.normal(0, 28))
            cols["oil_quality_pct"].append(float(np.clip(100 - 55 * wear + rng.normal(0, 2.5), 5, 100)))
            cols["load_factor"].append(load)
            cols["failure"].append(failed)
    frame = pd.DataFrame({"machine_id": machine_ids, **cols})
    frame["hour"] = frame["hour"].astype(int)
    frame["failure"] = frame["failure"].astype(int)
    return frame


def events_from_frame(frame: pd.DataFrame) -> list[TelemetryEvent]:
    """Convert a telemetry frame into event objects, ordered by (event_time, machine)."""
    ordered = frame.sort_values(["hour", "machine_id"], kind="stable")
    ids = ordered["machine_id"].to_numpy()
    hours = ordered["hour"].to_numpy(dtype=np.int64)
    sensors = ordered[list(SENSORS)].to_numpy(dtype=float)
    loads = ordered["load_factor"].to_numpy(dtype=float) if "load_factor" in ordered else np.ones(len(ordered))
    fails = ordered["failure"].to_numpy(dtype=int) if "failure" in ordered else np.zeros(len(ordered), dtype=int)
    return [
        TelemetryEvent(str(ids[i]), int(hours[i]), tuple(float(v) for v in sensors[i]), float(loads[i]), int(fails[i]))
        for i in range(len(ordered))
    ]


def shuffle_delivery(
    events: list[TelemetryEvent],
    out_of_order_delay: int,
    late_fraction: float,
    straggler_delay: tuple[int, int],
    seed: int,
) -> list[TelemetryEvent]:
    """Reorder events into a plausible broker delivery order.

    Each event is given a delivery key of ``event_time + jitter``; a stable sort on
    that key yields bounded out-of-orderness. A ``late_fraction`` slice instead
    receives a much larger straggler delay, modelling a machine that lost network
    and flushed its buffer hours later. Stable sorting keeps ties in the original
    (event_time, machine_id) order so the shuffle is fully deterministic.
    """
    if out_of_order_delay == 0 and late_fraction == 0.0:
        return list(events)
    rng = np.random.default_rng(seed)
    n = len(events)
    jitter = rng.integers(0, out_of_order_delay + 1, size=n) if out_of_order_delay > 0 else np.zeros(n, dtype=np.int64)
    if late_fraction > 0.0:
        n_late = round(n * late_fraction)
        if n_late > 0:
            picks = rng.choice(n, size=n_late, replace=False)
            jitter[picks] = rng.integers(straggler_delay[0], straggler_delay[1] + 1, size=n_late)
    keys = np.array([e.event_time for e in events], dtype=np.int64) + jitter
    order = np.argsort(keys, kind="stable")
    return [events[i] for i in order]


class SyntheticFleetSource(MaterializedSource):
    """Seeded fleet telemetry with a selectable delivery regime.

    ``out_of_order_delay=0, late_fraction=0.0`` is the historian backfill used for
    training; the live-stream defaults model a real broker.
    """

    def __init__(
        self,
        n_machines: int = 40,
        hours: int = 24 * 90,
        seed: int = 42,
        out_of_order_delay: int = 0,
        late_fraction: float = 0.0,
        straggler_delay: tuple[int, int] = (24, 96),
        delivery_seed: int | None = None,
    ) -> None:
        frame = _simulate(n_machines, hours, seed)
        ordered = events_from_frame(frame)
        delivered = shuffle_delivery(
            ordered, out_of_order_delay, late_fraction, straggler_delay,
            seed if delivery_seed is None else delivery_seed,
        )
        super().__init__(delivered, frame)
        self.name = f"synthetic[{n_machines}x{hours}h,seed={seed},jitter={out_of_order_delay}h,late={late_fraction}]"
        self.n_machines = n_machines
        self.hours = hours
