"""Maintenance-schedule sink: the cost-based scheduler, preserved and made streaming.

The economics are carried over from the validated engine unchanged — a service is
only booked when ``risk x unplanned_cost > planned_cost``, and crews are a hard
capacity constraint. What changed is the input: instead of a batch "final state"
frame, the sink tracks the newest scored window per machine as the stream flows
and re-optimises whenever asked. A plant can therefore pull an up-to-date work
order at any point in the stream, not only after the batch job finishes.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.events import ScoredRecord
from .base import Sink


@dataclass(frozen=True, slots=True)
class CostModel:
    """Plant economics. Frozen so a schedule can never be produced under mutated costs."""

    planned_maintenance: float = 1_200.0  # cost of a planned service
    unplanned_failure: float = 8_500.0  # parts + emergency labour
    downtime_per_hour: float = 350.0
    unplanned_downtime_h: float = 18.0
    planned_downtime_h: float = 4.0

    def expected_saving(self, risk_prob: float) -> float:
        """Expected net saving of servicing now versus running to probable failure."""
        unplanned = self.unplanned_failure + self.downtime_per_hour * self.unplanned_downtime_h
        planned = self.planned_maintenance + self.downtime_per_hour * self.planned_downtime_h
        return risk_prob * unplanned - planned

    @property
    def breakeven_risk(self) -> float:
        """Risk probability at which servicing becomes cost-neutral."""
        unplanned = self.unplanned_failure + self.downtime_per_hour * self.unplanned_downtime_h
        planned = self.planned_maintenance + self.downtime_per_hour * self.planned_downtime_h
        return planned / unplanned


@dataclass(frozen=True, slots=True)
class ServiceOrder:
    machine_id: str
    day: int
    event_time: int
    risk_prob: float
    rul_est_h: float
    expected_saving: float
    reason: str


#: Priority bonus applied to an anomalous machine so a novel failure signature can
#: earn a crew slot even when its modelled risk is still low.
ANOMALY_PRIORITY_BONUS = 500.0


def build_schedule(
    latest: dict[str, ScoredRecord],
    costs: CostModel | None = None,
    crews_per_day: int = 3,
    horizon_days: int = 7,
) -> list[ServiceOrder]:
    """Assign cost-positive (or anomalous) machines to crew slots, best first.

    Ordering is by ``(-priority, machine_id)`` so the plan is reproducible; the
    baseline relied on an unstable pandas sort for ties.
    """
    costs = costs or CostModel()
    candidates: list[tuple[float, str, ScoredRecord, float]] = []
    for machine_id, rec in latest.items():
        saving = costs.expected_saving(rec.risk_prob)
        if saving <= 0.0 and rec.anomaly != 1:
            continue
        priority = saving + rec.anomaly * ANOMALY_PRIORITY_BONUS
        candidates.append((priority, machine_id, rec, saving))
    candidates.sort(key=lambda c: (-c[0], c[1]))

    orders: list[ServiceOrder] = []
    for slot, (_, machine_id, rec, saving) in enumerate(candidates):
        day = slot // crews_per_day + 1
        if day > horizon_days:
            break
        reasons = []
        if saving > 0.0:
            reasons.append(f"risk {rec.risk_prob:.0%} makes service cost-positive")
        if rec.anomaly == 1:
            reasons.append("anomalous sensor signature")
        orders.append(
            ServiceOrder(
                machine_id=machine_id,
                day=day,
                event_time=rec.event_time,
                risk_prob=round(rec.risk_prob, 4),
                rul_est_h=round(rec.rul_est_h, 1),
                expected_saving=round(max(saving, 0.0), 2),
                reason="; ".join(reasons),
            )
        )
    return orders


class MaintenanceScheduleSink(Sink):
    """Tracks each machine's newest scored window and emits a crew schedule."""

    name = "schedule"
    accepts = (ScoredRecord,)

    def __init__(
        self,
        path: str | Path,
        costs: CostModel | None = None,
        crews_per_day: int = 3,
        horizon_days: int = 7,
    ) -> None:
        self.path = Path(path)
        self.costs = costs or CostModel()
        self.crews_per_day = crews_per_day
        self.horizon_days = horizon_days
        self.latest: dict[str, ScoredRecord] = {}
        self.orders: list[ServiceOrder] = []
        self.updates = 0

    def write(self, record: ScoredRecord) -> None:
        current = self.latest.get(record.key)
        # Guard against a late window overwriting a newer one after restore.
        if current is None or record.event_time >= current.event_time:
            self.latest[record.key] = record
        self.updates += 1

    def current_schedule(self) -> list[ServiceOrder]:
        """Re-optimise from the newest state seen so far — callable mid-stream."""
        return build_schedule(self.latest, self.costs, self.crews_per_day, self.horizon_days)

    def close(self) -> None:
        self.orders = self.current_schedule()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["machine_id", "day", "event_time", "risk_prob", "rul_est_h",
                             "expected_saving", "reason"])
            for order in self.orders:
                writer.writerow([order.machine_id, order.day, order.event_time, order.risk_prob,
                                 order.rul_est_h, order.expected_saving, order.reason])

    def summary(self) -> dict[str, Any]:
        return {
            "machines_tracked": len(self.latest),
            "service_orders": len(self.orders),
            "total_expected_saving": round(sum(o.expected_saving for o in self.orders), 2),
            "breakeven_risk": round(self.costs.breakeven_risk, 4),
            "path": str(self.path),
        }

    def state(self) -> dict[str, Any]:
        return {
            "updates": self.updates,
            "latest": {k: [v.event_time, v.risk_prob, v.rul_est_h, v.anomaly]
                       for k, v in sorted(self.latest.items())},
        }

    def restore(self, state: dict[str, Any]) -> None:
        self.updates = int(state["updates"])
        self.latest = {
            k: ScoredRecord(k, int(v[0]), float(v[1]), float(v[2]), int(v[3]))
            for k, v in state["latest"].items()
        }
