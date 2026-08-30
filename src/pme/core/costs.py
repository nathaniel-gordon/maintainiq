"""Plant economics.

Lives in ``core`` with the other pure domain objects because two very different
parts of the system need it: the scheduler that ranks work orders, and the
configuration that has to record which economics a run was carried out under. A
resume under different costs would produce a different schedule from the same
stream, so the numbers belong in the checkpoint's semantic contract.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CostModel:
    """Frozen so a schedule can never be produced under mutated costs."""

    planned_maintenance: float = 1_200.0  # cost of a planned service
    unplanned_failure: float = 8_500.0  # parts + emergency labour
    downtime_per_hour: float = 350.0
    unplanned_downtime_h: float = 18.0
    planned_downtime_h: float = 4.0

    def expected_saving(self, risk_prob: float) -> float:
        """Expected net saving of servicing now versus running to probable failure."""
        return risk_prob * self._unplanned_total - self._planned_total

    @property
    def breakeven_risk(self) -> float:
        """Risk probability at which servicing becomes cost-neutral."""
        return self._planned_total / self._unplanned_total

    @property
    def _unplanned_total(self) -> float:
        return self.unplanned_failure + self.downtime_per_hour * self.unplanned_downtime_h

    @property
    def _planned_total(self) -> float:
        return self.planned_maintenance + self.downtime_per_hour * self.planned_downtime_h

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> CostModel:
        return CostModel(**{k: float(v) for k, v in payload.items()})


__all__ = ["CostModel"]
