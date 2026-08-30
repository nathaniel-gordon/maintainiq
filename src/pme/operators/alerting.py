"""Alerting operator: turns scored windows into de-duplicated maintenance alerts.

The batch engine could only ever report the fleet's *final* state. A daemon sees
every window close, so it can alert the hour a machine crosses a threshold — but
it would also emit one alert per hour per machine forever without suppression.

The cooldown is measured in **event time**, not wall-clock: replaying a month of
history in ten seconds must produce exactly the alerts the live run would have.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..core.events import Alert, ScoredRecord
from .base import Operator

#: Default risk bands, checked highest-first. Overridable through
#: :class:`~pme.runtime.config.StreamConfig` so a run records the bands it used.
CRITICAL_RISK = 0.5
WARNING_RISK = 0.2


class AlertOperator(Operator):
    """Threshold + anomaly alerting with a per-machine, per-severity cooldown."""

    name = "alerting"

    def __init__(self, cooldown_h: int = 24, critical_risk: float = CRITICAL_RISK,
                 warning_risk: float = WARNING_RISK) -> None:
        if cooldown_h < 0:
            raise ValueError("cooldown_h must be >= 0")
        if not 0.0 <= warning_risk <= critical_risk <= 1.0:
            raise ValueError("risk bands must satisfy 0 <= warning <= critical <= 1")
        self.cooldown_h = cooldown_h
        self.critical_risk = critical_risk
        self.warning_risk = warning_risk
        self._last: dict[str, int] = {}
        self.emitted = 0
        self.suppressed = 0

    def process(self, record: ScoredRecord) -> Sequence[Alert]:
        severity, reason = self._classify(record)
        if severity is None:
            return ()
        gate = f"{record.key}|{severity}"
        previous = self._last.get(gate)
        if previous is not None and record.event_time - previous < self.cooldown_h:
            self.suppressed += 1
            return ()
        self._last[gate] = record.event_time
        self.emitted += 1
        return (
            Alert(
                key=record.key,
                event_time=record.event_time,
                severity=severity,
                risk_prob=record.risk_prob,
                rul_est_h=record.rul_est_h,
                anomaly=record.anomaly,
                reason=reason,
            ),
        )

    def _classify(self, record: ScoredRecord) -> tuple[str | None, str]:
        if record.risk_prob > self.critical_risk:
            return "CRITICAL", f"P(fail<=72h)={record.risk_prob:.2f}, RUL~{record.rul_est_h:.0f}h"
        if record.risk_prob > self.warning_risk:
            return "WARNING", f"P(fail<=72h)={record.risk_prob:.2f}, RUL~{record.rul_est_h:.0f}h"
        if record.anomaly == 1:
            return "ANOMALY", f"novel sensor signature, RUL~{record.rul_est_h:.0f}h"
        return None, ""

    def state(self) -> dict[str, Any]:
        return {
            "cooldown_h": self.cooldown_h,
            "critical_risk": self.critical_risk,
            "warning_risk": self.warning_risk,
            "last": dict(sorted(self._last.items())),
            "emitted": self.emitted,
            "suppressed": self.suppressed,
        }

    def restore(self, state: dict[str, Any]) -> None:
        self.cooldown_h = int(state["cooldown_h"])
        self.critical_risk = float(state["critical_risk"])
        self.warning_risk = float(state["warning_risk"])
        self._last = {k: int(v) for k, v in state["last"].items()}
        self.emitted = int(state["emitted"])
        self.suppressed = int(state["suppressed"])
