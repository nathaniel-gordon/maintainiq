"""O(1) incremental event-time window aggregate (mean / sample std / OLS slope).

The validated baseline computed these with ``pandas.rolling`` over the whole
history — an O(n * window) re-scan that a daemon on an unbounded stream cannot
afford. This state object produces *numerically the same* statistics from running
sums that are updated once per admitted event and once per evicted event.

Given a window holding pairs ``(t_i, y_i)``:

* ``mean  = S / n``                             with ``S  = sum(y)``
* ``var   = (Q - S^2/n) / (n - 1)``             with ``Q  = sum(y^2)``  (ddof=1)
* ``slope = (P - Ts*S/n) / (Tq - Ts^2/n)``      with ``P = sum(t*y)``,
  ``Ts = sum(t)``, ``Tq = sum(t^2)`` — the ordinary-least-squares slope of y on t.

The slope formula is shift-invariant in ``t``, so timestamps are stored relative
to a per-machine origin. That matters: absolute hour indices grow without bound
on a long-running daemon and ``P`` would lose precision to cancellation in
``P - Ts*S/n``. The origin is rebased (an O(window) refresh) whenever the relative
clock exceeds :data:`REBASE_LIMIT`.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Any

from .events import SENSORS

#: Rebase the per-machine time origin once relative timestamps exceed this.
#: 2**20 hours is ~120 years of stream, so rebasing is rare in practice but the
#: code path is exercised by a dedicated test.
REBASE_LIMIT: int = 1 << 20

_N_SENSORS = len(SENSORS)


class MachineWindowState:
    """Per-machine sliding-window aggregate over all sensor channels at once.

    One state object per machine holds a single deque of admitted events plus
    ``3 * n_sensors + 3`` running scalars. Adding an event is O(n_sensors);
    evicting is O(n_sensors) per evicted event, amortised O(1) per event overall.
    """

    __slots__ = ("_times", "_rows", "_loads", "_s", "_q", "_p", "_ts", "_tq", "_load_sum", "_origin")

    def __init__(self) -> None:
        self._times: deque[int] = deque()
        self._rows: deque[tuple[float, ...]] = deque()
        self._loads: deque[float] = deque()
        self._s = [0.0] * _N_SENSORS
        self._q = [0.0] * _N_SENSORS
        self._p = [0.0] * _N_SENSORS
        self._ts = 0.0
        self._tq = 0.0
        self._load_sum = 0.0
        self._origin: int | None = None

    # ---------------------------------------------------------------- mutation

    def admit(self, event_time: int, readings: tuple[float, ...], load_factor: float) -> None:
        """Fold one event into the aggregate. Caller guarantees event-time order."""
        if self._origin is None:
            self._origin = event_time
        elif event_time - self._origin > REBASE_LIMIT:
            self._rebase(event_time)
        rel = float(event_time - self._origin)
        self._times.append(event_time)
        self._rows.append(readings)
        self._loads.append(load_factor)
        for k in range(_N_SENSORS):
            v = readings[k]
            self._s[k] += v
            self._q[k] += v * v
            self._p[k] += rel * v
        self._ts += rel
        self._tq += rel * rel
        self._load_sum += load_factor

    def evict_before(self, start: int) -> None:
        """Drop every buffered event with ``event_time < start`` (window scope)."""
        while self._times and self._times[0] < start:
            t = self._times.popleft()
            row = self._rows.popleft()
            load = self._loads.popleft()
            rel = float(t - self._origin)  # type: ignore[operator]
            for k in range(_N_SENSORS):
                v = row[k]
                self._s[k] -= v
                self._q[k] -= v * v
                self._p[k] -= rel * v
            self._ts -= rel
            self._tq -= rel * rel
            self._load_sum -= load

    def _rebase(self, new_origin: int) -> None:
        """Recompute the running sums against a fresh time origin.

        Called only when relative timestamps grow large enough to threaten the
        precision of the slope numerator. O(window), amortised to nothing.
        """
        self._origin = new_origin
        self._p = [0.0] * _N_SENSORS
        self._ts = 0.0
        self._tq = 0.0
        for t, row in zip(self._times, self._rows):
            rel = float(t - new_origin)
            for k in range(_N_SENSORS):
                self._p[k] += rel * row[k]
            self._ts += rel
            self._tq += rel * rel

    # -------------------------------------------------------------- inspection

    @property
    def samples(self) -> int:
        return len(self._times)

    @property
    def latest_time(self) -> int:
        return self._times[-1]

    def feature_vector(self) -> tuple[float, ...]:
        """``(value, mean, std, slope)`` per sensor, then the current load factor.

        Layout matches :data:`pme.core.schema.FEATURE_NAMES`.
        """
        n = len(self._times)
        if n == 0:
            raise ValueError("cannot build features from an empty window")
        latest = self._rows[-1]
        inv_n = 1.0 / n
        t_var = self._tq - self._ts * self._ts * inv_n
        out: list[float] = []
        for k in range(_N_SENSORS):
            s = self._s[k]
            mean = s * inv_n
            if n > 1:
                num = self._q[k] - s * s * inv_n
                # Clamp: catastrophic cancellation on a near-constant channel can
                # push the numerator a few ULP below zero.
                var = num / (n - 1) if num > 0.0 else 0.0
                std = math.sqrt(var)
            else:
                std = 0.0
            slope = (self._p[k] - self._ts * s * inv_n) / t_var if t_var > 0.0 else 0.0
            out.extend((latest[k], mean, std, slope))
        out.append(self._loads[-1])
        return tuple(out)

    # ------------------------------------------------------------- persistence

    def state(self) -> dict[str, Any]:
        """Serialise the *exact* running sums, not just the buffered events.

        Recomputing the sums from the deque on restore would produce values a few
        ULP away from the uninterrupted run and break byte-identical replay, so
        the accumulators themselves are part of the checkpoint.
        """
        return {
            "times": list(self._times),
            "rows": [list(r) for r in self._rows],
            "loads": list(self._loads),
            "s": list(self._s),
            "q": list(self._q),
            "p": list(self._p),
            "ts": self._ts,
            "tq": self._tq,
            "load_sum": self._load_sum,
            "origin": self._origin,
        }

    @staticmethod
    def from_state(state: dict[str, Any]) -> "MachineWindowState":
        obj = MachineWindowState()
        obj._times = deque(int(t) for t in state["times"])
        obj._rows = deque(tuple(float(v) for v in r) for r in state["rows"])
        obj._loads = deque(float(v) for v in state["loads"])
        obj._s = [float(v) for v in state["s"]]
        obj._q = [float(v) for v in state["q"]]
        obj._p = [float(v) for v in state["p"]]
        obj._ts = float(state["ts"])
        obj._tq = float(state["tq"])
        obj._load_sum = float(state["load_sum"])
        obj._origin = None if state["origin"] is None else int(state["origin"])
        return obj
