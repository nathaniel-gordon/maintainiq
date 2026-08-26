"""The feature contract shared by the streaming path and the offline trainer.

There is exactly one definition of the feature vector. The trainer builds its
design matrix by *replaying the stream through the same operator* the daemon
uses, so train/serve skew is structurally impossible rather than merely tested.
"""
from __future__ import annotations

from .events import SENSORS

#: Length of the sliding event-time window used for rolling statistics (hours).
WINDOW_SIZE_H: int = 24

#: A sliding window must hold at least this many samples before it emits. Gaps
#: appear when events are dropped as too-late, so a strict "24 of 24" rule would
#: silently stall a machine forever after one drop.
MIN_WINDOW_SAMPLES: int = 20

#: Classifier target: does this machine fail within the next N hours?
RISK_HORIZON_H: int = 72

#: RUL regression cap. Rows at the cap are "no failure in sight" and are excluded
#: from the regressor's training/eval sets, exactly as in the validated baseline.
RUL_CAP_H: float = 500.0


def feature_names() -> tuple[str, ...]:
    """Column names of :attr:`pme.core.events.FeatureRecord.values`, in order."""
    cols: list[str] = []
    for sensor in SENSORS:
        cols.append(sensor)
        cols.append(f"{sensor}_mean{WINDOW_SIZE_H}")
        cols.append(f"{sensor}_std{WINDOW_SIZE_H}")
        cols.append(f"{sensor}_slope{WINDOW_SIZE_H}")
    cols.append("load_factor")
    return tuple(cols)


FEATURE_NAMES: tuple[str, ...] = feature_names()
