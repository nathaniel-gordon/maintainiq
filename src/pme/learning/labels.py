"""Backward-looking labels built from a settled historian export.

Labels need the future (time to the *next* failure) and therefore cannot come
from the live stream — only from history. Keeping them here, apart from the
feature path, is what makes "features never see the future" a structural claim
rather than a hope.

The baseline computed RUL with a per-row scan; this is the vectorised equivalent
(``searchsorted`` over each machine's failure indices) and produces identical
values while being O(n log k) instead of O(n * k).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.schema import RISK_HORIZON_H, RUL_CAP_H


def build_labels(history: pd.DataFrame, risk_horizon_h: int = RISK_HORIZON_H,
                 rul_cap_h: float = RUL_CAP_H) -> pd.DataFrame:
    """Return ``machine_id, hour, rul_h, fail_within_h`` for every historical row.

    ``rul_h`` is the hours until the next failure event at or after this hour,
    capped at ``rul_cap_h``; ``fail_within_h`` is the binary classifier target.
    """
    if "failure" not in history.columns:
        raise ValueError("history must carry a binary 'failure' column to build labels")
    frames = []
    for mid, group in history.sort_values("hour", kind="stable").groupby("machine_id", sort=True):
        hours = group["hour"].to_numpy(dtype=np.int64)
        failure = group["failure"].to_numpy(dtype=np.int64)
        fail_pos = np.flatnonzero(failure == 1)
        rul = np.full(len(hours), float(rul_cap_h))
        if len(fail_pos):
            nxt = np.searchsorted(fail_pos, np.arange(len(hours)), side="left")
            has_next = nxt < len(fail_pos)
            idx = fail_pos[np.clip(nxt, 0, len(fail_pos) - 1)]
            gap = (hours[idx] - hours).astype(float)
            rul = np.where(has_next, np.minimum(gap, rul_cap_h), rul_cap_h)
        frames.append(
            pd.DataFrame(
                {
                    "machine_id": mid,
                    "hour": hours,
                    "rul_h": rul,
                    "fail_within_h": (rul <= risk_horizon_h).astype(int),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)
