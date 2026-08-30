"""Label construction from a settled historian export.

Labels are the one place the future is allowed to be consulted, so the vectorised
implementation is checked against a deliberately naive forward scan.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pme.core.schema import RISK_HORIZON_H, RUL_CAP_H
from pme.learning.labels import build_labels
from pme.sources.synthetic import _simulate


def _history(rows: list[tuple[str, int, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["machine_id", "hour", "failure"])


def _naive_rul(hours: np.ndarray, failure: np.ndarray, cap: float) -> list[float]:
    """O(n*k) forward scan: the definition the vectorised version has to match."""
    out: list[float] = []
    for i, hour in enumerate(hours):
        gap = cap
        for j in range(i, len(hours)):
            if failure[j] == 1:
                gap = min(float(hours[j] - hour), cap)
                break
        out.append(gap)
    return out


def test_rul_counts_down_to_the_next_failure():
    history = _history([("M1", h, 1 if h == 5 else 0) for h in range(8)])
    labels = build_labels(history)
    assert list(labels["rul_h"][:6]) == [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]


def test_hours_after_the_last_failure_sit_at_the_cap():
    history = _history([("M1", h, 1 if h == 2 else 0) for h in range(6)])
    labels = build_labels(history)
    assert list(labels["rul_h"][3:]) == [RUL_CAP_H] * 3


def test_a_machine_that_never_failed_is_all_cap():
    history = _history([("M1", h, 0) for h in range(10)])
    assert set(build_labels(history)["rul_h"]) == {RUL_CAP_H}


def test_the_risk_target_is_the_rul_horizon_threshold():
    history = _history([("M1", h, 1 if h == 100 else 0) for h in range(120)])
    labels = build_labels(history).sort_values("hour")
    positive = labels[labels["fail_within_h"] == 1]["hour"].to_numpy()
    assert positive.min() == 100 - RISK_HORIZON_H
    assert positive.max() == 100


def test_rul_is_capped_even_for_a_distant_failure():
    history = _history([("M1", h, 1 if h == 900 else 0) for h in range(901)])
    assert build_labels(history)["rul_h"].max() == RUL_CAP_H


def test_machines_are_labelled_independently():
    history = _history(
        [("M1", h, 1 if h == 3 else 0) for h in range(6)] + [("M2", h, 0) for h in range(6)]
    )
    labels = build_labels(history)
    m1 = labels[labels["machine_id"] == "M1"].sort_values("hour")
    m2 = labels[labels["machine_id"] == "M2"]
    assert list(m1["rul_h"][:4]) == [3.0, 2.0, 1.0, 0.0]
    assert set(m2["rul_h"]) == {RUL_CAP_H}


def test_repeated_failures_reset_the_countdown():
    history = _history([("M1", h, 1 if h in (3, 7) else 0) for h in range(9)])
    labels = build_labels(history).sort_values("hour")
    assert list(labels["rul_h"]) == [3.0, 2.0, 1.0, 0.0, 3.0, 2.0, 1.0, 0.0, RUL_CAP_H]


def test_the_vectorised_labels_match_a_naive_forward_scan():
    frame = _simulate(4, 24 * 20, seed=11)
    labels = build_labels(frame)
    for machine, group in frame.sort_values("hour").groupby("machine_id"):
        hours = group["hour"].to_numpy(dtype=np.int64)
        failure = group["failure"].to_numpy(dtype=np.int64)
        expected = _naive_rul(hours, failure, RUL_CAP_H)
        actual = labels[labels["machine_id"] == machine].sort_values("hour")["rul_h"].tolist()
        assert actual == expected


def test_every_history_row_gets_exactly_one_label():
    frame = _simulate(3, 48, seed=5)
    labels = build_labels(frame)
    assert len(labels) == len(frame)
    assert not labels.duplicated(subset=["machine_id", "hour"]).any()


def test_history_without_a_failure_column_is_refused():
    with pytest.raises(ValueError, match="failure"):
        build_labels(pd.DataFrame({"machine_id": ["M1"], "hour": [0]}))


def test_the_horizon_and_cap_are_configurable():
    history = _history([("M1", h, 1 if h == 20 else 0) for h in range(25)])
    labels = build_labels(history, risk_horizon_h=5, rul_cap_h=10.0)
    assert labels["rul_h"].max() == 10.0
    assert int(labels["fail_within_h"].sum()) == 6
