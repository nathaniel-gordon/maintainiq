"""Telemetry validation: coercion, ranges, duplicates and the quarantine."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pme.errors import DataQualityError
from pme.sources.synthetic import _simulate
from pme.sources.validation import validate_frame


def _frame(n_machines: int = 3, hours: int = 40) -> pd.DataFrame:
    return _simulate(n_machines, hours, seed=7)


def test_clean_telemetry_passes_untouched():
    frame = _frame()
    outcome = validate_frame(frame, source="t.csv")
    assert outcome.rows_out == len(frame)
    assert outcome.rejected == []


def test_missing_columns_are_fatal():
    with pytest.raises(DataQualityError):
        validate_frame(_frame().drop(columns=["temp_c"]), source="t.csv")


def test_a_nan_cell_is_quarantined_not_raised():
    frame = _frame()
    frame.loc[10, "temp_c"] = np.nan
    outcome = validate_frame(frame, source="t.csv")
    assert outcome.rows_out == len(frame) - 1
    assert [r.code for r in outcome.rejected] == ["not_numeric"]
    assert outcome.rejected[0].row_number == 10
    assert outcome.rejected[0].detail == "temp_c"


def test_infinities_and_sentinels_are_rejected():
    frame = _frame()
    frame.loc[3, "pressure_bar"] = np.inf
    frame.loc[4, "rpm"] = -9999.0
    outcome = validate_frame(frame, source="t.csv")
    assert sorted(r.code for r in outcome.rejected) == ["not_finite", "out_of_range"]


def test_negative_and_fractional_hours_are_rejected():
    frame = _frame()
    frame["hour"] = frame["hour"].astype(float)
    frame.loc[1, "hour"] = -5.0
    frame.loc[2, "hour"] = 3.5
    outcome = validate_frame(frame, source="t.csv")
    assert sorted(r.code for r in outcome.rejected) == ["hour_negative", "hour_not_integer"]


def test_duplicate_keys_collapse_with_a_counter():
    frame = _frame(2, 10)
    doubled = pd.concat([frame, frame.iloc[:5]], ignore_index=True)
    outcome = validate_frame(doubled, source="t.csv", max_bad_fraction=1.0)
    assert outcome.duplicates_dropped == 5
    assert outcome.rows_out == len(frame)
    assert outcome.frame.duplicated(subset=["machine_id", "hour"]).sum() == 0


def test_duplicate_policy_reject_drops_every_copy():
    frame = _frame(2, 10)
    doubled = pd.concat([frame, frame.iloc[:5]], ignore_index=True)
    outcome = validate_frame(doubled, source="t.csv", max_bad_fraction=1.0, duplicate_policy="reject")
    assert outcome.rows_out == len(frame) - 5


def test_too_much_damage_is_fatal():
    frame = _frame(2, 20)
    frame.loc[: len(frame) // 2, "temp_c"] = np.nan
    with pytest.raises(DataQualityError):
        validate_frame(frame, source="t.csv")
