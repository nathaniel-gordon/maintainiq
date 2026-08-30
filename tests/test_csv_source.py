"""CSV source: delivery-order policy, path limits and stream fingerprint."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pme.errors import PathError, ValidationError
from pme.sources.csvfile import CsvTelemetrySource
from pme.sources.synthetic import _simulate


def _frame(n_machines: int = 3, hours: int = 40) -> pd.DataFrame:
    return _simulate(n_machines, hours, seed=7)


def _machine_major_csv(tmp_path, n_machines: int = 3, hours: int = 40):
    frame = _frame(n_machines, hours).sort_values(["machine_id", "hour"], kind="stable")
    path = tmp_path / "tele.csv"
    frame.to_csv(path, index=False)
    return path


def test_machine_major_file_is_delivered_in_event_time_order(tmp_path):
    path = _machine_major_csv(tmp_path)
    source = CsvTelemetrySource(path)
    times = [e.event_time for e in source.emit()]
    assert times == sorted(times)


def test_file_order_is_available_but_not_the_default(tmp_path):
    path = _machine_major_csv(tmp_path)
    source = CsvTelemetrySource(path, order="file")
    times = [e.event_time for e in source.emit()]
    assert times != sorted(times)


def test_unknown_order_is_rejected(tmp_path):
    with pytest.raises(ValidationError):
        CsvTelemetrySource(_machine_major_csv(tmp_path), order="whatever")


def test_missing_file_is_a_path_error(tmp_path):
    with pytest.raises(PathError):
        CsvTelemetrySource(tmp_path / "nope.csv")


def test_oversized_file_is_refused(tmp_path):
    with pytest.raises(PathError):
        CsvTelemetrySource(_machine_major_csv(tmp_path), max_bytes=32)


def test_csv_source_exposes_its_quarantine(tmp_path):
    frame = _frame(2, 30)
    frame.loc[7, "vibration_mm_s"] = np.nan
    path = tmp_path / "dirty.csv"
    frame.to_csv(path, index=False)
    source = CsvTelemetrySource(path)
    assert len(source.rejected_rows) == 1
    assert source.validation_summary()["reject_codes"] == {"not_numeric": 1}
    assert len(source) == len(frame) - 1


def test_fingerprint_tracks_content_and_delivery_order(tmp_path):
    path = _machine_major_csv(tmp_path)
    ordered = CsvTelemetrySource(path).fingerprint()
    assert CsvTelemetrySource(path).fingerprint() == ordered
    assert CsvTelemetrySource(path, order="file").fingerprint() != ordered
    assert CsvTelemetrySource(path, out_of_order_delay=6).fingerprint() != ordered
