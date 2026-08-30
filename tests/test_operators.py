"""The three operators: the emission gate, batching invariance and the cooldown."""
from __future__ import annotations

import numpy as np
import pytest

from pme.core.events import FeatureRecord, ScoredRecord, SlidingTrigger, TelemetryEvent, Window
from pme.core.schema import FEATURE_NAMES
from pme.operators.alerting import AlertOperator
from pme.operators.features import FeatureOperator
from pme.operators.scoring import ScoringOperator

SENSOR_COUNT = 5
WINDOW = 24


class StubModels:
    """Row-independent stand-in: risk is the first feature, RUL the second."""

    min_window_samples = 20

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def score_batch(self, x: np.ndarray):
        self.batch_sizes.append(len(x))
        risk = np.clip(x[:, 0] / 100.0, 0.0, 1.0)
        rul = np.clip(x[:, 1], 0.0, 500.0)
        anomaly = (x[:, 0] > 90).astype(int)
        return risk, rul, anomaly


def _trigger(machine: str, t: int, value: float = 1.0) -> SlidingTrigger:
    event = TelemetryEvent(machine, t, tuple(value + i for i in range(SENSOR_COUNT)), 0.8, 0)
    return SlidingTrigger(machine, Window(t - WINDOW + 1, t + 1, "sliding"), event, t)


def _feature(machine: str, t: int, first: float = 10.0, second: float = 50.0) -> FeatureRecord:
    values = [0.0] * len(FEATURE_NAMES)
    values[0], values[1] = first, second
    return FeatureRecord(machine, t, Window(t - WINDOW + 1, t + 1, "sliding"), tuple(values), WINDOW)


def _scored(machine: str, t: int, risk: float = 0.0, anomaly: int = 0) -> ScoredRecord:
    return ScoredRecord(machine, t, risk, 40.0, anomaly)


# --------------------------------------------------------------------- features


def test_no_features_are_emitted_before_the_window_gate_is_met():
    operator = FeatureOperator(min_samples=5)
    emitted = [operator.process(_trigger("M1", t)) for t in range(4)]
    assert all(out == () for out in emitted)
    assert operator.warmup_skipped == 4
    assert operator.emitted == 0


def test_the_gate_opens_on_the_sample_that_reaches_it():
    operator = FeatureOperator(min_samples=5)
    for t in range(4):
        operator.process(_trigger("M1", t))
    out = operator.process(_trigger("M1", 4))
    assert len(out) == 1
    assert out[0].samples == 5
    assert operator.emitted == 1


def test_a_feature_vector_matches_the_declared_column_contract():
    operator = FeatureOperator(min_samples=1)
    record = operator.process(_trigger("M1", 0))[0]
    assert len(record.values) == len(FEATURE_NAMES)


def test_machines_keep_separate_window_state():
    operator = FeatureOperator(min_samples=2)
    operator.process(_trigger("M1", 0))
    operator.process(_trigger("M2", 0))
    assert operator.process(_trigger("M1", 1))
    assert operator.emitted == 1


def test_events_older_than_the_window_are_evicted():
    operator = FeatureOperator(min_samples=1)
    for t in range(WINDOW * 2):
        record = operator.process(_trigger("M1", t))[0]
    assert record.samples == WINDOW


def test_a_gap_in_the_stream_reopens_the_warmup_gate():
    """Dropped-late events leave holes, and the gate is what stops a thin window."""
    operator = FeatureOperator(min_samples=10)
    for t in range(WINDOW):
        operator.process(_trigger("M1", t))
    assert operator.emitted > 0
    emitted_before = operator.emitted
    out = operator.process(_trigger("M1", 500))
    assert out == ()
    assert operator.emitted == emitted_before


def test_feature_state_round_trips():
    operator = FeatureOperator(min_samples=3)
    for t in range(10):
        operator.process(_trigger("M1", t))
        operator.process(_trigger("M2", t))
    snapshot = operator.state()
    restored = FeatureOperator(min_samples=99)
    restored.restore(snapshot)
    assert restored.state() == snapshot
    assert restored.min_samples == 3
    assert restored.process(_trigger("M1", 10))[0].values == operator.process(_trigger("M1", 10))[0].values


# ---------------------------------------------------------------------- scoring


@pytest.mark.parametrize("batch_size", [1, 2, 7, 64, 1000])
def test_the_batch_size_changes_grouping_but_not_results(batch_size):
    """Micro-batching is only safe because the estimators are row-independent."""
    features = [_feature("M1", t, first=float(t), second=float(t * 2)) for t in range(50)]
    operator = ScoringOperator(StubModels(), batch_size=batch_size)
    out: list[ScoredRecord] = []
    for record in features:
        out.extend(operator.process(record))
    out.extend(operator.flush())

    reference = ScoringOperator(StubModels(), batch_size=1)
    expected: list[ScoredRecord] = []
    for record in features:
        expected.extend(reference.process(record))
    expected.extend(reference.flush())
    assert out == expected


def test_records_leave_the_buffer_in_arrival_order():
    operator = ScoringOperator(StubModels(), batch_size=4)
    out: list[ScoredRecord] = []
    for t in range(9):
        out.extend(operator.process(_feature("M1", t)))
    out.extend(operator.flush())
    assert [r.event_time for r in out] == list(range(9))


def test_flushing_an_empty_buffer_is_a_no_op():
    operator = ScoringOperator(StubModels(), batch_size=4)
    assert operator.flush() == ()
    assert operator.batches == 0


def test_a_zero_batch_size_is_refused():
    with pytest.raises(ValueError, match="batch_size"):
        ScoringOperator(StubModels(), batch_size=0)


def test_scoring_state_carries_the_pending_buffer():
    operator = ScoringOperator(StubModels(), batch_size=10)
    for t in range(4):
        operator.process(_feature("M1", t))
    snapshot = operator.state()
    assert len(snapshot["pending"]) == 4

    restored = ScoringOperator(StubModels(), batch_size=1)
    restored.restore(snapshot)
    assert restored.state() == snapshot
    assert [r.event_time for r in restored.flush()] == [0, 1, 2, 3]


# --------------------------------------------------------------------- alerting


@pytest.mark.parametrize(
    ("risk", "anomaly", "severity"),
    [
        (0.9, 0, "CRITICAL"),
        (0.51, 0, "CRITICAL"),
        (0.3, 0, "WARNING"),
        (0.21, 0, "WARNING"),
        (0.1, 1, "ANOMALY"),
        (0.1, 0, None),
        (0.5, 0, "WARNING"),
        (0.2, 0, None),
    ],
)
def test_risk_bands_are_checked_highest_first(risk, anomaly, severity):
    operator = AlertOperator(cooldown_h=0)
    out = operator.process(_scored("M1", 10, risk=risk, anomaly=anomaly))
    assert [a.severity for a in out] == ([severity] if severity else [])


def test_a_repeat_inside_the_cooldown_is_suppressed():
    operator = AlertOperator(cooldown_h=24)
    assert operator.process(_scored("M1", 0, risk=0.9))
    assert operator.process(_scored("M1", 23, risk=0.9)) == ()
    assert operator.suppressed == 1
    assert operator.process(_scored("M1", 24, risk=0.9))
    assert operator.emitted == 2


def test_the_cooldown_is_measured_in_event_time_not_wall_clock():
    """A month replayed in a second must alert exactly as the live run would."""
    operator = AlertOperator(cooldown_h=24)
    times = [t for t in range(0, 240) if operator.process(_scored("M1", t, risk=0.9))]
    assert times == list(range(0, 240, 24))


def test_the_cooldown_is_per_machine_and_per_severity():
    operator = AlertOperator(cooldown_h=100)
    assert operator.process(_scored("M1", 0, risk=0.9))
    assert operator.process(_scored("M2", 0, risk=0.9))
    assert operator.process(_scored("M1", 1, risk=0.3))
    assert operator.emitted == 3


def test_impossible_alert_settings_are_refused():
    with pytest.raises(ValueError, match="cooldown_h"):
        AlertOperator(cooldown_h=-1)
    with pytest.raises(ValueError, match="risk bands"):
        AlertOperator(critical_risk=0.1, warning_risk=0.5)


def test_the_alert_reason_carries_the_numbers_that_caused_it():
    alert = AlertOperator(cooldown_h=0).process(_scored("M1", 5, risk=0.77))[0]
    assert "0.77" in alert.reason
    assert alert.to_json_obj()["machine_id"] == "M1"


def test_alerting_state_round_trips():
    operator = AlertOperator(cooldown_h=12, critical_risk=0.6, warning_risk=0.3)
    for t in range(0, 60, 6):
        operator.process(_scored("M1", t, risk=0.8))
    snapshot = operator.state()
    restored = AlertOperator()
    restored.restore(snapshot)
    assert restored.state() == snapshot
    assert restored.cooldown_h == 12
    assert restored.critical_risk == 0.6
