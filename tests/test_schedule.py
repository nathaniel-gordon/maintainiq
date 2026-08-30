"""The schedule sink: cost gating, crew capacity and a reproducible ranking."""
from __future__ import annotations

import csv

import pytest

from pme.core.costs import CostModel
from pme.core.events import ScoredRecord
from pme.sinks.schedule import MaintenanceScheduleSink, build_schedule

CHEAP = CostModel(planned_maintenance=100.0, unplanned_failure=10_000.0,
                  downtime_per_hour=0.0, unplanned_downtime_h=0.0, planned_downtime_h=0.0)


def _record(machine: str, risk: float, anomaly: int = 0, t: int = 100) -> ScoredRecord:
    return ScoredRecord(machine, t, risk, 40.0, anomaly)


def _latest(*records: ScoredRecord) -> dict[str, ScoredRecord]:
    return {r.key: r for r in records}


def test_the_breakeven_risk_is_where_saving_turns_positive():
    costs = CostModel()
    assert costs.expected_saving(costs.breakeven_risk) == pytest.approx(0.0)
    assert costs.expected_saving(costs.breakeven_risk + 0.05) > 0
    assert costs.expected_saving(costs.breakeven_risk - 0.05) < 0


def test_a_machine_below_breakeven_is_not_scheduled():
    orders = build_schedule(_latest(_record("M1", 0.001)), costs=CHEAP)
    assert orders == []


def test_a_machine_above_breakeven_is_scheduled():
    orders = build_schedule(_latest(_record("M1", 0.9)), costs=CHEAP)
    assert [o.machine_id for o in orders] == ["M1"]
    assert orders[0].expected_saving > 0


def test_an_anomalous_machine_earns_a_slot_below_breakeven():
    """A novel signature the classifier has never seen still deserves a look."""
    orders = build_schedule(_latest(_record("M1", 0.001, anomaly=1)), costs=CHEAP)
    assert [o.machine_id for o in orders] == ["M1"]
    assert "anomalous" in orders[0].reason
    assert orders[0].expected_saving == 0.0


def test_the_anomaly_bonus_lifts_a_machine_up_the_ranking():
    latest = _latest(_record("M1", 0.30), _record("M2", 0.25, anomaly=1))
    without = build_schedule(latest, costs=CHEAP, anomaly_bonus=0.0)
    with_bonus = build_schedule(latest, costs=CHEAP, anomaly_bonus=100_000.0)
    assert [o.machine_id for o in without] == ["M1", "M2"]
    assert [o.machine_id for o in with_bonus] == ["M2", "M1"]


def test_crews_are_a_hard_capacity_per_day():
    latest = _latest(*[_record(f"M{i:02d}", 0.9 - i * 0.01) for i in range(7)])
    orders = build_schedule(latest, costs=CHEAP, crews_per_day=3, horizon_days=7)
    days = [o.day for o in orders]
    assert days == [1, 1, 1, 2, 2, 2, 3]


def test_the_horizon_truncates_the_plan():
    latest = _latest(*[_record(f"M{i:02d}", 0.9) for i in range(20)])
    orders = build_schedule(latest, costs=CHEAP, crews_per_day=2, horizon_days=3)
    assert len(orders) == 6
    assert max(o.day for o in orders) == 3


def test_ties_break_on_machine_id_so_the_plan_is_reproducible():
    latest = _latest(_record("M09", 0.5), _record("M01", 0.5), _record("M05", 0.5))
    first = build_schedule(latest, costs=CHEAP)
    second = build_schedule(dict(reversed(list(latest.items()))), costs=CHEAP)
    assert [o.machine_id for o in first] == ["M01", "M05", "M09"]
    assert first == second


def test_the_sink_keeps_the_newest_window_per_machine(tmp_path):
    sink = MaintenanceScheduleSink(tmp_path / "schedule.csv", costs=CHEAP)
    sink.write(_record("M1", 0.9, t=10))
    sink.write(_record("M1", 0.1, t=5))
    assert sink.latest["M1"].risk_prob == pytest.approx(0.9)
    assert sink.updates == 2


def test_a_newer_window_replaces_an_older_one(tmp_path):
    sink = MaintenanceScheduleSink(tmp_path / "schedule.csv", costs=CHEAP)
    sink.write(_record("M1", 0.1, t=5))
    sink.write(_record("M1", 0.9, t=10))
    assert sink.latest["M1"].event_time == 10


def test_the_schedule_can_be_read_mid_stream(tmp_path):
    sink = MaintenanceScheduleSink(tmp_path / "schedule.csv", costs=CHEAP)
    sink.write(_record("M1", 0.001))
    assert sink.current_schedule() == []
    sink.write(_record("M2", 0.9))
    assert [o.machine_id for o in sink.current_schedule()] == ["M2"]


def test_closing_writes_a_csv_with_a_header(tmp_path):
    path = tmp_path / "schedule.csv"
    sink = MaintenanceScheduleSink(path, costs=CHEAP)
    sink.write(_record("M1", 0.9))
    sink.close()
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["machine_id", "day", "event_time", "risk_prob", "rul_est_h",
                       "expected_saving", "reason"]
    assert rows[1][0] == "M1"
    assert sink.summary()["service_orders"] == 1


def test_an_empty_schedule_still_writes_its_header(tmp_path):
    path = tmp_path / "schedule.csv"
    sink = MaintenanceScheduleSink(path, costs=CHEAP)
    sink.close()
    assert path.read_text(encoding="utf-8").startswith("machine_id,day")


def test_schedule_state_round_trips(tmp_path):
    sink = MaintenanceScheduleSink(tmp_path / "schedule.csv", costs=CHEAP)
    for i in range(5):
        sink.write(_record(f"M{i}", 0.1 * i, anomaly=i % 2, t=i))
    snapshot = sink.state()
    restored = MaintenanceScheduleSink(tmp_path / "schedule.csv", costs=CHEAP)
    restored.restore(snapshot)
    assert restored.state() == snapshot
    assert restored.current_schedule() == sink.current_schedule()


def test_the_summary_reports_the_economics_it_used(tmp_path):
    sink = MaintenanceScheduleSink(tmp_path / "schedule.csv", costs=CHEAP)
    sink.write(_record("M1", 0.9))
    sink.close()
    summary = sink.summary()
    assert summary["breakeven_risk"] == pytest.approx(round(CHEAP.breakeven_risk, 4))
    assert summary["total_expected_saving"] > 0


def test_cost_models_round_trip_through_their_dict_form():
    costs = CostModel(unplanned_failure=20_000.0)
    assert CostModel.from_dict(costs.to_dict()) == costs
