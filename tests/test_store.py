"""Metadata store: migrations, constraints, transactions and pagination."""
from __future__ import annotations

from collections.abc import Iterator

import pytest

from pme.errors import StorageError, ValidationError
from pme.store.db import MEMORY, MIGRATIONS, SCHEMA_VERSION, Database, open_database
from pme.store.repository import MAX_PAGE, MetadataStore


@pytest.fixture
def store() -> Iterator[MetadataStore]:
    db = Database(MEMORY)
    db.migrate()
    s = MetadataStore(db)
    s.start_run("run-1", "csv[t.csv]", 100, "fp-abc", {"window_size_h": 24})
    yield s
    db.close()


def _alert(machine: str, t: int, severity: str = "WARNING", risk: float = 0.3) -> dict:
    return {"machine_id": machine, "event_time": t, "severity": severity, "risk_prob": risk,
            "rul_est_h": 42.0, "anomaly": 0, "reason": "because"}


def test_migrations_apply_once_and_are_recorded(tmp_path):
    with Database(tmp_path / "meta.db") as db:
        assert db.migrate() == [v for v, _, _ in MIGRATIONS]
        assert db.migrate() == []
        assert db.applied_versions() == [v for v, _, _ in MIGRATIONS]
    assert MIGRATIONS[-1][0] == SCHEMA_VERSION


def test_migration_two_adds_the_correlation_column(tmp_path):
    with open_database(tmp_path / "meta.db") as db:
        cols = {r["name"] for r in db.query("PRAGMA table_info(runs)")}
    assert "correlation_id" in cols


def test_wal_is_enabled_on_a_file_database(tmp_path):
    with open_database(tmp_path / "meta.db") as db:
        row = db.query_one("PRAGMA journal_mode")
    assert row is not None and row[0].lower() == "wal"


def test_check_constraint_rejects_a_bad_severity(store):
    with pytest.raises(StorageError):
        store.db.execute(
            "INSERT INTO alerts (run_id, machine_id, event_time, severity, risk_prob, rul_est_h,"
            " anomaly, reason) VALUES ('run-1', 'M1', 5, 'SEVERE', 0.4, 1.0, 0, 'x')"
        )


def test_foreign_key_rejects_an_unknown_run(store):
    with pytest.raises(StorageError):
        store.record_alerts("run-missing", [_alert("M1", 1)])


def test_transaction_rolls_back_on_error(store):
    with pytest.raises(RuntimeError), store.db.transaction():
        store.db.execute(
            "INSERT INTO alerts (run_id, machine_id, event_time, severity, risk_prob, rul_est_h,"
            " anomaly, reason) VALUES ('run-1', 'M1', 5, 'WARNING', 0.4, 1.0, 0, 'x')"
        )
        raise RuntimeError("boom")
    assert store.count_alerts_by_severity("run-1") == {}


def test_replayed_alerts_do_not_duplicate(store):
    store.record_alerts("run-1", [_alert("M1", 5), _alert("M2", 6)])
    store.record_alerts("run-1", [_alert("M1", 5)])
    assert store.count_alerts_by_severity("run-1") == {"WARNING": 2}


def test_alert_paging_walks_the_whole_feed(store):
    store.record_alerts("run-1", [_alert(f"M{i:02d}", i) for i in range(25)])
    seen: list[int] = []
    cursor = None
    pages = 0
    while True:
        page = store.list_alerts("run-1", limit=10, after=cursor)
        seen.extend(int(r["event_time"]) for r in page.items)
        pages += 1
        if page.next_cursor is None:
            break
        cursor = int(page.next_cursor)
    assert seen == list(range(25))
    assert pages == 3


def test_alert_filters_are_bound_not_interpolated(store):
    store.record_alerts("run-1", [_alert("M1", 1, "CRITICAL", 0.9), _alert("M2", 2)])
    page = store.list_alerts("run-1", severity="CRITICAL")
    assert [r["machine_id"] for r in page.items] == ["M1"]
    page = store.list_alerts("run-1", machine_id="M2' OR '1'='1")
    assert page.items == []


def test_unknown_severity_is_a_validation_error(store):
    with pytest.raises(ValidationError):
        store.list_alerts("run-1", severity="LOUD")


def test_limit_is_clamped_and_validated(store):
    store.record_alerts("run-1", [_alert("M1", 1)])
    assert len(store.list_alerts("run-1", limit=MAX_PAGE + 1000).items) == 1
    with pytest.raises(ValidationError):
        store.list_alerts("run-1", limit=0)


def test_machine_state_never_moves_backwards_in_event_time(store):
    store.upsert_machine_state("run-1", [{"machine_id": "M1", "event_time": 10, "risk_prob": 0.7,
                                          "rul_est_h": 5.0, "anomaly": 0}])
    store.upsert_machine_state("run-1", [{"machine_id": "M1", "event_time": 4, "risk_prob": 0.1,
                                          "rul_est_h": 90.0, "anomaly": 0}])
    row = store.list_machine_state("run-1").items[0]
    assert row["event_time"] == 10
    assert row["risk_prob"] == pytest.approx(0.7)


def test_run_lifecycle_and_offset_paging(store):
    store.start_run("run-2", "csv[t.csv]", 100, "fp-abc", {})
    store.finish_run("run-1", "completed", 100, 3, 7)
    page = store.list_runs(limit=1)
    assert page.total == 2
    assert page.next_cursor == "1"
    assert store.get_run("run-1")["status"] == "completed"
    with pytest.raises(ValidationError):
        store.list_runs(status="melted")


def test_totals_cover_every_table(store):
    store.record_alerts("run-1", [_alert("M1", 1)])
    store.record_late_events("run-1", [{"machine_id": "M1", "event_time": 2, "watermark": 9, "lateness_h": 7}])
    store.quarantine_rows("t.csv", [{"row_number": 3, "code": "nan_value", "detail": "temp_c", "payload": {"a": 1}}])
    totals = store.totals()
    assert totals["runs"] == 1
    assert totals["alerts"] == 1
    assert totals["late_events"] == 1
    assert totals["quarantine"] == 1
