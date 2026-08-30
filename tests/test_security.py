"""Regression tests for the weaknesses the hardening pass closed.

Every test in this file names one specific hole and fails if it reopens. They
duplicate a little of the coverage in the per-module files on purpose: those
tests describe behaviour, these describe attacks, and the two drift apart if
they share a home.

What is *not* claimed here: nothing in this system has been penetration tested,
and the threat model is a local operator plus whatever reaches the read-only
HTTP surface. See SECURITY.md.
"""
from __future__ import annotations

import http.client
import inspect
import io
import json
import re
import threading
from pathlib import Path

import joblib
import pytest

from pme.errors import (
    CheckpointError,
    ModelIntegrityError,
    PathError,
    StorageError,
    UnauthorizedError,
    ValidationError,
)
from pme.learning.bundle import MaintenanceModels, manifest_path
from pme.logging_setup import configure_logging, get_logger
from pme.paths import ensure_readable_file
from pme.runtime.checkpoint import CheckpointStore
from pme.service.app import Api, build_server
from pme.sources.csvfile import CsvTelemetrySource
from pme.store import repository
from pme.store.db import MEMORY, Database, open_database
from pme.store.repository import MAX_PAGE, MetadataStore

#: Payloads that break a query built by string concatenation and do nothing at
#: all against a bound parameter.
INJECTIONS = (
    "' OR '1'='1",
    "'; DROP TABLE alerts; --",
    "x' UNION SELECT run_id, 1, 1, 1, 1, 1, 1, 1, 1 FROM runs --",
    "\x00truncated",
)


@pytest.fixture
def api(store: MetadataStore) -> Api:
    store.start_run("run-1", "csv[t.csv]", 10, "fp", {})
    store.record_alerts("run-1", [{"machine_id": "M01", "event_time": 1, "severity": "WARNING",
                                   "risk_prob": 0.3, "rul_est_h": 9.0, "anomaly": 0, "reason": "x"}])
    store.quarantine_rows("t.csv", [{"row_number": 1, "code": "not_numeric", "detail": "d", "payload": {}}])
    store.finish_run("run-1", "completed", 10, 0, 1)
    return Api(store)


def call(api: Api, path: str, params: dict | None = None, method: str = "GET",
         auth: str | None = None, client: str = "1.2.3.4"):
    return api.dispatch(method, path, params or {}, auth, client)


# --------------------------------------------------------------- SQL injection


@pytest.mark.parametrize("payload", INJECTIONS)
def test_an_injection_in_a_filter_matches_nothing_and_breaks_nothing(api, payload):
    """Every filter is a bound parameter, so a quote is just a character."""
    _, body, _ = call(api, "/api/v1/runs/run-1/alerts", {"machine_id": [payload]})
    assert body["items"] == []
    assert call(api, "/api/v1/runs/run-1/alerts")[1]["count"] == 1


@pytest.mark.parametrize("payload", INJECTIONS)
def test_an_injection_in_the_quarantine_source_filter_is_inert(api, payload):
    _, body, _ = call(api, "/api/v1/quarantine", {"source": [payload]})
    assert body["items"] == []
    assert api.store.totals()["quarantine"] == 1


def test_an_injection_as_a_run_id_is_a_404_not_a_query(api):
    """The path segment reaches ``WHERE run_id = ?``; it never reaches the SQL text."""
    _, body, _ = api.handle("GET", "/api/v1/runs/'; DROP TABLE alerts; --", {}, None, "1.2.3.4")
    assert body["error"]["code"] == "not_found"
    assert api.store.totals()["alerts"] == 1


def test_a_severity_outside_the_whitelist_is_refused_before_the_query(api):
    with pytest.raises(ValidationError):
        call(api, "/api/v1/runs/run-1/alerts", {"severity": ["CRITICAL' OR 1=1 --"]})


def test_a_run_status_outside_the_whitelist_is_refused(api):
    with pytest.raises(ValidationError):
        call(api, "/api/v1/runs", {"status": ["completed' --"]})


# ------------------------------------------------------- metrics text injection


def test_a_run_id_cannot_forge_lines_in_the_prometheus_exposition(store):
    """``--run-id`` is operator text and lands in a metric label.

    Unescaped, a newline in it would let that text append arbitrary samples to
    the exposition, which a scraper would ingest as real metrics.
    """
    hostile = 'r"} 999\npme_forged_total 1\n#'
    store.start_run(hostile, "csv[t.csv]", 1, "fp", {})
    store.finish_run(hostile, "completed", 1, 0, 0)
    body = Api(store).dispatch("GET", "/metrics", {}, None, "cli")[1]
    assert isinstance(body, str)
    assert "\npme_forged_total" not in body
    assert r"\n" in body  # the newline survives, escaped
    for line in body.splitlines():
        assert not line.startswith("pme_forged_total")


# ------------------------------------------------------------ resource limits


def test_a_caller_cannot_ask_for_an_unbounded_page(api):
    """A page size is clamped, not trusted: ``limit=10_000_000`` is a memory bomb."""
    for i in range(2, 12):
        api.store.record_alerts("run-1", [{"machine_id": f"M{i:02d}", "event_time": i,
                                           "severity": "WARNING", "risk_prob": 0.3, "rul_est_h": 9.0,
                                           "anomaly": 0, "reason": "x"}])
    _, body, _ = call(api, "/api/v1/runs/run-1/alerts", {"limit": ["10000000"]})
    assert body["count"] <= MAX_PAGE


def test_a_zero_or_negative_page_size_is_a_validation_error(api):
    for bad in ("0", "-5"):
        with pytest.raises(ValidationError):
            call(api, "/api/v1/runs/run-1/alerts", {"limit": [bad]})


def test_an_overlong_string_parameter_is_refused_before_it_reaches_sqlite(api):
    with pytest.raises(ValidationError):
        call(api, "/api/v1/quarantine", {"source": ["x" * 100_000]})


def test_an_oversized_telemetry_file_is_refused_by_size_not_by_parsing(tmp_path):
    """The source materialises the file twice, so an unbounded read is a DoS."""
    big = tmp_path / "big.csv"
    big.write_bytes(b"machine_id,hour\n" + b"x" * 4096)
    with pytest.raises(PathError, match="size limit"):
        CsvTelemetrySource(big, max_bytes=512)


def test_the_input_cap_is_enforced_on_the_path_itself(tmp_path):
    target = tmp_path / "t.csv"
    target.write_bytes(b"x" * 100)
    with pytest.raises(PathError, match="size limit") as excinfo:
        ensure_readable_file(target, max_bytes=10)
    assert excinfo.value.details["limit"] == 10


# --------------------------------------------------------------------- auth


def test_the_api_is_shut_without_a_token_when_one_is_configured(store):
    api = Api(store, token="s3cret-token")
    with pytest.raises(UnauthorizedError):
        call(api, "/api/v1/runs")


@pytest.mark.parametrize(
    "header",
    [
        "",
        "s3cret-token",              # no scheme
        "Basic s3cret-token",        # wrong scheme
        "bearer s3cret-token",       # the scheme is case-sensitive here
        "Bearer s3cret-toke",        # a prefix must not pass
        "Bearer s3cret-tokenn",      # nor an extension
        "Bearer S3CRET-TOKEN",
    ],
)
def test_a_near_miss_token_does_not_open_the_api(store, header):
    api = Api(store, token="s3cret-token")
    with pytest.raises(UnauthorizedError):
        call(api, "/api/v1/runs", auth=header or None)


def test_the_probes_stay_open_so_a_liveness_check_needs_no_credential(store):
    api = Api(store, token="s3cret-token")
    assert call(api, "/healthz")[0] == 200
    assert call(api, "/readyz")[0] == 200
    assert call(api, "/metrics")[0] == 200


def test_an_unauthorized_response_carries_no_data(api_token_server):
    """A rejected request must not leak the resource it was refused."""
    host, port, token = api_token_server
    conn = http.client.HTTPConnection(host, port, timeout=5)
    conn.request("GET", "/api/v1/runs")
    response = conn.getresponse()
    payload = json.loads(response.read())
    conn.close()
    assert response.status == 401
    assert payload["error"]["code"] == "unauthorized"
    assert "items" not in payload
    assert token not in json.dumps(payload)


@pytest.fixture
def api_token_server(tmp_path):
    """A real server behind a bearer token, on a file-backed database."""
    store = MetadataStore(open_database(tmp_path / "meta.db"))
    store.start_run("run-secret", "csv[t.csv]", 1, "fp", {})
    server = build_server(store, host="127.0.0.1", port=0, token="tok-abcdef123456")
    host, port = server.server_address[:2]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield str(host), int(port), "tok-abcdef123456"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        store.db.close()


def test_the_bearer_token_never_reaches_the_log(api_token_server):
    """The handler logs *whether* a request was authenticated, never the credential."""
    host, port, token = api_token_server
    stream = io.StringIO()
    configure_logging(level="DEBUG", fmt="json", stream=stream)
    try:
        conn = http.client.HTTPConnection(host, port, timeout=5)
        conn.request("GET", "/api/v1/runs", headers={"Authorization": f"Bearer {token}"})
        assert conn.getresponse().status == 200
        conn.close()
    finally:
        text = stream.getvalue()
        configure_logging(level="INFO", fmt="plain", stream=io.StringIO())
    assert text  # the request really was logged
    assert token not in text
    assert "Bearer" not in text
    assert any(json.loads(line).get("authenticated") is True for line in text.splitlines() if line.strip())


def test_credential_shaped_log_fields_are_masked():
    stream = io.StringIO()
    configure_logging(level="INFO", fmt="json", stream=stream)
    try:
        get_logger("pme.test").info("configured", api_token="s3cret", password="hunter2", machine_id="M01")
    finally:
        configure_logging(level="INFO", fmt="plain", stream=io.StringIO())
    record = json.loads(stream.getvalue().splitlines()[0])
    assert record["api_token"] == "***"
    assert record["password"] == "***"
    assert record["machine_id"] == "M01"
    assert "s3cret" not in stream.getvalue()
    assert "hunter2" not in stream.getvalue()


# ----------------------------------------------------------- untrusted state


def test_a_bundle_without_a_manifest_is_never_unpickled(tmp_path):
    """joblib.load executes whatever is in the file; the manifest is the gate."""
    bundle = tmp_path / "models.joblib"
    joblib.dump({"not": "a bundle"}, bundle)
    with pytest.raises(ModelIntegrityError, match="no manifest"):
        MaintenanceModels.load(bundle)


def test_a_bundle_edited_after_its_manifest_was_written_is_refused(tmp_path):
    bundle = tmp_path / "models.joblib"
    joblib.dump(MaintenanceModels(seed=1), bundle)
    manifest_path(bundle).write_text(json.dumps({"sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(ModelIntegrityError, match="manifest digest"):
        MaintenanceModels.load(bundle)


def test_a_file_that_verifies_but_holds_something_else_is_still_refused(tmp_path):
    """A matching digest proves the bytes are unchanged, not that they are a bundle."""
    bundle = tmp_path / "models.joblib"
    joblib.dump(["arbitrary", "object"], bundle)
    from pme.learning.bundle import sha256_of
    manifest_path(bundle).write_text(json.dumps({"sha256": sha256_of(bundle)}), encoding="utf-8")
    with pytest.raises(ModelIntegrityError, match="did not contain a model bundle"):
        MaintenanceModels.load(bundle)


def test_an_edited_checkpoint_is_refused_by_its_digest(tmp_path):
    """The checkpoint is trusted state; an edit to it steers the next run."""
    store = CheckpointStore(tmp_path / "checkpoint.json")
    store.write({"events_consumed": 10, "source": {"cursor": 10}})
    payload = json.loads(store.path.read_text(encoding="utf-8"))
    payload["events_consumed"] = 10_000
    store.path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(CheckpointError, match="digest"):
        store.read()


def test_a_truncated_checkpoint_is_refused_rather_than_half_restored(tmp_path):
    store = CheckpointStore(tmp_path / "checkpoint.json")
    store.write({"events_consumed": 10})
    raw = store.path.read_text(encoding="utf-8")
    store.path.write_text(raw[: len(raw) // 2], encoding="utf-8")
    with pytest.raises(CheckpointError, match="not valid JSON"):
        store.read()


def test_a_checkpoint_from_an_older_layout_is_refused_not_reinterpreted(tmp_path):
    store = CheckpointStore(tmp_path / "checkpoint.json")
    store.write({"events_consumed": 10})
    payload = json.loads(store.path.read_text(encoding="utf-8"))
    payload["version"] = 1
    store.path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(CheckpointError, match="version"):
        store.read()


# ----------------------------------------------------------------- migrations


def test_a_failing_migration_leaves_no_half_applied_schema(tmp_path, monkeypatch):
    """``executescript`` used to commit the open transaction before running.

    A migration whose second statement failed therefore left the first one
    committed and the version row absent, so the next start-up tried to create a
    table that already existed and the database was stuck.
    """
    from pme.store import db as db_module

    broken = (*db_module.MIGRATIONS, (99, "broken", ("CREATE TABLE half_applied (a INTEGER)", "NOT SQL")))
    monkeypatch.setattr(db_module, "MIGRATIONS", broken)
    with Database(tmp_path / "meta.db") as database:
        with pytest.raises(StorageError, match="rolled back"):
            database.migrate()
        assert 99 not in database.applied_versions()
        assert database.query_one(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'half_applied'"
        ) is None


def test_foreign_keys_are_enforced_not_merely_declared():
    """SQLite ignores REFERENCES unless the pragma is on for the connection."""
    with Database(MEMORY) as database:
        database.migrate()
        store = MetadataStore(database)
        with pytest.raises(StorageError):
            store.record_alerts("no-such-run", [{"machine_id": "M01", "event_time": 1,
                                                 "severity": "WARNING", "risk_prob": 0.3,
                                                 "rul_est_h": 1.0, "anomaly": 0, "reason": "x"}])


def test_a_risk_probability_outside_zero_to_one_cannot_be_stored():
    """The check constraints are the last line against a bad score reaching the API."""
    with Database(MEMORY) as database:
        database.migrate()
        store = MetadataStore(database)
        store.start_run("run-1", "s", 1, "fp", {})
        with pytest.raises(StorageError):
            store.record_alerts("run-1", [{"machine_id": "M01", "event_time": 1, "severity": "WARNING",
                                           "risk_prob": 4.2, "rul_est_h": 1.0, "anomaly": 0,
                                           "reason": "x"}])


# ------------------------------------------------------------------- methods


def test_the_surface_is_read_only(api):
    """No route mutates. A write verb is refused before any handler runs."""
    for method in ("POST", "PUT", "PATCH", "DELETE"):
        status, body, _ = api.handle(method, "/api/v1/runs", {}, None, "1.2.3.4")
        assert status == 405
        assert body["error"]["code"] == "method_not_allowed"


def test_the_only_interpolated_sql_fragments_are_locally_built():
    """A static guard on the shape of the original bug, not on one payload.

    Every value in :mod:`pme.store.repository` is bound. The few f-strings in it
    splice in fragments the module builds itself from fixed literals — a WHERE
    clause assembled from whitelisted column names, a table name from a tuple in
    the source. If a statement ever grows a hole for caller text, this fails.
    """
    source = Path(inspect.getfile(repository)).read_text(encoding="utf-8")
    holes = set(re.findall(r'f"[^"]*\{([^}]+)\}', source))
    assert holes <= {"where", "table", "' AND '.join(clauses)"}, holes
