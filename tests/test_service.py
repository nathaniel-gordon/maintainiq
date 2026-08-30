"""HTTP surface: routing, validation, pagination, auth, rate limiting, metrics."""
from __future__ import annotations

import http.client
import json
import threading

import pytest

from pme.errors import MethodNotAllowedError, NotFoundError, RateLimitedError, UnauthorizedError, ValidationError
from pme.service.app import Api, RateLimiter, build_server
from pme.store.db import open_database
from pme.store.repository import MetadataStore


@pytest.fixture
def api(store) -> Api:
    store.start_run("run-1", "csv[t.csv]", 100, "fp", {"window_size_h": 24})
    store.record_alerts("run-1", [
        {"machine_id": f"M{i:02d}", "event_time": i, "severity": "CRITICAL" if i % 2 else "WARNING",
         "risk_prob": 0.6, "rul_est_h": 12.0, "anomaly": 0, "reason": "x"}
        for i in range(12)
    ])
    store.record_late_events("run-1", [{"machine_id": "M01", "event_time": 3, "watermark": 9, "lateness_h": 7}])
    store.upsert_machine_state("run-1", [{"machine_id": "M01", "event_time": 9, "risk_prob": 0.6,
                                          "rul_est_h": 12.0, "anomaly": 0}])
    store.quarantine_rows("t.csv", [{"row_number": 1, "code": "not_numeric", "detail": "temp_c", "payload": {}}])
    store.finish_run("run-1", "completed", 100, 1, 12)
    return Api(store)


def call(api: Api, path: str, params=None, method: str = "GET", auth=None, client: str = "1.2.3.4"):
    return api.dispatch(method, path, params or {}, auth, client)


def test_healthz_needs_no_dependencies(api):
    status, body, kind = call(api, "/healthz")
    assert status == 200
    assert body["status"] == "ok"
    assert kind == "application/json"


def test_readyz_reports_the_schema_version(api):
    status, body, _ = call(api, "/readyz")
    assert status == 200
    assert body["status"] == "ready"
    assert body["schema_version"] == body["expected_schema_version"]


def test_metrics_is_prometheus_text(api):
    status, body, kind = call(api, "/metrics")
    assert status == 200
    assert kind.startswith("text/plain")
    assert "# TYPE pme_rows_total gauge" in body
    assert 'pme_rows_total{table="alerts"} 12' in body
    assert "pme_last_run_drop_rate" in body


def test_unknown_paths_are_404(api):
    with pytest.raises(NotFoundError):
        call(api, "/nope")
    with pytest.raises(NotFoundError):
        call(api, "/api/v1/runs/run-1/nonsense")
    with pytest.raises(NotFoundError):
        call(api, "/api/v1/runs/missing")


def test_writes_are_not_allowed(api):
    with pytest.raises(MethodNotAllowedError):
        call(api, "/api/v1/runs", method="POST")


def test_runs_and_run_detail(api):
    _, body, _ = call(api, "/api/v1/runs")
    assert body["total"] == 1
    _, run, _ = call(api, "/api/v1/runs/run-1")
    assert run["status"] == "completed"
    assert run["alerts_by_severity"] == {"CRITICAL": 6, "WARNING": 6}


def test_alerts_are_paginated_by_cursor(api):
    _, first, _ = call(api, "/api/v1/runs/run-1/alerts", {"limit": ["5"]})
    assert len(first["items"]) == 5
    assert first["next_cursor"] is not None
    _, second, _ = call(api, "/api/v1/runs/run-1/alerts", {"limit": ["5"], "after": [first["next_cursor"]]})
    assert {r["id"] for r in first["items"]}.isdisjoint({r["id"] for r in second["items"]})


def test_alert_filters(api):
    _, body, _ = call(api, "/api/v1/runs/run-1/alerts", {"severity": ["CRITICAL"]})
    assert {r["severity"] for r in body["items"]} == {"CRITICAL"}
    with pytest.raises(ValidationError):
        call(api, "/api/v1/runs/run-1/alerts", {"severity": ["NOISY"]})


def test_bad_parameters_are_rejected(api):
    with pytest.raises(ValidationError):
        call(api, "/api/v1/runs/run-1/alerts", {"limit": ["banana"]})
    with pytest.raises(ValidationError):
        call(api, "/api/v1/runs", {"offset": ["-1"]})
    with pytest.raises(ValidationError):
        call(api, "/api/v1/quarantine", {"source": ["x" * 500]})


def test_machines_and_late_events_and_quarantine(api):
    _, machines, _ = call(api, "/api/v1/runs/run-1/machines")
    assert machines["items"][0]["machine_id"] == "M01"
    _, late, _ = call(api, "/api/v1/runs/run-1/late-events")
    assert late["items"][0]["lateness_h"] == 7
    _, quarantine, _ = call(api, "/api/v1/quarantine", {"source": ["t.csv"]})
    assert quarantine["items"][0]["code"] == "not_numeric"


def test_auth_guards_the_api_but_not_the_probes(store):
    api = Api(store, token="s3cret")
    assert call(api, "/healthz")[0] == 200
    assert call(api, "/metrics")[0] == 200
    with pytest.raises(UnauthorizedError):
        call(api, "/api/v1/runs")
    with pytest.raises(UnauthorizedError):
        call(api, "/api/v1/runs", auth="Bearer wrong")
    assert call(api, "/api/v1/runs", auth="Bearer s3cret")[0] == 200


def test_rate_limit_applies_per_client(store):
    api = Api(store, limiter=RateLimiter(limit=3, window_s=60.0))
    for _ in range(3):
        call(api, "/api/v1/runs", client="10.0.0.1")
    with pytest.raises(RateLimitedError):
        call(api, "/api/v1/runs", client="10.0.0.1")
    assert call(api, "/api/v1/runs", client="10.0.0.2")[0] == 200
    # Probes stay reachable while a client is limited.
    assert call(api, "/healthz", client="10.0.0.1")[0] == 200


def test_rate_limit_window_rolls_over():
    limiter = RateLimiter(limit=1, window_s=10.0)
    limiter.check("a", now=0.0)
    with pytest.raises(RateLimitedError):
        limiter.check("a", now=5.0)
    limiter.check("a", now=20.0)


def test_over_the_wire(tmp_path):
    """One real request through the socket, headers and all.

    File-backed rather than in-memory: the server serves each request on its own
    thread, and an in-memory database belongs to the connection that made it.
    """
    store = MetadataStore(open_database(tmp_path / "meta.db"))
    store.start_run("run-http", "csv[t.csv]", 10, "fp", {})
    server = build_server(store, host="127.0.0.1", port=0, token="")
    host, port = server.server_address[:2]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        conn = http.client.HTTPConnection(str(host), port, timeout=5)
        conn.request("GET", "/api/v1/runs?limit=1", headers={"X-Request-Id": "abc123"})
        response = conn.getresponse()
        payload = json.loads(response.read())
        assert response.status == 200
        assert response.headers["X-Request-Id"] == "abc123"
        assert payload["items"][0]["run_id"] == "run-http"

        conn.request("GET", "/healthz")
        assert conn.getresponse().status == 200
        conn.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        store.db.close()
