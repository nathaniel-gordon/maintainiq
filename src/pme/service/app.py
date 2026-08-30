"""HTTP surface: health, metrics and the run history.

Built on ``http.server`` with a thread per request. That is not a high-traffic
web server and it is not pretending to be one: the audience is a container
orchestrator asking whether the process is alive, a Prometheus scraper, and an
operator paging through the last run's alerts. Everything it serves is read-only
and comes from the SQLite store the daemon writes.

Contract
--------
* ``/healthz``  — the process is up. No dependencies checked, so it stays cheap
  and cannot fail because the database is busy.
* ``/readyz``   — the database opens and its schema is current. This is the one
  that should gate traffic.
* ``/metrics``  — Prometheus text exposition.
* ``/api/v1/…`` — versioned JSON. Every collection is paginated; every error is
  the same ``{"error": {"code", "message", "details"}}`` object.

Auth is opt-in through ``PME_API_TOKEN``: when it is set, ``/api/v1`` requires a
matching bearer token, while the health and metrics endpoints stay open so a
probe does not need a credential. Rate limiting applies to ``/api/v1`` only, for
the same reason.
"""
from __future__ import annotations

import hmac
import json
import os
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from ..errors import (
    EXIT_OK,
    MethodNotAllowedError,
    NotFoundError,
    PmeError,
    RateLimitedError,
    StorageError,
    UnauthorizedError,
    ValidationError,
)
from ..logging_setup import configure_logging, correlation_scope, get_logger, log_level_from_env, new_correlation_id
from ..metrics import REGISTRY, Registry, escape_label
from ..runtime.config import OutputPaths
from ..store.db import SCHEMA_VERSION, open_database
from ..store.repository import MetadataStore

log = get_logger(__name__)

API_PREFIX = "/api/v1"

#: Requests per window, per client, against /api/v1. Generous: this is a guard
#: against a runaway script, not a commercial quota.
RATE_LIMIT_REQUESTS = 120
RATE_LIMIT_WINDOW_S = 60.0

#: Environment variable holding the bearer token. Empty or unset disables auth.
TOKEN_ENV = "PME_API_TOKEN"


class RateLimiter:
    """Fixed-window request counter keyed by client address."""

    def __init__(self, limit: int = RATE_LIMIT_REQUESTS, window_s: float = RATE_LIMIT_WINDOW_S) -> None:
        self.limit = limit
        self.window_s = window_s
        self._hits: dict[str, deque[float]] = {}

    def check(self, client: str, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        hits = self._hits.setdefault(client, deque())
        while hits and now - hits[0] > self.window_s:
            hits.popleft()
        if len(hits) >= self.limit:
            raise RateLimitedError("too many requests", limit=self.limit, window_s=self.window_s)
        hits.append(now)


def _int_param(params: dict[str, list[str]], name: str) -> int | None:
    values = params.get(name) or []
    raw = values[0] if values else ""
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise ValidationError("parameter must be an integer", parameter=name, value=raw) from exc


def _str_param(params: dict[str, list[str]], name: str, max_len: int = 128) -> str | None:
    values = params.get(name) or []
    raw = values[0] if values else ""
    if not raw:
        return None
    if len(raw) > max_len:
        raise ValidationError("parameter is too long", parameter=name, limit=max_len)
    return raw


#: Sub-resources of a run that get their own route label.
_RUN_SUBRESOURCES = frozenset({"alerts", "machines", "late-events"})


def route_of(path: str) -> str:
    """Collapse a request path onto a route template.

    Metric labels must have bounded cardinality. Labelling by raw path would
    create a new time series per run id, so the id is replaced by a placeholder
    and anything unrecognised collapses to ``other``.
    """
    if path in ("/healthz", "/readyz", "/metrics"):
        return path
    if not path.startswith(API_PREFIX):
        return "other"
    segments = [s for s in path[len(API_PREFIX):].split("/") if s]
    if not segments:
        return API_PREFIX
    if segments[0] == "quarantine" and len(segments) == 1:
        return f"{API_PREFIX}/quarantine"
    if segments[0] == "runs":
        if len(segments) == 1:
            return f"{API_PREFIX}/runs"
        if len(segments) == 2:
            return f"{API_PREFIX}/runs/{{run_id}}"
        if len(segments) == 3 and segments[2] in _RUN_SUBRESOURCES:
            return f"{API_PREFIX}/runs/{{run_id}}/{segments[2]}"
    return "other"


class Api:
    """Routing and handlers, independent of the HTTP plumbing so it can be tested."""

    def __init__(self, store: MetadataStore, token: str = "", limiter: RateLimiter | None = None,
                 registry: Registry | None = None) -> None:
        self.store = store
        self.token = token
        self.limiter = limiter or RateLimiter()
        self.started_at = time.time()
        self.registry = registry or REGISTRY
        self.requests = self.registry.counter(
            "pme_http_requests_total",
            "HTTP requests answered, by method, route template and status.",
            ("method", "route", "status"),
        )
        self.latency = self.registry.histogram(
            "pme_http_request_duration_seconds",
            "Wall-clock time spent handling a request, by route template.",
            ("route",),
        )

    # ------------------------------------------------------------- entry point

    def handle(self, method: str, path: str, params: dict[str, list[str]],
               auth_header: str | None, client: str) -> tuple[int, dict[str, Any] | str, str]:
        """Dispatch one request, turn any failure into a response, and record it.

        The HTTP plumbing calls this rather than :meth:`dispatch` so that the
        counter and the latency histogram cover rejected requests too — a service
        whose metrics only describe its successes is the wrong way round.
        """
        route = route_of(path)
        started = time.perf_counter()
        try:
            status, body, content_type = self.dispatch(method, path, params, auth_header, client)
        except PmeError as exc:
            status, body, content_type = exc.http_status, {"error": exc.to_dict()}, "application/json"
            log.warning("request rejected", path=path, code=exc.code, status=status)
        except Exception as exc:  # pragma: no cover - defensive
            status, content_type = 500, "application/json"
            body = {"error": {"code": "internal_error", "message": "unhandled error"}}
            log.exception("request failed", path=path, detail=str(exc))
        self.latency.observe(time.perf_counter() - started, route=route)
        self.requests.inc(method=method, route=route, status=str(status))
        return status, body, content_type

    # ------------------------------------------------------------------ auth

    def authorize(self, header: str | None) -> None:
        if not self.token:
            return
        if not header or not header.startswith("Bearer "):
            raise UnauthorizedError("a bearer token is required")
        supplied = header.removeprefix("Bearer ").strip()
        if not hmac.compare_digest(supplied, self.token):
            raise UnauthorizedError("bearer token is not valid")

    # --------------------------------------------------------------- routing

    def dispatch(self, method: str, path: str, params: dict[str, list[str]],
                 auth_header: str | None, client: str) -> tuple[int, dict[str, Any] | str, str]:
        """Return ``(status, body, content_type)`` for one request."""
        if method not in ("GET", "HEAD"):
            raise MethodNotAllowedError("only GET is supported", method=method, path=path)

        if path == "/healthz":
            return 200, {"status": "ok", "uptime_s": round(time.time() - self.started_at, 3)}, "application/json"
        if path == "/readyz":
            return self._readyz()
        if path == "/metrics":
            return 200, self._metrics(), "text/plain; version=0.0.4"

        if not path.startswith(API_PREFIX):
            raise NotFoundError("no such resource", path=path)

        self.authorize(auth_header)
        self.limiter.check(client)
        rest = path[len(API_PREFIX):].rstrip("/") or "/"
        segments = [s for s in rest.split("/") if s]

        if segments == []:
            return 200, {"version": "v1", "resources": ["runs", "quarantine"]}, "application/json"
        if segments == ["runs"]:
            page = self.store.list_runs(limit=_int_param(params, "limit"),
                                        offset=_int_param(params, "offset") or 0,
                                        status=_str_param(params, "status"))
            return 200, page.to_dict(), "application/json"
        if segments == ["quarantine"]:
            page = self.store.list_quarantine(source=_str_param(params, "source"),
                                              limit=_int_param(params, "limit"),
                                              after=_int_param(params, "after"))
            return 200, page.to_dict(), "application/json"
        if len(segments) >= 2 and segments[0] == "runs":
            return self._run_routes(segments[1], segments[2:], params)
        raise NotFoundError("no such resource", path=path)

    def _run_routes(self, run_id: str, tail: list[str], params: dict[str, list[str]]
                    ) -> tuple[int, dict[str, Any], str]:
        run = self.store.get_run(run_id)
        if run is None:
            raise NotFoundError("no such run", run_id=run_id)
        if tail == []:
            run["alerts_by_severity"] = self.store.count_alerts_by_severity(run_id)
            run["latest_checkpoint"] = self.store.latest_checkpoint(run_id)
            return 200, run, "application/json"
        if tail == ["alerts"]:
            page = self.store.list_alerts(run_id, limit=_int_param(params, "limit"),
                                          after=_int_param(params, "after"),
                                          severity=_str_param(params, "severity"),
                                          machine_id=_str_param(params, "machine_id"))
            return 200, page.to_dict(), "application/json"
        if tail == ["machines"]:
            page = self.store.list_machine_state(run_id, limit=_int_param(params, "limit"),
                                                 offset=_int_param(params, "offset") or 0)
            return 200, page.to_dict(), "application/json"
        if tail == ["late-events"]:
            page = self.store.list_late_events(run_id, limit=_int_param(params, "limit"),
                                               after=_int_param(params, "after"))
            return 200, page.to_dict(), "application/json"
        raise NotFoundError("no such run sub-resource", run_id=run_id, path="/".join(tail))

    # ------------------------------------------------------------- readiness

    def _readyz(self) -> tuple[int, dict[str, Any], str]:
        try:
            versions = self.store.db.applied_versions()
        except StorageError as exc:
            return 503, {"status": "unready", "reason": "database unavailable", "detail": exc.message}, \
                "application/json"
        current = bool(versions) and max(versions) >= SCHEMA_VERSION
        body = {
            "status": "ready" if current else "unready",
            "schema_version": max(versions) if versions else None,
            "expected_schema_version": SCHEMA_VERSION,
        }
        return (200 if current else 503), body, "application/json"

    # --------------------------------------------------------------- metrics

    def _metrics(self) -> str:
        totals = self.store.totals()
        latest = self.store.list_runs(limit=1)
        lines = [
            "# HELP pme_build_info Static build information.",
            "# TYPE pme_build_info gauge",
            f'pme_build_info{{schema_version="{SCHEMA_VERSION}"}} 1',
            "# HELP pme_uptime_seconds Seconds since the service started.",
            "# TYPE pme_uptime_seconds gauge",
            f"pme_uptime_seconds {time.time() - self.started_at:.3f}",
        ]
        # One HELP and one TYPE line per metric name, then every labelled sample.
        # Repeating the header for each table made the exposition unparseable:
        # a scraper rejects a second HELP line for a name it has already seen.
        lines += [
            "# HELP pme_rows_total Rows stored in each table of the metadata database.",
            "# TYPE pme_rows_total gauge",
            *(f'pme_rows_total{{table="{table}"}} {count}' for table, count in sorted(totals.items())),
        ]
        if latest.items:
            run = latest.items[0]
            consumed = int(run["events_consumed"])
            dropped = int(run["events_dropped_late"])
            rate = dropped / consumed if consumed else 0.0
            # The run id can be chosen by the operator (``--run-id``), so it is
            # caller text and must be escaped: an unescaped newline in a label
            # would let it append arbitrary metric lines to the exposition.
            labels = 'run_id="' + escape_label(str(run["run_id"])) + '"'
            lines += [
                "# HELP pme_last_run_events_consumed Events consumed by the most recent run.",
                "# TYPE pme_last_run_events_consumed gauge",
                f"pme_last_run_events_consumed{{{labels}}} {consumed}",
                "# HELP pme_last_run_events_dropped_late Events the lateness policy refused.",
                "# TYPE pme_last_run_events_dropped_late gauge",
                f"pme_last_run_events_dropped_late{{{labels}}} {dropped}",
                "# HELP pme_last_run_drop_rate Share of the most recent run dropped as too-late.",
                "# TYPE pme_last_run_drop_rate gauge",
                f"pme_last_run_drop_rate{{{labels}}} {rate:.6f}",
                "# HELP pme_last_run_alerts_emitted Alerts emitted by the most recent run.",
                "# TYPE pme_last_run_alerts_emitted gauge",
                f"pme_last_run_alerts_emitted{{{labels}}} {int(run['alerts_emitted'])}",
            ]
        rendered = self.registry.render()
        if rendered:
            lines.append(rendered)
        return "\n".join(lines) + "\n"


class _Handler(BaseHTTPRequestHandler):
    """Thin adapter: HTTP in, :class:`Api` out."""

    server_version = "pme"
    sys_version = ""
    api: Api

    def do_GET(self) -> None:
        self._serve()

    def do_HEAD(self) -> None:
        self._serve(body=False)

    def do_POST(self) -> None:
        self._serve()

    def _serve(self, body: bool = True) -> None:
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        incoming = self.headers.get("X-Request-Id", "")
        cid = incoming[:64] if incoming else new_correlation_id()
        started = time.monotonic()
        with correlation_scope(cid):
            status, payload, content_type = self.api.handle(
                self.command, parsed.path, params,
                self.headers.get("Authorization"),
                self.client_address[0] if self.client_address else "unknown",
            )
            raw = (payload if isinstance(payload, str) else json.dumps(payload, sort_keys=True, default=str))
            encoded = raw.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("X-Request-Id", cid)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if body:
                self.wfile.write(encoded)
            # The Authorization header is never logged, only whether one was sent.
            log.info("request", method=self.command, path=parsed.path, status=status,
                     duration_ms=round((time.monotonic() - started) * 1000, 2),
                     authenticated=bool(self.headers.get("Authorization")))

    def log_message(self, fmt: str, *args: Any) -> None:
        """Silence the stdlib's stderr logging; requests are logged structurally."""


def build_server(store: MetadataStore, host: str = "127.0.0.1", port: int = 8080,
                 token: str | None = None) -> ThreadingHTTPServer:
    """Wire an :class:`Api` onto a threading HTTP server, without starting it."""
    api = Api(store, token=os.environ.get(TOKEN_ENV, "") if token is None else token)
    handler = type("PmeHandler", (_Handler,), {"api": api})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server


def serve(paths: OutputPaths, host: str = "127.0.0.1", port: int = 8080) -> int:
    """Run the service until interrupted. Used by ``pme serve``."""
    configure_logging(level=log_level_from_env(), fmt="json")
    store = MetadataStore(open_database(Path(paths.database)))
    server = build_server(store, host=host, port=port)
    bound_host, bound_port = server.server_address[:2]
    log.info("service listening", host=str(bound_host), port=bound_port,
             database=str(paths.database), auth=bool(os.environ.get(TOKEN_ENV, "")))
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover - interactive
        log.info("service stopping")
    finally:
        server.server_close()
        store.db.close()
    return EXIT_OK


__all__ = ["API_PREFIX", "Api", "RateLimiter", "build_server", "route_of", "serve"]
