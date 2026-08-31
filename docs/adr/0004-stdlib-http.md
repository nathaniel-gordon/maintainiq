# 4. The HTTP service is stdlib http.server, read-only

Status: accepted

## Context

Operators need health/readiness probes, Prometheus metrics, and a way to read
run history, alerts and machine state without opening the SQLite file by hand.
The environment constraint is explicit: no FastAPI/uvicorn class dependencies;
the runtime stack is the stdlib plus the scientific packages.

## Problem

Serve an HTTP API without adding a web framework.

## Options

1. **FastAPI/Flask** — nicer routing and validation, but a large dependency
   tree for what is a read-only reporting surface, and outside the project's
   dependency budget.
2. **No HTTP at all, CLI only** — no probe surface, so the container cannot
   express liveness/readiness and metrics scraping needs file access.
3. **`http.server.ThreadingHTTPServer` with hand-rolled routing.**

## Decision

Option 3. A threaded stdlib server, one sqlite connection per thread, explicit
route table, uniform `{"error": {code, message, details}}` errors, bearer-token
auth and per-address rate limiting on `/api/v1`, probes left open. The API is
strictly read-only — every mutation goes through the daemon and the store
layer, never through HTTP.

## Reasoning

The endpoint surface is small and stable enough that a framework buys little:
pagination, filtering and auth are a few hundred lines that mypy checks like
everything else. Read-only scope keeps the threat model narrow — no request
can change state, so the worst an unauthenticated probe hit can do is read a
health flag.

## Trade-offs and consequences

* No TLS, no HTTP/2, minimal request parsing; anything internet-facing needs a
  reverse proxy in front (see docs/deployment.md).
* Routing and validation are manual, so route tests carry the weight a
  framework's contract would; the service test module covers auth, rate
  limiting, pagination and every error shape.
* `http.server` is not built for high concurrency. For a read-only reporting
  API on one host this is fine; it is stated as a limitation, not hidden.
