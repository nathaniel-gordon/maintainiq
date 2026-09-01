# MaintainIQ — predictive maintenance as an event-time stream

MaintainIQ is a **reference implementation** of a stream processor for machine
telemetry. It reads hourly sensor readings, groups them into event-time windows
behind a watermark, scores each closed window for failure risk and remaining
useful life, raises de-duplicated maintenance alerts, and ranks the resulting
work orders against a cost model. It checkpoints its whole state and can be
resumed mid-stream.

It runs on synthetic telemetry from the simulator in this repository. It is not
deployed anywhere, it has never seen plant data, and nothing here is calibrated
against real failure rates.

## Topology

```
CSV / synthetic source
        │  delivery order ≠ event-time order
        ▼
  watermark  (max event time seen − out-of-orderness budget, monotonic)
        │
        ▼
  windowing engine ── reorder buffer (heap, keyed by event time)
        │           ├─ sliding windows → one trigger per released event
        │           ├─ tumbling windows → fleet-health roll-ups
        │           └─ too-late events → LateEvent side output
        ▼
  operators   features (incremental per-machine stats, O(1) per event)
        │  →  scoring  (micro-batched inference, FIFO)
        │  →  alerting (per machine+severity cooldown, in event time)
        ▼
  sink fan-out, dispatched on each sink's declared record types
        alerts.jsonl · schedule.csv · stream_metrics.json · late.jsonl · sqlite
```

Event time is the only clock the pipeline trusts. Nothing depends on when the
process runs, so a replay of a month of history produces exactly what the live
run would have produced.

### Watermarks, lateness and what "correct" means here

* The **watermark** is `max event time seen − max_out_of_orderness_h`, clamped so
  it never moves backwards.
* An event is **released** once the watermark has passed its window's close, plus
  `allowed_lateness_h`. Each window fires exactly once.
* An event that arrives after that point is **too late**: it does not enter a
  window, and it leaves the engine as a `LateEvent` written to `late.jsonl` and
  to the database, counted per machine.
* Because a window is held open for the whole budget and fires once, any delivery
  order that keeps every event inside the budget produces the same output as a
  perfectly ordered stream. `tests/test_daemon.py` exercises this by cutting one
  stream at several points and comparing the resumed run's alert feed and
  schedule byte for byte against an uninterrupted run.

**Known limitation:** `allowed_lateness_h` is currently only a firing delay. A
window that has already fired does not re-open and re-emit when a late event
arrives, so `(out_of_orderness=6, lateness=2)` behaves identically to
`(out_of_orderness=8, lateness=0)`. Real allowed-lateness semantics — keeping a
fired window's state alive and emitting an update — are not implemented.

## Running it

The package lives under `src/`. Either run the project directory directly, which
needs no install:

```bash
python . demo                 # seeded end-to-end run: train, stream, crash, resume, report
python . run --data telemetry.csv --models output/models.joblib
```

or install it and use the module/console entry points:

```bash
pip install -e .
python -m pme demo
pme --version
```

### Subcommands

| command | what it does |
| --- | --- |
| `demo` | synthetic fleet → train → live stream → mid-stream checkpoint/restore proof → report + chart |
| `train --data X.csv --out models.joblib` | fit the bundle by replaying a historian export through the same operators the daemon uses |
| `run --data X.csv --models models.joblib` | consume a telemetry CSV until it drains |
| `resume --data X.csv --models models.joblib` | restore from the checkpoint and finish the same stream |
| `serve --output DIR` | HTTP health, metrics and run history over the run database |

Common options are attached to each subcommand, so `python . run --data x
--output y` works in any order. `--log-level` and `--log-format json` control
logging; every log line carries a correlation id.

### Reading a telemetry CSV

Required columns: `machine_id`, `hour`, and the five sensors
(`vibration_mm_s`, `temp_c`, `pressure_bar`, `rpm`, `oil_quality_pct`).
`load_factor` and `failure` are optional.

Rows are coerced, range-checked and de-duplicated before anything is
constructed; bad rows are quarantined with a reason code rather than failing the
run, and the file is rejected outright if too many rows are unusable.

Delivery order defaults to **event time** (`--order event_time`). A historian
export written machine-major is not a delivery order, and treating it as one
makes the watermark discard most of the file. Use `--order file` when the row
order really is the order a broker delivered in, and `--jitter N` to re-deliver
rows up to N hours out of order on purpose.

A run whose late-drop rate exceeds `--max-drop-rate` (default 1→) exits non-zero
with `drop_rate_exceeded`, having written its output.

## HTTP surface

```bash
python . serve --output ./output --port 8080
```

| endpoint | |
| --- | --- |
| `GET /healthz` | process is up; no dependencies touched |
| `GET /readyz` | database opens and the schema is current (503 otherwise) |
| `GET /metrics` | Prometheus text exposition |
| `GET /api/v1/runs` | run history, offset-paginated |
| `GET /api/v1/runs/{id}` | one run, with alert counts and its latest checkpoint |
| `GET /api/v1/runs/{id}/alerts` | cursor-paginated; `severity` and `machine_id` filters |
| `GET /api/v1/runs/{id}/machines` | latest scored state per machine |
| `GET /api/v1/runs/{id}/late-events` | what the lateness policy refused |
| `GET /api/v1/quarantine` | rows rejected at the source boundary |

Errors are always `{"error": {"code", "message", "details"}}`. Set
`PME_API_TOKEN` to require a bearer token on `/api/v1`; the probes stay open so
they need no credential. `/api/v1` is rate limited per client address.

## Storage

* **Checkpoints** are JSON, not pickle, written atomically and digest-checked on
  read. A restore is refused unless the semantic config, the source name, its
  length and a fingerprint over the delivered event sequence all match.
* **SQLite** (WAL, foreign keys on, forward migrations) holds runs, alerts,
  machine state, late events, quarantined rows and checkpoint records. Every
  statement is parameterised.
* **Model bundles** are joblib, which is pickle. `save` writes a `sha256`
  manifest beside the bundle and `load` refuses a bundle without a matching one.
  That catches edits and truncation. It does not make a bundle from an untrusted
  source safe — a `.joblib` is executable code, so treat it as one.

## Configuration

`.env.example` lists the environment variables (`PME_API_TOKEN`,
`PME_LOG_LEVEL`). No secrets are committed.

Semantic settings — window sizes, watermark budget, lateness, alert bands, crew
capacity, the cost model — live in `StreamConfig` and are written into every
checkpoint, so a resume under changed semantics is refused rather than silently
producing different results.

## Container

```bash
docker build -t maintainiq:local .
docker compose up            # serves on 127.0.0.1:8080 over a named volume
```

State lives under `/data`; the container runs as a non-root user.

## Observability

Logs are structured JSON (`--log-format json`, the container default): one
object per line, level, logger, message, and a correlation id that ties every
record of a run or request together. The bearer token never appears in a log
line — there is a regression test for that.

Metrics are Prometheus text at `GET /metrics`, or one-shot via
`python . metrics` for a process without the server. The series that actually
emit:

| series | type |
| --- | --- |
| `pme_build_info{schema_version}` | gauge |
| `pme_uptime_seconds` | gauge |
| `pme_rows_total{table}` | gauge, rows per database table |
| `pme_http_requests_total{method,route,status}` | counter |
| `pme_http_request_duration_seconds{route}` | histogram |

Liveness is `/healthz` (touches nothing), readiness is `/readyz` (opens the
database, checks the schema version, 503 otherwise). The Dockerfile's
HEALTHCHECK uses `/healthz`.

## Tests

```bash
python -m pytest tests -q
```

Covers the windowing engine and its late side output, window geometry and the
incremental aggregates against direct numpy recomputation, checkpoint
round-trips and every refusal path, telemetry validation, the SQLite store's
constraints and pagination, the HTTP routes including auth and rate limiting,
and end-to-end restore reproducing an uninterrupted run byte for byte.

Measured on this machine (Windows, Python 3.14):

* `python -m pytest` — **395 passed** in ~73 s
* `python -m pytest --cov=src/pme --cov-report=term` — **96 →** line coverage
  (2792 statements, 99 missed)
* `python -m ruff check .` and `python -m mypy` — clean over 46 source files

CI (`.github/workflows/ci.yml`) runs the same three commands, then `python .
demo` with a clean-tree check, then builds the image and curls its probes.

```bash
python -m ruff check .
python -m mypy
```

Model-quality numbers are **not benchmarked** beyond what `demo` prints for
its own synthetic fleet; no figure from the simulator transfers to real data.

## Limitations

* Synthetic telemetry only. The simulator's failure signal is a monotone function
  of one latent wear variable, so it is far more separable than real plant data;
  model metrics from `demo` say nothing about real-world accuracy.
* Probabilities are **not calibrated** — no isotonic/Platt step, no reliability
  check. They are raw `predict_proba` outputs.
* The scheduler is a **greedy cost ranking** under a crew cap, not an optimiser.
* Sources are bounded and fully materialised in memory, and `__len__` is part of
  the source port, so an unbounded broker stream is not expressible yet.
* Keyed operator state is never evicted, so it grows with the number of distinct
  machine ids, and checkpoint size grows with fleet size.
* `allowed_lateness_h` is a firing delay only (see above).
* Nothing is deployed; there is no live URL and no production usage.

## Roadmap

Roughly in the order they earn their keep:

* Real allowed-lateness semantics — keep a fired window's state alive and emit
  versioned updates instead of treating lateness as a firing delay.
* An unbounded source port (drop `__len__`, add periodic checkpoint triggers)
  so a long-lived feed is expressible.
* State eviction for machines that go quiet, to bound keyed-operator state.
* Probability calibration with a reliability check in the report.

## Contributing, security, license

Setup, architecture notes and coding standards are in
[CONTRIBUTING.md](CONTRIBUTING.md). Vulnerability reporting and the honest
security boundary are in [SECURITY.md](SECURITY.md). Deployment (single Docker
host, and why) is in [docs/deployment.md](docs/deployment.md); the decisions
behind the design are recorded in [docs/adr/](docs/adr/). MIT licensed — see
[LICENSE](LICENSE).

---

## 👤 Author & Contact

- **Author**: Nathaniel Gordon
- **Role**: Senior AI & Machine Learning Engineer
- **GitHub**: [github.com/nathaniel-gordon](https://github.com/nathaniel-gordon)
- **Portfolio / Upwork**: [upwork.com/freelancers/~015fe5a704f8943797](https://www.upwork.com/freelancers/~015fe5a704f8943797)
- **Email**: nathanielgordon346@gmail.com
- **Location**: Tallahassee, FL, USA
