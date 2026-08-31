# Deployment

**Nothing is deployed today.** There is no live URL, no hosted instance and no
production usage. This document describes how *you* would put the reference
implementation on a host, and it has been exercised only as far as the steps
say — build the image, start it locally, hit the probes.

## Target: a single Docker host with compose

Chosen deliberately. The system is one process over a single-writer SQLite
file; there is nothing to scale horizontally and no second service to
orchestrate, so Kubernetes or a managed platform would add moving parts
without adding capability. A VM (or bare host) running Docker with the
provided `compose.yaml` matches the architecture exactly: one container, one
named volume, loopback-bound port with a reverse proxy in front if it must be
reachable from elsewhere.

## Prerequisites

* Docker Engine with the compose plugin (the image was built and probed here
  with Docker 29 / Compose v5; anything reasonably current works).
* Roughly 1 GB of disk for the image plus whatever telemetry, checkpoints and
  the database grow to on the volume.
* Outbound network only for the build (pip). The running container needs none.

## Environment variables

Documented in `.env.example`; compose reads a `.env` beside `compose.yaml`.

| variable | default | meaning |
| --- | --- | --- |
| `PME_API_TOKEN` | empty | Bearer token required on `/api/v1`. Empty means the API is open — acceptable only on a loopback bind. Generate one: `python -c "import secrets; print(secrets.token_urlsafe(32))"` |
| `PME_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING` or `ERROR` for every entry point. |

The container's CMD passes `--log-format json` on the command line (there is
no environment variable for the format), so container stdout is one JSON
object per line with a correlation id on every record.

## Migrations

There is no separate migration step. The SQLite store applies its
forward-only numbered migrations when it opens the database, and `/readyz`
reports 503 until the schema version matches what the code expects. Rollback
of a migration is not supported — see Rollback below.

## Build and deploy

```bash
git clone <repo> && cd maintainiq
cp .env.example .env          # set PME_API_TOKEN
docker compose up -d --build
```

Verify:

```bash
curl -fsS http://127.0.0.1:8080/healthz     # {"status": "ok", ...}
curl -fsS http://127.0.0.1:8080/readyz      # {"status": "ready", "schema_version": ...}
curl -fsS http://127.0.0.1:8080/metrics | head
```

The compose file publishes `127.0.0.1:8080` only. To expose it beyond the
host, put nginx/Caddy in front for TLS — the service itself speaks plain HTTP
and always will (see `docs/adr/0004-stdlib-http.md`).

Processing runs are one-shot commands against the same image and volume:

```bash
docker compose run --rm api python -m pme run \
    --data /data/telemetry.csv --models /data/models.joblib --output /data
```

Copy telemetry in with `docker cp` or bind-mount a drop directory; the
container's HEALTHCHECK covers the serving process, not batch runs.

## Health checks

* `GET /healthz` — process is up; touches nothing. This is the Docker
  HEALTHCHECK.
* `GET /readyz` — opens the database and checks the schema version; 503 with
  a reason otherwise. Gate traffic on this one.
* `GET /metrics` — Prometheus text exposition (see the README's
  observability section for the emitted series).

## Rollback

Code and schema roll back differently:

1. **Code only** (schema unchanged): `git checkout <previous tag>`,
   `docker compose up -d --build`. The volume is untouched.
2. **After a migration**: migrations are forward-only, so an old binary
   against a new schema will refuse readiness. Restore the database file from
   the backup you took before deploying: stop the container, replace
   `maintainiq.db` (and its `-wal`/`-shm` siblings) on the volume, start the
   old image. Take that backup every time — the image has no sqlite CLI, so
   use the stdlib backup API against the live file:
   `docker compose exec api python -c "import sqlite3; s=sqlite3.connect('/data/maintainiq.db'); d=sqlite3.connect('/data/pre-deploy.db'); s.backup(d); d.close(); s.close()"`
   (or just copy the files while the container is stopped).
3. Checkpoints refuse a resume across semantic-config changes by design; a
   rollback that changes `StreamConfig` means affected runs restart from
   their source rather than resuming.

## Troubleshooting

* **`/readyz` 503, `schema_version` mismatch** — the image and the database
  disagree; you are running an old image against a migrated database. Roll
  the image forward or restore the pre-deploy backup.
* **401 from `/api/v1`** — `PME_API_TOKEN` is set and the request lacks the
  matching `Authorization: Bearer` header. The token is never logged, so
  check the value in `.env`, not the logs.
* **429 from `/api/v1`** — per-address rate limit; back off. Probes are not
  rate limited.
* **run exits non-zero with `drop_rate_exceeded`** — more than
  `--max-drop-rate` of events were too late for the watermark. Usually the
  file was fed with `--order file` when it is not in delivery order; use the
  default event-time ordering, or raise the out-of-orderness budget.
* **`checkpoint_incompatible` on resume** — config, source or data changed
  since the checkpoint. Intentional; start a fresh run.
* **Database locked** — something else has the SQLite file open (a second
  daemon, a shell). One writer per database file is the supported topology.
