"""Queries over the metadata database.

One module owns every statement so the SQL is reviewable in one place and the
rest of the codebase never builds a query. Values are always bound; the only
interpolated fragments are fixed literals chosen from a whitelist (a sort
direction, a column name), never caller text.

Lists are paginated because every one of these tables grows with the stream. The
alert and late-event feeds are append-only with a monotonic id, so they page by
keyset (``after``) which stays correct while rows are being inserted; the small
summary tables page by offset.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..errors import ValidationError
from .db import Database, utc_now

#: Page sizes. A caller may ask for fewer, never for more.
DEFAULT_PAGE = 50
MAX_PAGE = 500

_ALERT_SEVERITIES = frozenset({"CRITICAL", "WARNING", "ANOMALY"})
_RUN_STATUSES = frozenset({"running", "completed", "stopped", "failed"})


@dataclass(frozen=True, slots=True)
class Page:
    """One page of rows plus the cursor that fetches the next one."""

    items: list[dict[str, Any]]
    next_cursor: str | None = None
    total: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"items": self.items, "count": len(self.items)}
        if self.total is not None:
            payload["total"] = self.total
        payload["next_cursor"] = self.next_cursor
        return payload


def clamp_limit(limit: int | None) -> int:
    if limit is None:
        return DEFAULT_PAGE
    if limit < 1:
        raise ValidationError("limit must be >= 1", limit=limit)
    return min(limit, MAX_PAGE)


def _rows(result: list[Any]) -> list[dict[str, Any]]:
    return [dict(r) for r in result]


class MetadataStore:
    """Everything the daemon and the HTTP service persist outside the checkpoint."""

    def __init__(self, db: Database) -> None:
        self.db = db

    # ------------------------------------------------------------------ runs

    def start_run(
        self,
        run_id: str,
        source_name: str,
        source_length: int,
        source_fingerprint: str,
        config: dict[str, Any],
        correlation_id: str = "",
    ) -> None:
        """Register a run as ``running``. Re-registering the same id is a resume."""
        with self.db.transaction():
            self.db.execute(
                "INSERT INTO runs (run_id, source_name, source_length, source_fingerprint,"
                " config_json, status, started_at, correlation_id)"
                " VALUES (?, ?, ?, ?, ?, 'running', ?, ?)"
                " ON CONFLICT (run_id) DO UPDATE SET status = 'running', finished_at = NULL,"
                " correlation_id = excluded.correlation_id",
                (run_id, source_name, source_length, source_fingerprint,
                 json.dumps(config, sort_keys=True), utc_now(), correlation_id),
            )

    def finish_run(
        self,
        run_id: str,
        status: str,
        events_consumed: int,
        events_dropped_late: int,
        alerts_emitted: int,
    ) -> None:
        if status not in _RUN_STATUSES:
            raise ValidationError("unknown run status", status=status)
        with self.db.transaction():
            self.db.execute(
                "UPDATE runs SET status = ?, finished_at = ?, events_consumed = ?,"
                " events_dropped_late = ?, alerts_emitted = ? WHERE run_id = ?",
                (status, utc_now(), events_consumed, events_dropped_late, alerts_emitted, run_id),
            )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        row = self.db.query_one("SELECT * FROM runs WHERE run_id = ?", (run_id,))
        return dict(row) if row is not None else None

    def list_runs(self, limit: int | None = None, offset: int = 0, status: str | None = None) -> Page:
        limit = clamp_limit(limit)
        if offset < 0:
            raise ValidationError("offset must be >= 0", offset=offset)
        if status is not None and status not in _RUN_STATUSES:
            raise ValidationError("unknown run status", status=status)
        where, params = ("WHERE status = ?", [status]) if status else ("", [])
        total_row = self.db.query_one(f"SELECT COUNT(*) AS n FROM runs {where}", params)
        rows = self.db.query(
            f"SELECT * FROM runs {where} ORDER BY started_at DESC, run_id DESC LIMIT ? OFFSET ?",
            [*params, limit, offset],
        )
        total = int(total_row["n"]) if total_row else 0
        nxt = str(offset + limit) if offset + limit < total else None
        return Page(items=_rows(rows), next_cursor=nxt, total=total)

    # ---------------------------------------------------------------- alerts

    def record_alerts(self, run_id: str, alerts: list[dict[str, Any]], correlation_id: str = "") -> int:
        """Append alerts. Replaying the same window after a restore is a no-op."""
        if not alerts:
            return 0
        rows = [
            (run_id, a["machine_id"], int(a["event_time"]), a["severity"], float(a["risk_prob"]),
             max(float(a["rul_est_h"]), 0.0), int(a["anomaly"]), a["reason"], correlation_id)
            for a in alerts
        ]
        with self.db.transaction():
            self.db.executemany(
                "INSERT OR IGNORE INTO alerts (run_id, machine_id, event_time, severity, risk_prob,"
                " rul_est_h, anomaly, reason, correlation_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
        return len(rows)

    def list_alerts(
        self,
        run_id: str,
        limit: int | None = None,
        after: int | None = None,
        severity: str | None = None,
        machine_id: str | None = None,
    ) -> Page:
        limit = clamp_limit(limit)
        if severity is not None and severity not in _ALERT_SEVERITIES:
            raise ValidationError("unknown severity", severity=severity)
        clauses = ["run_id = ?"]
        params: list[Any] = [run_id]
        if after is not None:
            clauses.append("id > ?")
            params.append(after)
        if severity is not None:
            clauses.append("severity = ?")
            params.append(severity)
        if machine_id is not None:
            clauses.append("machine_id = ?")
            params.append(machine_id)
        where = " AND ".join(clauses)
        rows = self.db.query(
            f"SELECT * FROM alerts WHERE {where} ORDER BY id LIMIT ?", [*params, limit + 1]
        )
        items = _rows(rows[:limit])
        nxt = str(items[-1]["id"]) if len(rows) > limit and items else None
        return Page(items=items, next_cursor=nxt)

    def count_alerts_by_severity(self, run_id: str) -> dict[str, int]:
        rows = self.db.query(
            "SELECT severity, COUNT(*) AS n FROM alerts WHERE run_id = ? GROUP BY severity", (run_id,)
        )
        return {str(r["severity"]): int(r["n"]) for r in rows}

    # --------------------------------------------------------- machine state

    def upsert_machine_state(self, run_id: str, records: list[dict[str, Any]]) -> None:
        """Last-write-wins on ``(run_id, machine_id)``, but never backwards in event time."""
        if not records:
            return
        rows = [
            (run_id, r["machine_id"], int(r["event_time"]), float(r["risk_prob"]),
             max(float(r["rul_est_h"]), 0.0), int(r["anomaly"]), utc_now())
            for r in records
        ]
        with self.db.transaction():
            self.db.executemany(
                "INSERT INTO machine_state (run_id, machine_id, event_time, risk_prob, rul_est_h,"
                " anomaly, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT (run_id, machine_id) DO UPDATE SET"
                " event_time = excluded.event_time, risk_prob = excluded.risk_prob,"
                " rul_est_h = excluded.rul_est_h, anomaly = excluded.anomaly,"
                " updated_at = excluded.updated_at"
                " WHERE excluded.event_time >= machine_state.event_time",
                rows,
            )

    def list_machine_state(self, run_id: str, limit: int | None = None, offset: int = 0) -> Page:
        limit = clamp_limit(limit)
        if offset < 0:
            raise ValidationError("offset must be >= 0", offset=offset)
        total_row = self.db.query_one(
            "SELECT COUNT(*) AS n FROM machine_state WHERE run_id = ?", (run_id,)
        )
        rows = self.db.query(
            "SELECT * FROM machine_state WHERE run_id = ?"
            " ORDER BY risk_prob DESC, machine_id LIMIT ? OFFSET ?",
            (run_id, limit, offset),
        )
        total = int(total_row["n"]) if total_row else 0
        nxt = str(offset + limit) if offset + limit < total else None
        return Page(items=_rows(rows), next_cursor=nxt, total=total)

    # ----------------------------------------------------------- late events

    def record_late_events(self, run_id: str, events: list[dict[str, Any]]) -> None:
        if not events:
            return
        rows = [
            (run_id, e["machine_id"], int(e["event_time"]), int(e["watermark"]), int(e["lateness_h"]))
            for e in events
        ]
        with self.db.transaction():
            self.db.executemany(
                "INSERT INTO late_events (run_id, machine_id, event_time, watermark, lateness_h)"
                " VALUES (?, ?, ?, ?, ?)",
                rows,
            )

    def list_late_events(self, run_id: str, limit: int | None = None, after: int | None = None) -> Page:
        limit = clamp_limit(limit)
        clauses = ["run_id = ?"]
        params: list[Any] = [run_id]
        if after is not None:
            clauses.append("id > ?")
            params.append(after)
        where = " AND ".join(clauses)
        rows = self.db.query(f"SELECT * FROM late_events WHERE {where} ORDER BY id LIMIT ?", [*params, limit + 1])
        items = _rows(rows[:limit])
        nxt = str(items[-1]["id"]) if len(rows) > limit and items else None
        return Page(items=items, next_cursor=nxt)

    # ------------------------------------------------------------ quarantine

    def quarantine_rows(self, source: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        payload = [
            (source, int(r["row_number"]), r["code"], r["detail"], json.dumps(r["payload"], default=str), utc_now())
            for r in rows
        ]
        with self.db.transaction():
            self.db.executemany(
                "INSERT INTO quarantine (source, row_number, code, detail, payload_json, recorded_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                payload,
            )

    def list_quarantine(self, source: str | None = None, limit: int | None = None, after: int | None = None) -> Page:
        limit = clamp_limit(limit)
        clauses: list[str] = []
        params: list[Any] = []
        if source is not None:
            clauses.append("source = ?")
            params.append(source)
        if after is not None:
            clauses.append("id > ?")
            params.append(after)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.db.query(f"SELECT * FROM quarantine {where} ORDER BY id LIMIT ?", [*params, limit + 1])
        items = _rows(rows[:limit])
        nxt = str(items[-1]["id"]) if len(rows) > limit and items else None
        return Page(items=items, next_cursor=nxt)

    # ----------------------------------------------------------- checkpoints

    def record_checkpoint(
        self, run_id: str, cursor: int, events_consumed: int, version: int, sha256: str, path: str
    ) -> None:
        with self.db.transaction():
            self.db.execute(
                "INSERT INTO checkpoints (run_id, cursor, events_consumed, version, sha256, path, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (run_id, cursor, events_consumed, version, sha256, path, utc_now()),
            )

    def latest_checkpoint(self, run_id: str) -> dict[str, Any] | None:
        row = self.db.query_one(
            "SELECT * FROM checkpoints WHERE run_id = ? ORDER BY id DESC LIMIT 1", (run_id,)
        )
        return dict(row) if row is not None else None

    # -------------------------------------------------------------- counters

    def totals(self) -> dict[str, int]:
        """Row counts the metrics endpoint exposes as gauges."""
        out: dict[str, int] = {}
        for table in ("runs", "alerts", "machine_state", "late_events", "quarantine", "checkpoints"):
            row = self.db.query_one(f"SELECT COUNT(*) AS n FROM {table}")  # fixed literal, not caller input
            out[table] = int(row["n"]) if row else 0
        return out


__all__ = ["DEFAULT_PAGE", "MAX_PAGE", "MetadataStore", "Page", "clamp_limit"]
