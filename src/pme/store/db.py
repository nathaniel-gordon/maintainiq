"""SQLite connection handling and forward migrations.

The stream itself is stateless-by-replay — its durable state is the checkpoint —
but everything *around* the stream needs a queryable record: which runs happened,
what they alerted on, which rows were quarantined, where each machine ended up.
That is a database's job, and SQLite is a real one: constraints are enforced by
the engine, readers do not block the writer under WAL, and the file is the whole
deployment.

Connections are per-thread (SQLite objects are not shareable across threads), so
the HTTP service can serve reads while the daemon writes. Every statement in this
package is parameterised; no SQL is ever built by string interpolation.
"""
from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..errors import StorageError
from ..logging_setup import get_logger

log = get_logger(__name__)

#: In-memory databases are per-connection, so a thread-local pool would hand out
#: separate empty databases. Tests that need one must stay on a single thread.
MEMORY = ":memory:"


def utc_now() -> str:
    """Timestamp format used by every ``*_at`` column."""
    return datetime.now(UTC).isoformat(timespec="seconds")


#: Ordered forward migrations. Each entry is a list of statements applied exactly
#: once, inside one transaction, and recorded in ``schema_migrations``. Statements
#: are listed individually rather than as one script because ``executescript``
#: commits whatever transaction is open before it runs. Never edit a migration
#: that has shipped; append a new one.
#:
#: 0001 is the base schema. 0002 adds the correlation id that arrived with
#: structured logging: a separate step rather than a rewrite of 0001, so the
#: upgrade path is the one that actually runs against an existing database.
MIGRATIONS: tuple[tuple[int, str, tuple[str, ...]], ...] = (
    (
        1,
        "base_schema",
        (
            """
            CREATE TABLE runs (
                run_id             TEXT PRIMARY KEY,
                source_name        TEXT NOT NULL,
                source_length      INTEGER NOT NULL CHECK (source_length >= 0),
                source_fingerprint TEXT NOT NULL,
                config_json        TEXT NOT NULL,
                status             TEXT NOT NULL
                                   CHECK (status IN ('running', 'completed', 'stopped', 'failed')),
                started_at         TEXT NOT NULL,
                finished_at        TEXT,
                events_consumed    INTEGER NOT NULL DEFAULT 0 CHECK (events_consumed >= 0),
                events_dropped_late INTEGER NOT NULL DEFAULT 0 CHECK (events_dropped_late >= 0),
                alerts_emitted     INTEGER NOT NULL DEFAULT 0 CHECK (alerts_emitted >= 0)
            )
            """,
            "CREATE INDEX idx_runs_started ON runs (started_at DESC)",
            """
            CREATE TABLE alerts (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id      TEXT NOT NULL REFERENCES runs (run_id) ON DELETE CASCADE,
                machine_id  TEXT NOT NULL,
                event_time  INTEGER NOT NULL,
                severity    TEXT NOT NULL CHECK (severity IN ('CRITICAL', 'WARNING', 'ANOMALY')),
                risk_prob   REAL NOT NULL CHECK (risk_prob >= 0.0 AND risk_prob <= 1.0),
                rul_est_h   REAL NOT NULL CHECK (rul_est_h >= 0.0),
                anomaly     INTEGER NOT NULL CHECK (anomaly IN (0, 1)),
                reason      TEXT NOT NULL,
                UNIQUE (run_id, machine_id, event_time, severity)
            )
            """,
            "CREATE INDEX idx_alerts_run_time ON alerts (run_id, event_time)",
            "CREATE INDEX idx_alerts_machine ON alerts (run_id, machine_id, event_time)",
            "CREATE INDEX idx_alerts_severity ON alerts (run_id, severity, id)",
            """
            CREATE TABLE machine_state (
                run_id      TEXT NOT NULL REFERENCES runs (run_id) ON DELETE CASCADE,
                machine_id  TEXT NOT NULL,
                event_time  INTEGER NOT NULL,
                risk_prob   REAL NOT NULL CHECK (risk_prob >= 0.0 AND risk_prob <= 1.0),
                rul_est_h   REAL NOT NULL CHECK (rul_est_h >= 0.0),
                anomaly     INTEGER NOT NULL CHECK (anomaly IN (0, 1)),
                updated_at  TEXT NOT NULL,
                PRIMARY KEY (run_id, machine_id)
            )
            """,
            "CREATE INDEX idx_state_risk ON machine_state (run_id, risk_prob DESC)",
            """
            CREATE TABLE late_events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id      TEXT NOT NULL REFERENCES runs (run_id) ON DELETE CASCADE,
                machine_id  TEXT NOT NULL,
                event_time  INTEGER NOT NULL,
                watermark   INTEGER NOT NULL,
                lateness_h  INTEGER NOT NULL CHECK (lateness_h > 0)
            )
            """,
            "CREATE INDEX idx_late_run ON late_events (run_id, id)",
            "CREATE INDEX idx_late_machine ON late_events (run_id, machine_id)",
            """
            CREATE TABLE quarantine (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                source       TEXT NOT NULL,
                row_number   INTEGER NOT NULL CHECK (row_number >= 0),
                code         TEXT NOT NULL,
                detail       TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                recorded_at  TEXT NOT NULL
            )
            """,
            "CREATE INDEX idx_quarantine_source ON quarantine (source, id)",
            """
            CREATE TABLE checkpoints (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id          TEXT NOT NULL REFERENCES runs (run_id) ON DELETE CASCADE,
                cursor          INTEGER NOT NULL CHECK (cursor >= 0),
                events_consumed INTEGER NOT NULL CHECK (events_consumed >= 0),
                version         INTEGER NOT NULL,
                sha256          TEXT NOT NULL,
                path            TEXT NOT NULL,
                created_at      TEXT NOT NULL
            )
            """,
            "CREATE INDEX idx_checkpoints_run ON checkpoints (run_id, id DESC)",
        ),
    ),
    (
        2,
        "correlation_ids",
        (
            "ALTER TABLE runs ADD COLUMN correlation_id TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE alerts ADD COLUMN correlation_id TEXT NOT NULL DEFAULT ''",
        ),
    ),
)

SCHEMA_VERSION = MIGRATIONS[-1][0]


class Database:
    """A migrated SQLite database with per-thread connections."""

    def __init__(self, path: str | Path, timeout: float = 30.0) -> None:
        self.path = MEMORY if str(path) == MEMORY else str(Path(path))
        self.timeout = timeout
        self._local = threading.local()
        self._memory_conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()
        if self.path != MEMORY:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------- connections

    def connection(self) -> sqlite3.Connection:
        """The calling thread's connection, opened and configured on first use."""
        if self.path == MEMORY:
            if self._memory_conn is None:
                self._memory_conn = self._open()
            return self._memory_conn
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._open()
            self._local.conn = conn
        return conn

    def _open(self) -> sqlite3.Connection:
        try:
            conn = sqlite3.connect(self.path, timeout=self.timeout, isolation_level=None)
        except sqlite3.Error as exc:
            raise StorageError("could not open the database", path=self.path, detail=str(exc)) from exc
        conn.row_factory = sqlite3.Row
        if self.path != MEMORY:
            conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute(f"PRAGMA busy_timeout = {int(self.timeout * 1000)}")
        return conn

    def close_thread_connection(self) -> None:
        """Close the connection the calling thread opened, if it opened one.

        The HTTP server gives each client connection its own thread, and a thread
        that has answered a request holds a sqlite handle until the process ends.
        Releasing it at the end of the connection bounds open handles by
        concurrency instead of by total requests served. In-memory databases are
        untouched: their single connection is shared and closing it would discard
        the database.
        """
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def close(self) -> None:
        """Close this thread's connection (and the shared in-memory one, if any)."""
        self.close_thread_connection()
        if self._memory_conn is not None:
            self._memory_conn.close()
            self._memory_conn = None

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---------------------------------------------------------------- access

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """``BEGIN IMMEDIATE`` … commit, or roll back on any exception.

        Immediate rather than deferred: writers take the reserved lock up front,
        so two processes contending for the same file fail fast on the busy
        timeout instead of half-way through a batch.
        """
        conn = self.connection()
        if conn.in_transaction:  # nested call — the outermost block owns the commit
            yield conn
            return
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")

    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        try:
            return self.connection().execute(sql, tuple(params))
        except sqlite3.IntegrityError as exc:
            raise StorageError("constraint violated", detail=str(exc)) from exc
        except sqlite3.Error as exc:
            raise StorageError("sqlite error", detail=str(exc)) from exc

    def executemany(self, sql: str, rows: Sequence[Sequence[Any]]) -> None:
        if not rows:
            return
        try:
            self.connection().executemany(sql, [tuple(r) for r in rows])
        except sqlite3.IntegrityError as exc:
            raise StorageError("constraint violated", detail=str(exc)) from exc
        except sqlite3.Error as exc:
            raise StorageError("sqlite error", detail=str(exc)) from exc

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        return list(self.execute(sql, params).fetchall())

    def query_one(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        rows = self.execute(sql, params).fetchmany(1)
        return rows[0] if rows else None

    # ------------------------------------------------------------ migrations

    def applied_versions(self) -> list[int]:
        row = self.query_one(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'schema_migrations'"
        )
        if row is None:
            return []
        return [int(r["version"]) for r in self.query("SELECT version FROM schema_migrations ORDER BY version")]

    def migrate(self) -> list[int]:
        """Apply every migration this database has not seen. Returns those applied."""
        with self._lock:
            self.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                " version INTEGER PRIMARY KEY,"
                " name TEXT NOT NULL,"
                " applied_at TEXT NOT NULL)"
            )
            done = set(self.applied_versions())
            applied: list[int] = []
            for version, name, statements in MIGRATIONS:
                if version in done:
                    continue
                with self.transaction() as tx:
                    for statement in statements:
                        tx.execute(statement)
                    tx.execute(
                        "INSERT INTO schema_migrations (version, name, applied_at) VALUES (?, ?, ?)",
                        (version, name, utc_now()),
                    )
                applied.append(version)
                log.info("migration applied", version=version, migration=name)
            return applied


def open_database(path: str | Path) -> Database:
    """Open and migrate in one step — the only way the rest of the code gets a db."""
    db = Database(path)
    db.migrate()
    return db


__all__ = ["MEMORY", "MIGRATIONS", "SCHEMA_VERSION", "Database", "open_database", "utc_now"]
