# 2. SQLite is the persistence engine

Status: accepted

## Context

The daemon needs durable run history: runs, alerts, per-machine state, late
events, quarantined rows, checkpoint records. The HTTP service reads the same
data. The system is a single process on a single host, plus short-lived
one-shot commands sharing a data directory.

## Problem

Choose a store that the tests in this repository can actually exercise.

## Options

1. **PostgreSQL** — the conventional choice, but it adds a second process, a
   driver dependency (`psycopg2`), and a test story that either mocks the
   database or requires infrastructure the CI job would have to provision.
2. **JSON/JSONL files only** — zero dependencies, but no constraints, no
   transactional writes across tables, and pagination/filtering in the API
   would mean loading everything.
3. **SQLite via the stdlib** — in-process, transactional, enforces uniqueness
   and foreign keys, and every test runs against the real engine in a temp
   directory.

## Decision

SQLite, WAL mode, `foreign_keys=ON`, forward-only numbered migrations applied
at open, every statement parameterised. JSONL/CSV sinks remain as feed outputs
but are not the system of record.

## Reasoning

The honest test suite was the deciding factor: with SQLite the store tests,
API tests and end-to-end tests all run against the production engine, not a
fake. WAL lets the read-only HTTP threads read while the daemon writes.
Nothing in the access pattern (single writer, small reads) needs a server
database.

## Trade-offs and consequences

* Single-writer: a second concurrent daemon against the same file would
  contend; that topology is out of scope and documented as such.
* Migrations are forward-only; a downgrade means restoring from a copy of the
  database file.
* If a PostgreSQL path is ever added it must come with tests that run it, or
  be labelled untested — the repository does not ship claims its CI does not
  cover.
