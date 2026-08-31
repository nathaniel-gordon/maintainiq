# Contributing

## Setup

```bash
git clone <your fork>
cd maintainiq
python -m venv .venv && . .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install pytest pytest-cov ruff mypy
```

No install of the package itself is needed: `pyproject.toml` puts `src/` on
`pythonpath` for pytest, and `python . <cmd>` runs the project directory
directly. `pip install -e .` also works if you prefer the `pme` entry point.

## Running things

```bash
python . demo                  # full seeded run: train, stream, crash, resume, report
python -m pytest               # the suite; slow end-to-end tests are marked "slow"
python -m pytest -m "not slow" # quicker loop while iterating
python -m ruff check .
python -m mypy
```

All three must be clean before a PR; CI runs exactly these commands plus a
docker build that boots the image and curls its probes.

## Architecture in one paragraph

`src/pme/core` is the event-time engine: watermark, reorder buffer, window
assigners, incremental aggregates. `operators/` (features → scoring →
alerting) transform released events; `sinks/` receive records fan-out style,
dispatched on the record types each sink declares; `sources/` produce
validated telemetry. `runtime/` wires a config plus those ports into a daemon
with JSON checkpoints; `store/` is the SQLite layer (WAL, forward
migrations); `service/` is a read-only stdlib HTTP server over the store.
The dependency direction is inward: core knows nothing about sinks, sources,
sqlite or HTTP. Keep it that way — a new integration should be a new adapter
implementing an existing port, not a change to the engine.

## Coding standards

* Python ≥ 3.11, type-annotated; `mypy --strict` is the gate, not a suggestion.
* Ruff enforces style; the selected rule set is in `pyproject.toml`. Don't add
  per-line ignores without a comment saying why.
* Every SQL statement is parameterised. No exceptions.
* Behaviour that matters gets a test that would fail without the change.
  Determinism claims (ordering, resume, replay) get a byte-for-byte comparison,
  not a spot check.
* No new runtime dependencies without discussion — the stdlib-plus-scientific
  stack is deliberate.

## Commits and PRs

Small, self-contained commits with plain imperative subjects ("fix the
watermark clamp on duplicate timestamps"). Explain *why* in the body only when
the subject doesn't carry it. A PR should say what behaviour changed and how
you verified it.
