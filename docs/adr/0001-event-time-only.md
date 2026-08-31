# 1. Event time is the only clock

Status: accepted

## Context

Telemetry arrives out of order: historian exports are machine-major, brokers
re-deliver, and a resumed run replays hours of history in seconds. Any logic
keyed to arrival time or wall-clock time produces different alerts depending
on *when* the pipeline runs, which makes replay testing impossible and makes a
crash-recovery run diverge from the run it replaced.

## Problem

Pick the time semantics for windowing, alert cooldowns and scheduling.

## Options

1. **Processing time** — window on arrival. Simple, no buffering, but output
   depends on delivery order and machine speed; irreproducible.
2. **Ingestion time** — stamp on entry. Same problems one step removed.
3. **Event time behind a watermark** — buffer events, advance a watermark at
   `max event time seen − out_of_orderness budget`, fire each window exactly
   once when the watermark passes its close, shunt later arrivals to a side
   output.

## Decision

Option 3, everywhere. The watermark is monotonic; the engine holds a reorder
buffer (a heap keyed by event time); alert cooldowns and the maintenance
schedule are computed in event time; too-late events leave as `LateEvent`
records, never silently dropped.

## Reasoning

Determinism is the property everything else in this repo leans on: the
crash/resume tests compare a resumed run against an uninterrupted one byte for
byte, and that comparison is only meaningful if delivery order cannot change
the answer. Event time behind a watermark gives exactly that guarantee for any
delivery order within the budget.

## Trade-offs and consequences

* Latency: an event is not visible in output until the watermark passes it,
  so results trail the newest data by the out-of-orderness budget.
* Memory: the reorder buffer and open windows hold state proportional to the
  budget times event rate.
* `allowed_lateness_h` is currently only a firing delay — a fired window does
  not re-open and re-emit. Real retraction/update semantics would require
  keeping fired-window state alive and versioning sink records; documented as
  a limitation instead of half-built.
