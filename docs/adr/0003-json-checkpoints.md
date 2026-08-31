# 3. Checkpoints are digest-checked JSON, and resume can refuse

Status: accepted

## Context

The daemon checkpoints its full state (watermark, reorder buffer, open
windows, operator state, sink positions) so a crashed run can resume and
produce the same output an uninterrupted run would have. A checkpoint is also
an attractive place for silent corruption: a truncated write, a hand-edit, or
a resume against a *different* configuration or input file.

## Problem

Choose the checkpoint format and decide what a restore is allowed to trust.

## Options

1. **Pickle the runtime objects** — least code, but a pickle is executable on
   load, is unreadable when debugging, and breaks silently across refactors.
2. **JSON, trust the contents** — safe to load, but a resume under changed
   window sizes or a different input file "works" and quietly produces
   different results.
3. **JSON, atomic write, sha256 digest, and explicit compatibility checks
   before restore.**

## Decision

Option 3. The checkpoint is JSON written to a temp file and renamed into
place; a digest over the payload is stored and verified on read. Restore is
refused unless the semantic config (window geometry, watermark budget, alert
bands, cost model), the source name, its length and a fingerprint over the
event sequence delivered so far all match the checkpoint.

## Reasoning

The failure mode worth defending against is not an attacker, it is an
operator resuming with a tweaked config and trusting the merged output. A
loud `checkpoint_incompatible` error is cheaper than debugging a schedule
that is subtly wrong. JSON keeps checkpoints inspectable with any editor,
which has already paid for itself during debugging.

## Trade-offs and consequences

* Serialisation code must be written per stateful component instead of
  falling out of pickle; state added without (de)serialisation shows up as a
  failing round-trip test.
* Any change to semantic config invalidates outstanding checkpoints on
  purpose. Changing alert bands mid-stream means starting the run over.
* Digest checking catches corruption, not tampering by someone who can also
  rewrite the digest; the checkpoint directory carries the same trust as the
  database file.
