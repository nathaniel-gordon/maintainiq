# 5. Model bundles ship with a sha256 manifest, and load refuses without one

Status: accepted

## Context

Trained models are persisted with joblib, which is pickle: loading one
executes whatever the file says. scikit-learn offers no complete non-pickle
serialisation for the estimators used here, so dropping joblib entirely was
not on the table without dropping sklearn.

## Problem

Reduce the ways a model bundle can hurt a run, given that the format is
inherently executable.

## Options

1. **Plain joblib** — status quo; a truncated or edited file either crashes
   mid-run or silently loads something else.
2. **Re-implement inference without pickle** (export coefficients to JSON) —
   safe, but forks from sklearn's predict path and breaks the moment the model
   class changes.
3. **Keep joblib, write a manifest at save time** — sha256 over the bundle
   bytes plus schema metadata, stored beside the file; `load` refuses a bundle
   whose manifest is missing or whose digest does not match.

## Decision

Option 3, with the limitation stated wherever bundles are documented: the
manifest catches corruption, truncation and casual tampering, and it makes
"which bundle produced this run" answerable; it does **not** make a bundle
from an untrusted source safe to load.

## Reasoning

The realistic failure here is operational, not adversarial: a half-copied
file on a volume, or the wrong bundle paired with the wrong run. A digest
check turns both into an immediate, named error. Pretending to solve the
adversarial case (an attacker who writes the bundle can write the manifest)
would be a false claim, so the docs say plainly what the check does and does
not cover.

## Trade-offs and consequences

* Two files travel together; a bundle without its manifest is unusable by
  design, which is mildly annoying and entirely the point.
* Anyone loading third-party bundles is on their own and the README says so.
* If sklearn grows a trustworthy non-pickle format, option 2 becomes worth
  revisiting.
