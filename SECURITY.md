# Security policy

## Reporting

Email nathanielgordon346@gmail.com with a description and reproduction steps.
Please do not open a public issue for anything exploitable. You should hear
back within a week; this is a single-maintainer project, so there is no
security team and no bounty.

## Supported versions

Only the tip of `main`. There are no maintained release branches.

## What the code does and does not defend

This is a reference implementation, not a hardened service. The honest
boundary:

* `/api/v1` supports bearer-token auth (`PME_API_TOKEN`) and per-address rate
  limiting, and the token is compared with `hmac.compare_digest` and kept out
  of the logs. The probes (`/healthz`, `/readyz`, `/metrics`) are deliberately
  unauthenticated.
* There is no TLS. The server binds loopback by default; anything beyond a
  single trusted host needs a reverse proxy in front (see
  `docs/deployment.md`).
* All SQL is parameterised, and output-directory paths are containment-checked
  before use.
* Model bundles are joblib, i.e. pickle. The sha256 manifest catches
  corruption and tampering after `save`, but a bundle from an untrusted source
  is arbitrary code execution the moment you load it. Never load bundles you
  did not produce.
* Telemetry CSVs are validated, coerced and quarantined row by row, so a
  malformed file degrades a run rather than crashing it — but validation is
  about correctness, not about parsing hostile input from the internet.
