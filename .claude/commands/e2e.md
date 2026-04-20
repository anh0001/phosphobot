---
description: Legacy slash-entry shim for the e2e-testing skill. Prefer the skill directly.
---

# E2E Command

Use this when you want end-to-end coverage for user-facing flows in `dashboard/`, `dataset-viewer/`, or `cloud/frontend/`.

## Canonical Surface

- Prefer the `e2e-testing` skill directly.
- Keep this file as a lightweight compatibility entry point.

## Arguments

`$ARGUMENTS`

## Delegation

Apply the `e2e-testing` skill.
- Generate or update E2E coverage only for the requested flow.
- Reuse existing framework/tooling in the touched app instead of inventing a new harness.
- Capture the usual artifacts and report failures, flake risk, and next fixes.
- Keep browser coverage secondary to repo-safe verification: if hardware behavior is involved, validate simulation/backend behavior separately.
