---
description: Legacy slash-entry shim for the tdd-workflow skill. Prefer the skill directly.
---

# TDD Command

Use this when you want a test-first workflow for backend, dashboard, or integration changes.

## Canonical Surface

- Prefer the `tdd-workflow` skill directly.
- Keep this file as a lightweight compatibility entry point.

## Arguments

`$ARGUMENTS`

## Delegation

Apply the `tdd-workflow` skill.
- Stay strict on RED -> GREEN -> REFACTOR.
- Use repo-native tests:
  - backend: targeted `pytest` under `phosphobot/tests/`
  - API: `make test_server` plus `phosphobot/tests/api/`
  - dashboard: `npm run test`
- Keep hardware changes simulation-first even when following TDD.
- Use the skill as the maintained TDD body instead of duplicating long examples here.
