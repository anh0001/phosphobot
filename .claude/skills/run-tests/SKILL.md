---
name: run-tests
description: Use when the user asks to run tests, verify changes, check types, or validate code in phosphobot. Covers Python unit tests, API integration tests, frontend tests, mypy type checking, and ruff import sorting. Fires on phrases like "run tests", "check types", "verify it works".
---

# Running tests in phosphobot

All commands run from the repo root.

## Commands

- **Python unit tests:** `make tests` → `cd phosphobot && uv run pytest tests/phosphobot/ -n 5`
- **API integration tests:** need a running server first
  1. `make test_server` (starts server in background)
  2. `cd phosphobot && uv run pytest tests/api/ -s -v`
- **Frontend tests:** `cd dashboard && npm run test`
- **Type checking:** `make types` (mypy strict mode)
- **Import sorting:** `make sort` (ruff)

## Gotchas

- API tests fail silently if the test server isn't up — always start `make test_server` first and wait for it.
- `make types` uses strict flags (`--disallow-untyped-defs`). A new function without annotations will fail CI even if tests pass.
- Hardware driver tests may require physical hardware; check the test file's fixtures before assuming a failure is real.
- `uv run pytest -n 5` runs parallel — tests must be isolated, no shared global state.

## When done

Report pass/fail per suite. If types or sort fail, fix before declaring done — CI runs both on every PR ([.github/workflows/mypy_tests.yml](../../../.github/workflows/mypy_tests.yml), [.github/workflows/pytest_tests.yml](../../../.github/workflows/pytest_tests.yml)).
