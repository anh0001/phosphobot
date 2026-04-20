---
name: pre-pr
description: Use before opening a PR or when the user says "ready to commit", "prep for PR", "final check". Runs the same gates CI will run so failures surface locally first.
---

# Pre-PR checklist for phosphobot

CI runs mypy + pytest on every PR touching `phosphobot/`. Match that locally.

## Gates (in order)

1. `make sort` — ruff import sort (auto-fix).
2. `make types` — strict mypy. Must pass.
3. `make tests` — Python unit tests.
4. If API touched: `make test_server` then `cd phosphobot && uv run pytest tests/api/ -s -v`.
5. If dashboard touched: `cd dashboard && npm run test` + manual browser check.

## Constraints

- Keep the PR small and focused. Split unrelated cleanup into a separate PR.
- No `--no-verify`, no skipping hooks.
- Don't commit `.env`, credentials, or large binaries. Stage files explicitly, not `git add -A`.
- Commit message style: follow recent history (`git log --oneline -10`). Recent commits use `feat:`, `fix:`, `test:` prefixes.

## Gotchas

- Frontend build artifacts under [phosphobot/resources/dist/](../../../phosphobot/resources/dist/) are committed — rebuild if you changed dashboard source and the artifacts are stale.
- `bullet3/` is a submodule. Don't accidentally commit submodule pointer changes unless intended.

## Output

Report per-gate pass/fail. Do not claim "ready" until every applicable gate is green.
