---
description: Repo-specific implementation workflow for backend, dashboard, hardware, and packaging work in phosphobot.
argument-hint: [optional feature or scope]
---

# Feature Development

Use this workflow when implementing a feature or non-trivial fix in `phosphobot`.

## Input

`$ARGUMENTS`

## Workflow

1. Inspect local context first:
   - read `AGENTS.md`, `CLAUDE.md`, and the touched package/app
   - find similar endpoints, drivers, models, or UI patterns before editing
2. Pick the smallest correct surface:
   - backend/API in `phosphobot/phosphobot/`
   - robot integration in `phosphobot/phosphobot/hardware/`
   - shipped UI in `dashboard/`
   - separate web apps in `dataset-viewer/` or `cloud/frontend/`
3. Apply repo guardrails:
   - simulation-first for hardware/control changes
   - no manual edits in `phosphobot/resources/dist`
   - avoid unrelated lockfile churn
4. Verify narrowly:
   - backend: targeted `pytest`, then `make types` if needed
   - dashboard: `npm run build`, plus `npm run test` when applicable
   - Next.js apps: `npm run build`
   - packaged dashboard changes: rebuild frontend assets
5. Summarize outcome:
   - changed behavior
   - verification performed
   - remaining risk or unverified areas
