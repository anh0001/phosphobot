# Codex Prompt: Feature Development

Use this when implementing a feature or non-trivial fix in `phosphobot`.

## Source Material

- Primary repo guidance: `AGENTS.md`
- Existing Claude workflow: `.claude/commands/feature-development.md`
- Research workflow when external facts matter: `.claude/research/phosphobot-research-playbook.md`

## Working Mode

1. Inspect local context first:
   - read the touched package or app before proposing changes
   - find similar endpoints, drivers, models, tests, or UI patterns
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
5. Summarize:
   - changed behavior
   - verification performed
   - remaining risk or unverified areas
