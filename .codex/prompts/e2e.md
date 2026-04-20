# Codex Prompt: E2E

Use this when you need end-to-end coverage for user-facing flows in `dashboard/`, `dataset-viewer/`, or `cloud/frontend/`.

## Source Material

- Existing Claude command shim: `.claude/commands/e2e.md`
- Maintained skill body: `.claude/skills/e2e-testing/SKILL.md`
- E2E role: `.codex/agents/e2e-runner.toml`

## Instructions

- Generate or update E2E coverage only for the requested flow.
- Reuse the existing framework and project structure in the touched app.
- Prefer stable selectors and assertions over brittle timing.
- Capture the useful artifacts and report failures, flake risk, and next fixes.
- If the work also affects robot behavior, validate simulation or backend behavior separately from browser coverage.
