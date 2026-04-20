# Codex Supplement for phosphobot

This file supplements the root `AGENTS.md` with Codex-specific guidance.

## Intent

- Keep the Codex surface small and repo-local.
- Reuse the existing `.claude/` guidance instead of duplicating it.
- Treat Codex as instruction-driven: no slash commands, no repo-local hook automation.

## Codex Defaults

- Read `AGENTS.md` first, then this file.
- For deeper repo-specific workflows, explicitly open the matching file under:
  - `.codex/prompts/` for Codex-ready workflow prompts
  - `.claude/commands/` for the existing workflow source
  - `.claude/rules/` and `.claude/skills/` when a task needs more detail
- Prefer the smallest correct change and preserve the current worktree.
- Treat robot-facing changes as safety-sensitive and validate in simulation first.

## Recommended Prompt Files

These are the Codex equivalents of the current Claude workflow surface:

- `.codex/prompts/feature-development.md`
- `.codex/prompts/plan.md`
- `.codex/prompts/tdd.md`
- `.codex/prompts/e2e.md`
- `.codex/prompts/python-review.md`
- `.codex/prompts/verify.md`

Use them as explicit task framing, for example:

> Follow `.codex/prompts/tdd.md` for this backend fix.

## Repo Workflow Mapping

- Backend/API work:
  - use `uv`, targeted `pytest`, and `make types` when types change
- Dashboard work:
  - use `npm run build`, plus `npm run test` when state or utility logic changes
- Dataset viewer / cloud frontend:
  - use the app-local `npm run build`
- Hardware, CAN, calibration, URDF, or robot motion changes:
  - validate in headless simulation or the pybullet GUI before suggesting real hardware runs

## Research Discipline

- Follow `.claude/research/phosphobot-research-playbook.md` when the task depends on external facts, upstream APIs, or hardware/vendor docs.
- Prefer primary sources and keep an evidence trail with exact file paths, commands, and dates.

## Multi-Agent Roles

If Codex multi-agent is enabled, prefer these project-local roles:

- `explorer` for read-only execution-path tracing
- `planner` for multi-step implementation plans
- `reviewer` for general correctness and regression review
- `python_reviewer` for Python-heavy diffs in `phosphobot/`
- `docs_researcher` for upstream API and release-note verification
- `e2e_runner` for browser-flow coverage in `dashboard/`, `dataset-viewer/`, or `cloud/frontend/`

## What Is Intentionally Not Ported

The repo already has a strong Claude surface. This Codex layer intentionally does not duplicate:

- the full `.claude/skills/` tree
- hook automation
- global install scripts
- large generic command catalogs unrelated to phosphobot

The goal is a thin Codex adapter over the repo's existing guidance, not a second parallel system.
