# Phosphobot Research Playbook

Use this when a task needs external facts, upstream API behavior, hardware documentation, or broad repo context.

## Defaults

- Read local code, `AGENTS.md`, and `CLAUDE.md` first.
- Prefer primary sources: repo code, official docs, upstream library docs, and hardware/vendor documentation.
- Treat robot-facing behavior as safety-sensitive. Validate in simulation before suggesting real hardware execution.
- Include exact dates when facts may have changed recently.
- Keep an evidence trail with file paths, commands, and links.

## Suggested Flow

1. Inspect the relevant local package or app first.
2. Check existing tests, workflows, and build commands before proposing new ones.
3. Browse only for unstable or external facts:
   - hardware specs and SDK behavior
   - FastAPI, React, Vite, Tailwind, pybullet, Modal, or LeRobot API changes
   - package versions, CI behavior, or release notes
4. Prefer the smallest change that fits current repo patterns.
5. Summarize findings with concrete references, then implement.

## Repo-Specific Signals

- Backend: Python 3.10, FastAPI, uvicorn, Pydantic, pybullet
- Frontend: React 19 + Vite in `dashboard/`, Next.js apps in `dataset-viewer/` and `cloud/frontend/`
- Package managers: `uv` for Python, `npm` for frontend work
- Verification defaults:
  - backend: targeted `pytest`, then `make types` when types are affected
  - dashboard: `npm run build`, plus `npm run test` when state/util logic changes
  - Next.js apps: `npm run build`
  - hardware/control changes: simulation-first verification
