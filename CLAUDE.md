# CLAUDE.md

## Project Overview

phosphobot is an open-source AI-ready robotics control framework. It provides teleoperation (keyboard, gamepad, leader arm, Meta Quest), dataset recording, and Vision Language Action (VLA) model training/inference. Supports SO-100, SO-101, Piper, Koch v1.1, WX-250, LeCabot, and Unitree Go2.

## Repository Structure

- `phosphobot/` — Python backend (FastAPI + hardware drivers), installed via `uv`
- `dashboard/` — React frontend (Vite + TypeScript + Tailwind CSS 4)
- `simulation/` — PyBullet simulation
- `inference/` — Model inference code
- `cloud/` — Cloud deployment
- `bullet3/` — Git submodule (pybullet with patches)

## Development Commands

All commands run from the repo root:

```bash
# Production (builds frontend, runs backend)
make                        # or: make prod

# Dev mode (localhost:8080, simulation GUI)
make local

# Frontend only (dev server with hot reload)
cd dashboard && npm run dev

# Backend only (no frontend rebuild)
make prod_back
```

## Testing

```bash
# Unit tests (Python)
make tests                  # runs: cd phosphobot && uv run pytest tests/phosphobot/ -n 5

# Integration tests (starts test server, then runs API tests)
make test_server            # start server in background
cd phosphobot && uv run pytest tests/api/ -s -v

# Frontend tests
cd dashboard && npm run test

# Type checking
make types                  # runs: cd phosphobot && uv run mypy . --check-untyped-defs --disallow-untyped-defs --ignore-missing-imports --follow-imports=silent

# Import sorting
make sort                   # runs: cd phosphobot && uv run ruff check --select I --fix .
```

## Tech Stack

- **Backend:** Python 3.10, FastAPI, uvicorn, pydantic, pybullet
- **Frontend:** React 19, TypeScript, Vite 6, Tailwind CSS 4, Radix UI, Zustand
- **Package managers:** `uv` (Python), `npm` (frontend, Node >=20)
- **Linting:** Ruff (Python), ESLint 9 (TypeScript)
- **Formatting:** Prettier (frontend)
- **Type checking:** mypy (Python strict mode)

## Key Conventions

- Python entry point: `phosphobot/phosphobot/main.py` (typer CLI)
- FastAPI app: `phosphobot/phosphobot/app.py`
- Hardware drivers live in `phosphobot/phosphobot/hardware/` — each robot type has its own module
- API endpoints in `phosphobot/phosphobot/endpoints/`
- Frontend builds to `phosphobot/resources/dist/` and is served by the backend
- URDFs stored in `phosphobot/resources/urdf/`

## CI/CD

- `pytest_tests.yml` — Runs unit + integration tests on PRs touching `phosphobot/`
- `mypy_tests.yml` — Type checking on PRs/pushes to main
- `publish.yml` — Builds binaries (macOS, Linux, Windows), publishes to PyPI
<!-- ARIS:BEGIN -->
## ARIS Skill Scope
ARIS skills installed in this project: 80 entries.
Manifest: `.aris/installed-skills.txt` (lists every skill ARIS installed and its upstream target).
For ARIS workflows, prefer the project-local skills under `.claude/skills/` over global skills.
Do not modify or delete files inside any skill that is a symlink (symlinks point into `/srv/data/users/anhar/codes/Auto-claude-code-research-in-sleep`).
Update with: `bash /srv/data/users/anhar/codes/Auto-claude-code-research-in-sleep/tools/install_aris.sh`  (re-runnable; reconciles new/removed skills).
<!-- ARIS:END -->