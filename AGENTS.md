# AGENTS.md

This repository is a monorepo centered on the `phosphobot` Python package — an open-source AI-ready robotics control framework supporting teleoperation, dataset recording, and VLA model training/inference for SO-100, SO-101, Piper, Koch v1.1, WX-250, LeCabot, and Unitree Go2.

See also: [CLAUDE.md](CLAUDE.md) for Claude Code users.

## Scope

Use this file as the first-stop guide for code agents working in this repo:

- Prefer the smallest change that satisfies the request.
- Respect the current worktree. This repo is often edited in parallel.
- Treat robot-facing changes as safety-sensitive. Default to simulation-first verification.

## Repo Map

- `phosphobot/`: main Python package, CLI, FastAPI server, simulation, robot drivers, tests, and packaged frontend assets
- `phosphobot/phosphobot/endpoints/`: backend API routes
- `phosphobot/phosphobot/hardware/`: robot abstractions, hardware drivers, simulation integration
- `phosphobot/phosphobot/models/`: shared models and dataset abstractions
- `phosphobot/resources/`: URDFs, calibration defaults, frontend build output under `resources/dist`
- `dashboard/`: embedded React + Vite control dashboard shipped with the backend
- `dataset-viewer/`: separate Next.js dataset viewer
- `cloud/frontend/`: separate Next.js cloud frontend
- `modal/`: Modal-based inference and service code
- `examples/`: example scripts and demos
- `simulation/pybullet/`: pybullet-specific simulation package

## Tech Stack

- **Backend:** Python 3.10, FastAPI, uvicorn, Pydantic, pybullet
- **Frontend:** React 19, TypeScript, Vite 6, Tailwind CSS 4, Radix UI, Zustand
- **Package managers:** `uv` (Python), `npm` (Node >= 20)
- **Linting:** Ruff (Python), ESLint 9 (TypeScript), Prettier (frontend)
- **Type checking:** mypy (strict mode)

## Setup And Tooling

- Python work is managed with `uv`.
- Preferred Python version for development is `3.10`.
- Node work expects modern Node/npm. The root README recommends `node >= 20`.
- Root `make` is the main local dev entrypoint on macOS/Linux.

## Common Commands

From repo root:

- `make`: build `dashboard/` and start the backend in headless simulation mode
- `make prod_back`: start backend only, assuming frontend assets already exist
- `make prod_gui`: start backend with the simulation GUI
- `make chat`: run chat mode
- `make test_server`: start backend in simulated test mode on `127.0.0.1:8080`
- `make tests`: run backend pytest suite under `phosphobot/tests/phosphobot/`
- `make types`: run mypy for the Python package
- `make sort`: run Ruff import sorting

From `phosphobot/`:

- `uv run --python 3.10 phosphobot run --simulation=headless`
- `uv run --python 3.10 phosphobot run --simulation=gui`
- `uv run pytest tests/phosphobot/`
- `uv run pytest tests/api/`

From `dashboard/`:

- `npm install`
- `npm run dev`
- `npm run build`
- `npm run test`
- `npm run lint`

From `dataset-viewer/`:

- `npm install`
- `npm run dev`
- `npm run build`
- `npm run lint`

From `cloud/frontend/`:

- `npm install`
- `npm run dev`
- `npm run build`
- `npm run lint`

From `modal/`:

- `uv sync`
- `uv run ...` for targeted scripts/tests

## Architecture Notes

- The Python CLI entrypoint is `phosphobot/phosphobot/main.py` (Typer CLI).
- The FastAPI app is defined in `phosphobot/phosphobot/app.py`.
- API endpoints live in `phosphobot/phosphobot/endpoints/`.
- Robot discovery is coordinated by `phosphobot/phosphobot/robot.py` via `RobotConnectionManager`.
- Manipulator-style hardware implementations inherit from abstractions in `phosphobot/phosphobot/hardware/base.py`.
- The backend serves bundled dashboard assets from `phosphobot/resources/dist`, which are built from `dashboard/`.
- URDFs are stored in `phosphobot/resources/urdf/`.
- New robot support usually touches three areas:
  - a hardware driver under `phosphobot/phosphobot/hardware/`
  - robot detection/registration in `phosphobot/phosphobot/robot.py`
  - URDF or calibration assets under `phosphobot/resources/`
- For adding new LeRobot policy models, see [ADDING_LEROBOT_MODELS.md](ADDING_LEROBOT_MODELS.md).

## Working Rules

- Prefer targeted edits over broad refactors unless the task explicitly calls for structural work.
- Do not overwrite unrelated user changes in a dirty worktree.
- Keep lockfile changes intentional. If `package.json` or Python dependencies do not change, avoid incidental dependency churn.
- Follow existing local patterns for typing, logging, and API shapes rather than introducing new abstractions by default.
- When changing backend/frontend integration, remember that dashboard output is copied into `phosphobot/resources/dist`.

## Verification Expectations

- Backend logic changes:
  - run the narrowest relevant `pytest` target first
  - run `make types` or targeted `mypy` if types are affected
- Dashboard changes:
  - run `npm run build`
  - run `npm run test` when touching tested utility or state logic
- Next.js app changes:
  - run `npm run build`
- Packaging or integration changes:
  - run the smallest end-to-end command that proves the flow still starts

If you cannot run a full verification locally, state exactly what you did run and what remains unverified.

## Hardware Safety

- Default to `--only-simulation`, headless simulation, or the pybullet GUI when validating robot control changes.
- Do not assume physical hardware is connected.
- Avoid commands that can move a real robot unless the task explicitly requires that and the user asked for it.
- When changing motor, calibration, CAN, or URDF behavior, validate in simulation before suggesting hardware execution.

## Frontend Asset Sync

If you change files in `dashboard/` and those changes need to ship with the backend, rebuild the dashboard and copy the output into `phosphobot/resources/dist` via:

- `make`
- or `make build_frontend`

Do not manually edit files inside `phosphobot/resources/dist` unless the task is specifically about generated frontend output.

## CI/CD

Workflows live in `.github/workflows/`:

- `pytest_tests.yml` — Unit + integration tests on PRs touching `phosphobot/`
- `mypy_tests.yml` — Type checking on PRs and pushes to main
- `publish.yml` — Builds binaries (macOS, Linux, Windows), publishes to PyPI
- `deploy_modal.yml` — Modal deployment

## Related Documentation

- [CONTRIBUTING.md](CONTRIBUTING.md) — Fork/branch/PR workflow, bounty program
- [ADDING_LEROBOT_MODELS.md](ADDING_LEROBOT_MODELS.md) — Detailed guide for adding new LeRobot models
- [phosphobot/README.md](phosphobot/README.md) — Install from source, build instructions
- [tutorials/](tutorials/) — Fine-tuning guides (e.g., GR00T VLA)
- [examples/](examples/) — Runnable demo scripts (circles, voice, hand tracking, etc.)
