# Runbook — commands for each Phase S step

All commands run from `sim/libero_active/` after `source .venv/bin/activate`, prefixed
with `env -u PYTHONPATH MUJOCO_GL=egl PYOPENGL_PLATFORM=egl` (isolates the system ROS
install and selects headless MuJoCo rendering).

## Unit tests (PS.3 — no GPU)

    env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -q

## PS.4 — offline-pipeline sanity gate

Smoke (pipeline check, ~5 min, success rate not gated):

    python offline_sanity.py --smoke --suite libero_spatial

Real gate (~40 demos, 60k steps, multi-hour on one RTX 6000):

    python offline_sanity.py --suite libero_spatial --n-demos 40 --steps 60000
    # GATE: pc_success >= 80% over 20 rollouts -> proceed to PS.5/PS.6

## PS.6 — main-results sweep

Dry run (print the grid):

    python run_sweep.py --full --dry-run

Single cell (debug the active loop):

    python run_sweep.py --methods random --suites libero_spatial --seeds 0

Full paper grid (5 methods x 2 suites x 3 seeds = 30 active loops):

    python run_sweep.py --full

## Notes

- First run downloads SmolVLA weights + the `HuggingFaceVLA/libero` dataset.
- Set `HF_TOKEN` for faster, rate-limit-free downloads.
- `policy_runner.episode_signals` must be validated against the PS.4 checkpoint before
  the PS.6 sweep — it is the one module that needs a live policy (see its docstring).
