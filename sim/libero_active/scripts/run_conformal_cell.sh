#!/usr/bin/env bash
# Launch one PS.5 conformal cell on a chosen GPU.
# Usage:  GPU=0 SEED=0 SUITE=libero_spatial bash scripts/run_conformal_cell.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

GPU="${GPU:-0}"
SEED="${SEED:-0}"
SUITE="${SUITE:-libero_spatial}"
TAG="conformal__${SUITE}__seed${SEED}"

# shellcheck disable=SC1091
source "$HERE/.venv/bin/activate"

rm -rf "results/ps5_${TAG}" "results/ps5_${TAG}.log"

# expandable_segments reduces fragmentation; helps survive transient pressure
# from a co-tenant on the same physical card.
env -u PYTHONPATH CUDA_VISIBLE_DEVICES="$GPU" \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python - <<PY > "results/ps5_${TAG}.log" 2>&1
from dataclasses import replace
from conformal_active.active_loop import run_active_loop
from conformal_active.config import (
    ActiveLoopConfig, ConformalConfig, ExperimentConfig, TrainConfig,
)
cfg = ExperimentConfig(
    method="conformal",
    suite="${SUITE}",
    seed=${SEED},
    conformal=ConformalConfig(target_alpha=0.1, n_action_samples=4),
    train=replace(TrainConfig(), steps_per_round=4000),
    loop=ActiveLoopConfig(
        seed_demos=5, demos_per_round=5, max_demos=20,
        eval_episodes=20, max_candidates_per_round=30,
    ),
    results_dir="results/ps5_${TAG}",
)
out = run_active_loop(cfg)
import json; print(json.dumps(out, indent=2))
PY
echo "rc=$?"
