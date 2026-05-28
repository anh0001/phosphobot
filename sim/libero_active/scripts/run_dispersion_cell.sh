#!/usr/bin/env bash
# Launch one PS.5 dispersion cell on a chosen GPU.
# Usage:  GPU=0 SEED=0 SUITE=libero_spatial bash scripts/run_dispersion_cell.sh
# Pass METHOD=dispersion (default) or METHOD=dispersion_quota to use the quota variant.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

GPU="${GPU:-0}"
SEED="${SEED:-0}"
SUITE="${SUITE:-libero_spatial}"
METHOD="${METHOD:-dispersion}"
TAG="${METHOD}__${SUITE}__seed${SEED}"

# shellcheck disable=SC1091
source "$HERE/.venv/bin/activate"

rm -rf "results/ps5_${TAG}" "results/ps5_${TAG}.log"

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
    method="${METHOD}",
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
