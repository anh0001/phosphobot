#!/usr/bin/env bash
# gpu_watcher.sh — wait for a free GPU, then run the active-loop smoke and (if it
# passes) one PS.5 single-cell run. Designed for a polite shared-lab box: it never
# touches anyone else's job, and only fires when a card has truly settled.
#
# Free-GPU definition: <FREE_MIB_MAX MiB used AND <FREE_UTIL_MAX% util, sustained
# for FREE_HOLD_S seconds. Polled every POLL_S seconds.
#
# Usage:
#   nohup bash scripts/gpu_watcher.sh > results/watcher.log 2>&1 &
# Stop:
#   pkill -f gpu_watcher.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
mkdir -p results

VENV="$HERE/.venv/bin/activate"
POLL_S=${POLL_S:-60}            # poll every 60 s
FREE_HOLD_S=${FREE_HOLD_S:-300} # require 5 min of sustained idle before launching
FREE_MIB_MAX=${FREE_MIB_MAX:-2000}
FREE_UTIL_MAX=${FREE_UTIL_MAX:-5}

ts() { date '+%Y-%m-%d %H:%M:%S'; }
# Logs go to stderr so they do NOT leak into command substitution
# (e.g. `GPU=$(wait_for_free_gpu)` must capture ONLY the GPU index).
log() { echo "[$(ts)] $*" >&2; }

# -- pick the first GPU index that is free RIGHT NOW (0 == found, prints index) --
pick_free_gpu() {
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used \
    --format=csv,noheader,nounits 2>/dev/null | awk -F', ' \
    -v M="$FREE_MIB_MAX" -v U="$FREE_UTIL_MAX" \
    '$2 <= U && $3 <= M { print $1; exit }'
}

# -- wait until ONE GPU stays free for FREE_HOLD_S seconds --
wait_for_free_gpu() {
  local hold_start="" candidate=""
  while :; do
    local idx
    idx="$(pick_free_gpu)"
    if [ -n "${idx:-}" ]; then
      if [ "$idx" != "${candidate:-}" ]; then
        candidate="$idx"; hold_start="$(date +%s)"
        log "GPU $idx looks idle; starting hold-down timer (${FREE_HOLD_S}s)"
      else
        local elapsed=$(( $(date +%s) - hold_start ))
        if [ "$elapsed" -ge "$FREE_HOLD_S" ]; then
          log "GPU $idx idle for ${elapsed}s — claiming"
          echo "$idx"
          return 0
        fi
      fi
    elif [ -n "$candidate" ]; then
      log "GPU was reclaimed before hold-down expired; resetting"
      candidate=""; hold_start=""
    fi
    sleep "$POLL_S"
  done
}

run_smoke() {
  local gpu="$1"
  log "launching active-loop smoke on GPU $gpu (~30-45 min)"
  rm -rf results/ps5_smoke results/ps5_smoke.log
  # shellcheck disable=SC1090
  source "$VENV"
  env -u PYTHONPATH CUDA_VISIBLE_DEVICES="$gpu" \
      MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
      python scripts/smoke_active_loop.py > results/ps5_smoke.log 2>&1
  local rc=$?
  log "smoke exited rc=$rc"
  tail -3 results/ps5_smoke.log
  return $rc
}

run_ps5_cell() {
  local gpu="$1"
  log "launching PS.5 single cell on GPU $gpu (random method, spatial, seed 0, max_demos=20)"
  rm -rf results/ps5_single_cell results/ps5_single_cell.log
  # shellcheck disable=SC1090
  source "$VENV"
  env -u PYTHONPATH CUDA_VISIBLE_DEVICES="$gpu" \
      MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
      python - <<'PY' > results/ps5_single_cell.log 2>&1
from dataclasses import replace
from conformal_active.active_loop import run_active_loop
from conformal_active.config import (
    ActiveLoopConfig, ConformalConfig, ExperimentConfig, TrainConfig,
)
cfg = ExperimentConfig(
    method="random",   # produces the random baseline curve (real PS.6 contributor)
    suite="libero_spatial",
    seed=0,
    conformal=ConformalConfig(target_alpha=0.1, n_action_samples=4),
    train=replace(TrainConfig(), steps_per_round=4000),
    loop=ActiveLoopConfig(
        seed_demos=5, demos_per_round=5, max_demos=20,
        eval_episodes=20, max_candidates_per_round=30,
    ),
    results_dir="results/ps5_single_cell",
)
out = run_active_loop(cfg)
import json; print(json.dumps(out, indent=2))
PY
  local rc=$?
  log "PS.5 cell exited rc=$rc"
  tail -10 results/ps5_single_cell.log
  return $rc
}

log "watcher starting (POLL_S=$POLL_S, FREE_HOLD_S=$FREE_HOLD_S, FREE_MIB_MAX=$FREE_MIB_MAX, FREE_UTIL_MAX=$FREE_UTIL_MAX)"
GPU="$(wait_for_free_gpu)"
log "claimed GPU $GPU"
if run_smoke "$GPU"; then
  log "smoke PASSED -> launching PS.5 single cell"
  run_ps5_cell "$GPU"
  log "PS.5 cell finished -> watcher exiting"
else
  log "smoke FAILED -> NOT launching PS.5 cell; investigate results/ps5_smoke.log"
fi
