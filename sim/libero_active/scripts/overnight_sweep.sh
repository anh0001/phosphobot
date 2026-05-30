#!/usr/bin/env bash
# Autonomous overnight N=20 sweep orchestrator.
# Queue of cells; for each: wait for GPU0 free -> launch -> wait for completion
# -> retry on OOM (up to 3x). Detached, survives session end. Logs to LOG.
set -u
REPO=/srv/storage/roboserver1/home/anhar/codes/phosphobot
LIBERO=$REPO/sim/libero_active
LOG=$REPO/refine-logs/overnight_sweep.log
cd "$LIBERO" || { echo "FATAL: cd $LIBERO failed" >> "$LOG"; exit 1; }

GPU=0                 # kubotal+ holds GPU1; we live on GPU0 opportunistically
MIN_FREE_MB=10000     # need ~8GB; require 10GB headroom before launching
MAX_RETRY=3
POLL=120

log () { echo "$(date '+%Y-%m-%d %H:%M:%S')  $*" >> "$LOG"; }

gpu_free () { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU 2>/dev/null | tr -d ' '; }

pts () {  # $1=outer $2=inner
  python3 -c "import json,os;p='results/ps5_$1/$2/curve.json';print(len(json.load(open(p)).get('curve',[])) if os.path.exists(p) else 0)" 2>/dev/null || echo 0
}

cell_running () {  # $1=outer  -> 0 if a proc is writing to this cell's results dir
  pgrep -f "results/ps5_$1/" >/dev/null 2>&1
}

wait_gpu () {
  while :; do
    f=$(gpu_free); f=${f:-0}
    if [ "$f" -ge "$MIN_FREE_MB" ]; then
      sleep 20; f=$(gpu_free); f=${f:-0}   # re-confirm after 20s
      [ "$f" -ge "$MIN_FREE_MB" ] && return 0
    fi
    sleep $POLL
  done
}

# launch one cell. $1=method $2=seed $3=maxdemos
launch () {
  local method=$1 seed=$2 md=$3
  local suffix="__N${md}"
  if [ "$method" = "random" ]; then
    GPU=$GPU SEED=$seed SUITE=libero_spatial MAX_DEMOS=$md TAG_SUFFIX=$suffix \
      nohup bash scripts/run_random_cell.sh > "results/ps5_${method}_s${seed}_N${md}_sweep.log" 2>&1 &
  else
    GPU=$GPU SEED=$seed SUITE=libero_spatial METHOD=$method MAX_DEMOS=$md TAG_SUFFIX=$suffix \
      nohup bash scripts/run_dispersion_cell.sh > "results/ps5_${method}_s${seed}_N${md}_sweep.log" 2>&1 &
  fi
}

# wait for a launched cell to reach target pts or fail. $1=outer $2=inner $3=target
wait_cell () {
  local outer=$1 inner=$2 target=$3
  while :; do
    [ "$(pts "$outer" "$inner")" = "$target" ] && return 0
    if grep -qE "Traceback|OutOfMemoryError|train failed rc=" "results/ps5_${outer}.log" 2>/dev/null; then
      # failed only if no proc is still working on this cell
      cell_running "$outer" || return 1
    fi
    sleep $POLL
  done
}

# ensure one cell completes (with retry). $1=method $2=seed $3=maxdemos
ensure () {
  local method=$1 seed=$2 md=$3
  local outer="${method}__libero_spatial__seed${seed}__N${md}"
  local inner="${method}__libero_spatial__seed${seed}"
  local target=$(( md / 5 ))
  local a
  for a in $(seq 1 $MAX_RETRY); do
    if [ "$(pts "$outer" "$inner")" = "$target" ]; then
      log "SKIP $outer already complete ($target pts)"; return 0
    fi
    if cell_running "$outer"; then
      log "WAIT $outer already running (attempt context $a)"
    else
      log "GPU-WAIT for $outer (need ${MIN_FREE_MB}MB free on GPU$GPU)"
      wait_gpu
      log "LAUNCH $outer (attempt $a)"
      launch "$method" "$seed" "$md"
      sleep 40
    fi
    if wait_cell "$outer" "$inner" "$target"; then
      local curve=$(python3 -c "import json;print(json.load(open('results/ps5_${outer}/${inner}/curve.json'))['curve'])" 2>/dev/null)
      log "DONE $outer -> $curve"
      return 0
    else
      log "FAIL $outer (attempt $a) -- likely OOM; will retry after GPU frees"
      sleep 60
    fi
  done
  log "GIVEUP $outer after $MAX_RETRY attempts"
  return 1
}

log "=== overnight sweep START ==="
# Queue: most informative first. quota seeds, then paired random seeds.
ensure dispersion_quota 1 20
ensure dispersion_quota 2 20
ensure dispersion_quota 3 20
ensure random 2 20
ensure random 3 20
log "=== overnight sweep COMPLETE ==="

# Final summary
{
  echo "--- FINAL CURVES $(date '+%Y-%m-%d %H:%M:%S') ---"
  for c in dispersion_quota__libero_spatial__seed1__N20:dispersion_quota__libero_spatial__seed1 \
           dispersion_quota__libero_spatial__seed2__N20:dispersion_quota__libero_spatial__seed2 \
           dispersion_quota__libero_spatial__seed3__N20:dispersion_quota__libero_spatial__seed3 \
           random__libero_spatial__seed2__N20:random__libero_spatial__seed2 \
           random__libero_spatial__seed3__N20:random__libero_spatial__seed3; do
    o=${c%%:*}; i=${c##*:}
    echo -n "$o -> "
    python3 -c "import json,os;p='results/ps5_$o/$i/curve.json';print(json.load(open(p))['curve'] if os.path.exists(p) else 'MISSING')" 2>/dev/null || echo "ERR"
  done
} >> "$LOG"
echo "SWEEP_DONE" >> "$LOG"
