#!/usr/bin/env bash
# Autonomous N=20 sweep orchestrator (v2: nan-aware completion).
# A cell is COMPLETE only when its curve has `target` points AND the last
# point is non-nan. An OOM at the final round writes a nan entry that the
# v1 count-based check wrongly treated as complete; v2 retries it.
set -u
REPO=/srv/storage/roboserver1/home/anhar/codes/phosphobot
LIBERO=$REPO/sim/libero_active
LOG=$REPO/refine-logs/overnight_sweep.log
cd "$LIBERO" || { echo "FATAL: cd $LIBERO failed" >> "$LOG"; exit 1; }

GPU=0
MIN_FREE_MB=10000
MAX_RETRY=5
POLL=120

log () { echo "$(date '+%Y-%m-%d %H:%M:%S')  $*" >> "$LOG"; }
gpu_free () { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU 2>/dev/null | tr -d ' '; }

# COMPLETE iff curve has >=target points AND last point pc_success is non-nan.
complete () {  # $1=outer $2=inner $3=target
  python3 -c "
import json,os,math
p='results/ps5_$1/$2/curve.json'
ok='no'
if os.path.exists(p):
    cv=json.load(open(p)).get('curve',[])
    if len(cv)>=$3 and cv:
        v=cv[-1]['pc_success']
        if v is not None and not (isinstance(v,float) and math.isnan(v)): ok='yes'
print(ok)
" 2>/dev/null || echo no
}

cell_running () { pgrep -f "results/ps5_$1/" >/dev/null 2>&1; }

wait_gpu () {
  while :; do
    f=$(gpu_free); f=${f:-0}
    if [ "$f" -ge "$MIN_FREE_MB" ]; then
      sleep 20; f=$(gpu_free); f=${f:-0}
      [ "$f" -ge "$MIN_FREE_MB" ] && return 0
    fi
    sleep $POLL
  done
}

launch () {  # $1=method $2=seed $3=maxdemos
  local method=$1 seed=$2 md=$3 suffix="__N${3}"
  if [ "$method" = "random" ]; then
    GPU=$GPU SEED=$seed SUITE=libero_spatial MAX_DEMOS=$md TAG_SUFFIX=$suffix \
      nohup bash scripts/run_random_cell.sh > "results/ps5_${method}_s${seed}_N${md}_sweep.log" 2>&1 &
  else
    GPU=$GPU SEED=$seed SUITE=libero_spatial METHOD=$method MAX_DEMOS=$md TAG_SUFFIX=$suffix \
      nohup bash scripts/run_dispersion_cell.sh > "results/ps5_${method}_s${seed}_N${md}_sweep.log" 2>&1 &
  fi
}

wait_cell () {  # $1=outer $2=inner $3=target  -> 0 done, 1 failed
  local outer=$1 inner=$2 target=$3
  while :; do
    [ "$(complete "$outer" "$inner" "$target")" = "yes" ] && return 0
    if grep -qE "Traceback|OutOfMemoryError|train failed rc=" "results/ps5_${outer}.log" 2>/dev/null; then
      cell_running "$outer" || return 1
    fi
    sleep $POLL
  done
}

ensure () {  # $1=method $2=seed $3=maxdemos
  local method=$1 seed=$2 md=$3
  local outer="${method}__libero_spatial__seed${seed}__N${md}"
  local inner="${method}__libero_spatial__seed${seed}"
  local target=$(( md / 5 )) a
  for a in $(seq 1 $MAX_RETRY); do
    if [ "$(complete "$outer" "$inner" "$target")" = "yes" ]; then
      log "SKIP $outer already complete"; return 0
    fi
    if cell_running "$outer"; then
      log "WAIT $outer already running"
    else
      log "GPU-WAIT for $outer (need ${MIN_FREE_MB}MB free)"
      wait_gpu
      log "LAUNCH $outer (attempt $a)"
      launch "$method" "$seed" "$md"
      sleep 40
    fi
    if wait_cell "$outer" "$inner" "$target"; then
      local curve=$(python3 -c "import json;print(json.load(open('results/ps5_${outer}/${inner}/curve.json'))['curve'])" 2>/dev/null)
      log "DONE $outer -> $curve"; return 0
    fi
    log "FAIL $outer (attempt $a) -- OOM/last-round-nan; will retry after GPU frees"
    sleep 60
  done
  log "GIVEUP $outer after $MAX_RETRY attempts"; return 1
}

log "=== overnight sweep v2 START (nan-aware) ==="
ensure dispersion_quota 2 20   # redo: v1 left nan at N=20
ensure dispersion_quota 3 20
ensure random 2 20
ensure random 3 20
log "=== overnight sweep v2 COMPLETE ==="
{
  echo "--- FINAL CURVES $(date '+%Y-%m-%d %H:%M:%S') ---"
  for c in dispersion_quota__libero_spatial__seed1__N20:dispersion_quota__libero_spatial__seed1 \
           dispersion_quota__libero_spatial__seed2__N20:dispersion_quota__libero_spatial__seed2 \
           dispersion_quota__libero_spatial__seed3__N20:dispersion_quota__libero_spatial__seed3 \
           random__libero_spatial__seed2__N20:random__libero_spatial__seed2 \
           random__libero_spatial__seed3__N20:random__libero_spatial__seed3; do
    o=${c%%:*}; i=${c##*:}; echo -n "$o -> "
    python3 -c "import json,os;p='results/ps5_$o/$i/curve.json';print(json.load(open(p))['curve'] if os.path.exists(p) else 'MISSING')" 2>/dev/null || echo ERR
  done
} >> "$LOG"
echo "SWEEP_DONE" >> "$LOG"
