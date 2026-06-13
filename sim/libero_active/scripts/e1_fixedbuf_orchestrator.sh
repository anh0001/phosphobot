#!/usr/bin/env bash
# E1 — fixed-buffer LoRA-retrain variance study ("Unreliable by Default", load-bearing experiment).
# Waits for the in-flight random s0 N=20 cell to finish (frees the single GPU + yields the
# random N=20 buffer), then retrains LoRA k times per FROZEN buffer (quota + random) with
# different train seeds and a FIXED eval seed (1000) to isolate train-seed variance.
set -u
LIBERO=/srv/data/users/anhar/codes/phosphobot/sim/libero_active
LOG=/srv/data/users/anhar/codes/phosphobot/refine-logs/e1_fixedbuf.log
cd "$LIBERO" || { echo "FATAL cd" >> "$LOG"; exit 1; }
source .venv/bin/activate

QUOTA_BUF=1310,1315,1327,1363,1388,1394,1405,1406,1409,1434,1437,1456,1518,1521,1523,1575,1579,1587,1631,1673
SEEDS="${SEEDS:-0,1,2,3,4}"
RAND_RUN=results/ps5_random__libero_spatial__seed0/random__libero_spatial__seed0

log(){ echo "$(date '+%F %T')  $*" >> "$LOG"; }

complete(){ python3 -c "
import json,os,math
p='$RAND_RUN/curve.json'
ok='no'
if os.path.exists(p):
    cv=json.load(open(p)).get('curve',[])
    if len(cv)>=4 and cv:
        v=cv[-1]['pc_success']
        if v is not None and not (isinstance(v,float) and math.isnan(v)): ok='yes'
print(ok)" 2>/dev/null || echo no; }

run_e1(){  # $1=label $2=buffer
  log "E1 START $1 (seeds $SEEDS)"
  env -u PYTHONPATH CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python fixed_buffer_retrain.py --label "$1" --buffer "$2" --seeds "$SEEDS" \
      --suite libero_spatial --steps 4000 --eval-episodes 20 >> "$LOG" 2>&1
  log "E1 DONE $1 rc=$?"
}

log "=== E1 orchestrator START; waiting for random s0 cell (poll 300s) ==="
while [ "$(complete)" != "yes" ]; do sleep 300; done
log "random s0 cell complete"
RAND_BUF=$(python3 -c "
import json,glob
g=glob.glob('$RAND_RUN/round3_N20/train/checkpoints/*/pretrained_model/train_config.json') \
  or glob.glob('$RAND_RUN/round3_N20/train/**/train_config.json',recursive=True)
print(','.join(map(str,json.load(open(g[0]))['dataset']['episodes'])) if g else '')
" 2>/dev/null)
log "random N20 buffer = $RAND_BUF"

run_e1 quota_N20 "$QUOTA_BUF"
if [ -n "$RAND_BUF" ]; then run_e1 random_N20 "$RAND_BUF"; else log "SKIP random_N20 (no buffer)"; fi
log "=== E1 orchestrator COMPLETE ==="
echo "E1_DONE" >> "$LOG"
