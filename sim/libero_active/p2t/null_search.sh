#!/usr/bin/env bash
# M1 / R010 — degradation-null noise-scale search.
# Run the BASE checkpoint's reach field @50mm under several action-noise scales;
# pick the scale whose clean AND displaced@50 success match M2 (clean 52%, disp 37%).
# The matched scale becomes the PRIMARY competence-matched degradation null:
# same policy (same grounding), degraded only by execution noise.
set -uo pipefail
cd "$(dirname "$0")/.."
VENV="$PWD/.venv"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=1 CUDA_VISIBLE_DEVICES=1
PY="env -u PYTHONPATH $VENV/bin/python"
BASE=results/e0_full_expert_100k/seed0/train/checkpoints/100000/pretrained_model

for S in 0.10 0.15 0.20 0.25; do
  OUT="p2t/null_search_s${S}.jsonl"
  if [ ! -f "p2t/sentinels/nullsearch_${S}" ]; then
    echo "[null-search] sigma=$S starting $(date '+%T')"
    P2T_ACTION_NOISE=$S $PY preempt/reach_field.py --ckpt "$BASE" --mags 0.05 \
      --out "$OUT" --log-dir "p2t/null_search_s${S}_logs" --no-flow-samples \
      > "p2t/logs/nullsearch_s${S}.log" 2>&1 || { echo "FATAL sigma=$S"; exit 1; }
    touch "p2t/sentinels/nullsearch_${S}"
  fi
  $PY - "$OUT" "$S" <<'EOF'
import json, sys
import numpy as np
recs=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
recs=[r for r in recs if 'error' not in r]
clean=[r['success'] for r in recs if r.get('cond')=='clean']
disp=[r['success'] for r in recs if r.get('cond','').startswith('d50mm')]
print(f"[null-search] sigma={sys.argv[2]}: clean {np.mean(clean):.0%} ({sum(clean)}/{len(clean)})  "
      f"disp@50 {np.mean(disp):.0%} ({sum(disp)}/{len(disp)})  | target M2: clean 52% disp 37%")
EOF
done
echo "NULL_SEARCH_DONE $(date '+%T')"
