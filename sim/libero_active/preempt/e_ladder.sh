#!/usr/bin/env bash
# Emergence ladder: reduced reach-field (50mm, 4 dirs) on full-FT ckpts to plot
# bias(cos) vs competence(clean success). 100k already covered by E1r.
set -euo pipefail
cd "$(dirname "$0")/.."
VENV="$PWD/.venv"; CKDIR=results/e0_full_expert_100k/seed0/train/checkpoints
for k in 010000 040000 070000; do
  echo "=== LADDER ckpt $k ==="
  env -u PYTHONPATH MUJOCO_GL=egl PYOPENGL_PLATFORM=egl "$VENV/bin/python" preempt/reach_field.py \
    --ckpt "$CKDIR/$k/pretrained_model" \
    --tasks 0,1,2,3,4,5,6,7,8,9 --seeds 1000,1001,1002,1003,1004 \
    --mags 0.05 --dirs px,nx,py,ny --no-flow-samples \
    --out "preempt/e_ladder_${k}.jsonl" --log-dir "preempt/e_ladder_${k}_logs"
done
echo "E_LADDER_DONE"
