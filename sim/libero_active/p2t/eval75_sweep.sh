#!/usr/bin/env bash
# Post-hoc SECONDARY analysis (flagged 2026-07-07 21:00, before D/E results known):
# A's synthesis mass sits at 75-100mm; the 50mm primary eval may under-credit it.
set -uo pipefail
cd "$(dirname "$0")/.."
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=1 CUDA_VISIBLE_DEVICES=1
PY="env -u PYTHONPATH $PWD/.venv/bin/python"
for COND in A_gain B_noaug C_uniform D_failure E_sham; do
  CKPT="p2t/ft/$COND/train/checkpoints/020000/pretrained_model"
  $PY preempt/reach_field.py --ckpt "$CKPT" --mags 0.075 \
    --out "p2t/eval75_$COND.jsonl" --log-dir "p2t/eval75_${COND}_logs" \
    --no-flow-samples > "p2t/logs/s7b_eval75_$COND.log" 2>&1 || { echo "FATAL $COND"; exit 1; }
  echo "eval75 $COND DONE"
done
echo "EVAL75_SWEEP_DONE"
