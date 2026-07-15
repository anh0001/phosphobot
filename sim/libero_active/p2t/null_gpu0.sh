#!/usr/bin/env bash
cd "$(dirname "$0")/.."
VENV="$PWD/.venv"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0
PY="env -u PYTHONPATH $VENV/bin/python"
BASE=results/e0_full_expert_100k/seed0/train/checkpoints/100000/pretrained_model
SENT=p2t/sentinels; SIG=0.15
for NS in 1 2 3; do
  [ -f "$SENT/m2s_null_ns${NS}" ] && { echo "ns$NS already done"; continue; }
  echo "[null-gpu0] ns=$NS starting $(date '+%T')"
  P2T_ACTION_NOISE=$SIG P2T_NOISE_SEED=$NS $PY preempt/reach_field.py --ckpt "$BASE" \
    --mags 0.05,0.075 --out "p2t/eval_null_s${SIG}_ns${NS}.jsonl" \
    --log-dir "p2t/eval_null_s${SIG}_ns${NS}_logs" --no-flow-samples \
    > "p2t/logs/nullgpu0_ns${NS}.log" 2>&1 || { echo "FATAL ns$NS"; exit 1; }
  grep -q REACH_FIELD_DONE "p2t/logs/nullgpu0_ns${NS}.log" && touch "$SENT/m2s_null_ns${NS}" \
    && echo "[null-gpu0] ns=$NS DONE $(date '+%T')"
done
echo "NULL_GPU0_DONE $(date '+%T')"
