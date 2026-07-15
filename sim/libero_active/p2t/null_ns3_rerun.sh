#!/usr/bin/env bash
cd "$(dirname "$0")/.."
VENV="$PWD/.venv"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0
env -u PYTHONPATH P2T_ACTION_NOISE=0.15 P2T_NOISE_SEED=3 "$VENV/bin/python" preempt/reach_field.py \
  --ckpt results/e0_full_expert_100k/seed0/train/checkpoints/100000/pretrained_model \
  --mags 0.05,0.075 --out p2t/eval_null_s0.15_ns3.jsonl \
  --log-dir p2t/eval_null_s0.15_ns3_logs --no-flow-samples \
  --seeds 1000,1001,1002,1003,1004 > p2t/logs/null_ns3_rerun.log 2>&1
grep -q REACH_FIELD_DONE p2t/logs/null_ns3_rerun.log && touch p2t/sentinels/m2s_null_ns3 \
  && echo "NS3_RERUN_DONE $(date '+%T')" || echo "NS3_RERUN_FAILED"
