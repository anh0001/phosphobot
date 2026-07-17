#!/usr/bin/env bash
cd "$(dirname "$0")/.."
VENV="$PWD/.venv"
CK=p2t/ft/ACT_base/train/checkpoints/100000/pretrained_model
# wait for ACT base ckpt + training proc to exit (frees GPU 1)
while ! ls "$CK"/*.safetensors >/dev/null 2>&1; do sleep 30; done
while pgrep -f "lerobot-train.*act" >/dev/null 2>&1; do sleep 20; done
echo "[act] base ready, reach-field on GPU 1 $(date '+%T')"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=1 CUDA_VISIBLE_DEVICES=1 E_SUITE=libero_spatial
env -u PYTHONPATH E_SUITE=libero_spatial "$VENV/bin/python" preempt/reach_field.py \
  --ckpt "$CK" --mags 0.05,0.075 --out p2t/eval_act_base.jsonl \
  --log-dir p2t/eval_act_base_logs --no-flow-samples > p2t/logs/act_reach.log 2>&1
grep -q REACH_FIELD_DONE p2t/logs/act_reach.log && echo ACT_REACH_DONE || echo ACT_REACH_FAILED
