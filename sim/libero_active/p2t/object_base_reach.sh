#!/usr/bin/env bash
cd "$(dirname "$0")/.."
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=1 CUDA_VISIBLE_DEVICES=1 E_SUITE=libero_object
env -u PYTHONPATH E_SUITE=libero_object .venv/bin/python preempt/reach_field.py \
  --ckpt results/e0_full_expert_object_100k/seed0/train/checkpoints/100000/pretrained_model --mags 0.05,0.075 --src preempt/pair_meta_libero_object.json \
  --out p2t/eval_object_base.jsonl --log-dir p2t/eval_object_base_logs --no-flow-samples \
  > p2t/logs/object_base_reach.log 2>&1
grep -q REACH_FIELD_DONE p2t/logs/object_base_reach.log && echo OBJECT_BASE_REACH_DONE || echo OBJECT_BASE_REACH_FAILED
