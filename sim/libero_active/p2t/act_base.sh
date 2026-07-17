#!/usr/bin/env bash
cd "$(dirname "$0")/.."
VENV="$PWD/.venv"; export CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
EPS=$(cat p2t/orig_episodes.json)
env -u PYTHONPATH PATH="$VENV/bin:$PATH" "$VENV/bin/accelerate" launch --num_processes=1 --mixed_precision=bf16 \
  "$VENV/bin/lerobot-train" --policy.type=act --policy.push_to_hub=false \
  --dataset.repo_id=local/p2t_mixed --dataset.root=p2t/datasets/p2t_mixed --dataset.episodes="$EPS" \
  --policy.chunk_size=100 --policy.n_action_steps=100 --policy.optimizer_lr=1e-5 \
  --env.type=libero --env.task=libero_spatial --steps=100000 --batch_size=8 \
  --save_freq=100000 --eval_freq=0 --num_workers=8 --output_dir=p2t/ft/ACT_base/train \
  --seed=0 --wandb.enable=false > p2t/logs/act_base.log 2>&1
ls p2t/ft/ACT_base/train/checkpoints/100000/pretrained_model/*.safetensors >/dev/null 2>&1 && echo ACT_BASE_DONE || echo ACT_BASE_FAILED
