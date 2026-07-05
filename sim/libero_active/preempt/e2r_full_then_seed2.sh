#!/usr/bin/env bash
# Post-reframe critical path (Codex): after seed-2 training frees the GPU,
# (1) full 740-rollout E2r on the seed-0 strong ckpt (CT primary, the new headline),
# (2) seed-2 E1r-50mm subset on the seed-1 ckpt (seed-robustness, fills the paper placeholder).
set -uo pipefail
cd "$(dirname "$0")/.."
VENV="$PWD/.venv"
# wait for seed-2 training to finish (ckpt present AND trainer gone)
while true; do
  if ls "results/e0_full_expert_100k/seed1/train/checkpoints/100000/pretrained_model"/*.safetensors >/dev/null 2>&1 && ! pgrep -f "seed1/train" >/dev/null 2>&1; then break; fi
  sleep 120
done
echo "[chain] seed-2 done; GPU free -> full E2r"
env -u PYTHONPATH MUJOCO_GL=egl PYOPENGL_PLATFORM=egl "$VENV/bin/python" preempt/object_local_paste.py \
  --ckpt "results/e0_full_expert_100k/seed0/train/checkpoints/100000/pretrained_model" --mags 0.05 --dirs px,nx,py,ny --conditions CC,CT,TC,TT,CR --edit-cams image,image2 \
  --out preempt/e2r_object_local_paste_50mm_FULL.jsonl --log-dir preempt/e2r_full_logs > preempt/e2r_full.run.log 2>&1
echo "[chain] full E2r done -> seed-2 E1r-50mm"
env -u PYTHONPATH MUJOCO_GL=egl PYOPENGL_PLATFORM=egl "$VENV/bin/python" preempt/reach_field.py \
  --ckpt "results/e0_full_expert_100k/seed1/train/checkpoints/100000/pretrained_model" --tasks 0,1,2,3,4,5,6,7,8,9 --seeds 1000,1001,1002,1003,1004 \
  --mags 0.05 --dirs px,nx,py,ny --no-flow-samples \
  --out preempt/e1r_seed1_50mm.jsonl --log-dir preempt/e1r_seed1_50mm_logs > preempt/e1r_seed1.run.log 2>&1
echo "CHAIN_ALL_DONE"
