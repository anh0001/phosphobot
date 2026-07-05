#!/usr/bin/env bash
# E0 strong-checkpoint RESUME — the 40k-step run (e0_train_strong.py) was SIGKILLed
# (rc=-9, host-RAM OOM) during the in-training eval right after the 20k checkpoint.
# Training itself ran 20k steps clean; only the lerobot eval-env spawn blew up memory.
#
# Resume from checkpoints/last (=020000, full training_state present) and run to 40k,
# with the in-training eval DISABLED (--eval_freq=0) so the env-spawn OOM cannot recur.
# Checkpoints still save at 25k/30k/35k and the final 40k (lerobot_train.py:485).
# The real gate is our position-disciplined preempt/eval_clean.py, run AFTER training.
set -euo pipefail
cd "$(dirname "$0")/.."   # sim/libero_active

CKPT=results/e0_strong_full432/seed0/train/checkpoints/last/pretrained_model/train_config.json
LOG=results/e0_strong_full432/seed0/train.resume.log

env -u PYTHONPATH MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  .venv/bin/lerobot-train \
    --config_path="$CKPT" \
    --resume=true \
    --eval_freq=0 \
  2>&1 | tee "$LOG"
echo "[e0_resume] lerobot-train exit=${PIPESTATUS[0]}"
