#!/usr/bin/env bash
# E0 escalation — FULL ACTION-EXPERT fine-tune (frozen VLM), per Codex xhigh after the
# LR hypothesis was refuted (both lr=1e-3 and lr=1e-4 LoRA-r64 plateau ~36-38%).
# This is the SmolVLA paper recipe (frozen VLM + train action expert, lr=1e-4, bf16;
# paper reports LIBERO-Spatial 90% for the 0.45B model). Difference vs the LoRA runs:
# NO --peft.* flags -> the action expert + projections get full gradients (not r=64
# adapters); VLM stays frozen (freeze_vision_encoder/train_expert_only are SmolVLA
# defaults, passed explicitly for safety). bf16 via accelerate to fit + match paper.
#
# PRE-REGISTERED CHEAP GATE (Codex): 30k pilot, eval 10k/20k/30k. Continue to 100k ONLY
# if 30k clears >=24/50 (preferably 25+). STOP if <=22/50 (gap is not just LoRA capacity).
# Final gate stays >=33/50.
set -euo pipefail
cd "$(dirname "$0")/.."   # sim/libero_active
VENV="$PWD/.venv"
BATCH="${BATCH:-32}"      # Codex: 48GB fits batch 32 for action-expert FT; fallback BATCH=16 on OOM
STEPS="${STEPS:-30000}"
SEED="${SEED:-0}"
SUITE="${SUITE:-libero_spatial}"
DECAY_STEPS="${DECAY_STEPS:-$STEPS}"  # cosine decays over the FULL run (single clean cycle); never < STEPS
OUT="${OUT:-results/e0_full_expert_full432/seed${SEED}/train}"

EPISODES=$(SUITE="$SUITE" "$VENV/bin/python" - <<'PY'
import json, os
m = json.load(open("results/suite_episode_map.json"))[os.environ["SUITE"]]
eps = sorted(int(e) for xs in m.values() for e in xs)
print(json.dumps(eps, separators=(",", ":")))
PY
)
echo "[e0_full_expert] episodes=$(echo "$EPISODES" | tr -cd , | wc -c)+1  batch=$BATCH steps=$STEPS out=$OUT"

env -u PYTHONPATH PATH="$VENV/bin:$PATH" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  "$VENV/bin/accelerate" launch --num_processes=1 --mixed_precision=bf16 \
    "$VENV/bin/lerobot-train" \
    --policy.path=lerobot/smolvla_base \
    --policy.push_to_hub=false \
    --dataset.repo_id=HuggingFaceVLA/libero \
    --dataset.episodes="$EPISODES" \
    --policy.output_features=null \
    --policy.input_features=null \
    --policy.optimizer_lr=1e-4 \
    --policy.scheduler_warmup_steps=1000 \
    --policy.scheduler_decay_steps="$DECAY_STEPS" \
    --policy.scheduler_decay_lr=2.5e-6 \
    --policy.freeze_vision_encoder=true \
    --policy.train_expert_only=true \
    --policy.train_state_proj=true \
    --policy.compile_model=false \
    --env.type=libero \
    --env.task="$SUITE" \
    --steps="$STEPS" \
    --batch_size="$BATCH" \
    --save_freq=10000 \
    --eval_freq=0 \
    --num_workers=16 \
    --output_dir="$OUT" \
    --seed="$SEED" \
    --wandb.enable=false
echo "[e0_full_expert] lerobot-train exit=$?"
