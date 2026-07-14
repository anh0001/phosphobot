#!/usr/bin/env bash
# P2T ROUND 2 — constraint-elevation experiment (pre-registered 2026-07-12).
#
# Round-1 finding: ALL conditions kept median proj ~= 1.0 (fully anchored
# steering) at 50 AND 75 mm — success gains came without any tracking. The
# binding constraint is NOT the acquisition signal. Three hypotheses:
#   H1 mixture ratio   (83% originals still teach canonical; signal drowned)
#   H2 plasticity locus (VLM+vision frozen; gain cannot be wired via expert)
#   H3 representation  (chunked expert cannot express conditional retarget)
# Arms (all 20k steps from the retrained strong ckpt):
#   M1_ratio50 : A's 88 synthetic + seeded 88-subsample of originals (50/50),
#                expert-only.            -> tests H1 at fixed synthetic count
#   M2_pure    : ALL 352 pooled synthetic, 0 originals, expert-only.
#                                        -> max counterfactual signal, H1 strong form
#   M3_plastic : same 352 synthetic, train_expert_only=false +
#                freeze_vision_encoder=false (batch 16).  -> tests H2
# Readout: reach_field --mags 0.05,0.075 (450 rollouts/arm).
# PRIMARY metric = median proj (steering), NOT success.
# Gate "steering acquired": median proj@50mm <= 0.70.
#   H1 confirmed if M1 or M2 crosses; H2 if only M3 crosses; H3 supported if none.
set -uo pipefail
cd "$(dirname "$0")/.."
VENV="$PWD/.venv"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=1 CUDA_VISIBLE_DEVICES=1
PY="env -u PYTHONPATH $VENV/bin/python"
SENT=p2t/sentinels; mkdir -p "$SENT" p2t/logs
STRONG=results/e0_full_expert_100k/seed0/train/checkpoints/100000/pretrained_model
DS_ROOT=p2t/datasets/p2t_mixed
FT_STEPS=20000

stage() { [ -f "$SENT/$1" ]; }
mark()  { touch "$SENT/$1"; echo "[round2] stage $1 DONE $(date '+%F %T')"; }
die()   { echo "[round2] FATAL: $1 $(date '+%F %T')"; exit 1; }

episodes_r2() {  # $1 = arm
  $PY - "$1" <<'EOF'
import json, sys
import numpy as np
m = json.load(open("p2t/datasets/p2t_mixed/p2t_manifest.json"))
arm = sys.argv[1]
if arm == "M1_ratio50":
    rng = np.random.default_rng(1)
    orig = sorted(rng.choice(m["originals"], size=88, replace=False).tolist())
    eps = orig + m["conditions"]["A_gain"]
else:  # M2_pure / M3_plastic: all pooled synthetic
    eps = sorted(e for c in m["conditions"].values() for e in c)
print(json.dumps(sorted(eps), separators=(",", ":")))
EOF
}

train_arm() {  # $1=arm $2=batch $3=extra policy flags (string)
  local ARM=$1 BATCH=$2 EXTRA=$3
  local OUT="p2t/ft/$ARM/train"
  local EPS; EPS=$(episodes_r2 "$ARM")
  # shellcheck disable=SC2086
  env -u PYTHONPATH PATH="$VENV/bin:$PATH" \
    "$VENV/bin/accelerate" launch --num_processes=1 --mixed_precision=bf16 \
    "$VENV/bin/lerobot-train" \
    --policy.path="$STRONG" \
    --policy.push_to_hub=false \
    --dataset.repo_id=local/p2t_mixed \
    --dataset.root="$DS_ROOT" \
    --dataset.episodes="$EPS" \
    --policy.optimizer_lr=1e-4 \
    --policy.scheduler_warmup_steps=500 \
    --policy.scheduler_decay_steps=$FT_STEPS \
    --policy.scheduler_decay_lr=2.5e-6 \
    $EXTRA \
    --policy.train_state_proj=true \
    --policy.compile_model=false \
    --env.type=libero --env.task=libero_spatial \
    --steps=$FT_STEPS --batch_size="$BATCH" --save_freq=$FT_STEPS --eval_freq=0 \
    --num_workers=16 --output_dir="$OUT" --seed=0 --wandb.enable=false \
    > "p2t/logs/r2_ft_$ARM.log" 2>&1
}

for SPEC in "M1_ratio50|32|--policy.freeze_vision_encoder=true --policy.train_expert_only=true" \
            "M2_pure|32|--policy.freeze_vision_encoder=true --policy.train_expert_only=true" \
            "M3_plastic|16|--policy.freeze_vision_encoder=false --policy.train_expert_only=false"; do
  ARM=${SPEC%%|*}; REST=${SPEC#*|}; BATCH=${REST%%|*}; EXTRA=${REST#*|}
  if ! stage "r2_ft_$ARM"; then
    train_arm "$ARM" "$BATCH" "$EXTRA" || true
    if ! ls "p2t/ft/$ARM/train/checkpoints/0$FT_STEPS/pretrained_model"/*.safetensors >/dev/null 2>&1; then
      if [ "$ARM" = "M3_plastic" ]; then
        echo "[round2] M3 failed at batch 16 — retrying batch 8"
        rm -rf "p2t/ft/$ARM/train"
        train_arm "$ARM" 8 "$EXTRA" || true
      fi
    fi
    ls "p2t/ft/$ARM/train/checkpoints/0$FT_STEPS/pretrained_model"/*.safetensors >/dev/null 2>&1 \
      || die "fine-tune $ARM produced no ckpt"
    mark "r2_ft_$ARM"
  fi
done

for ARM in M1_ratio50 M2_pure M3_plastic; do
  if ! stage "r2_eval_$ARM"; then
    CKPT="p2t/ft/$ARM/train/checkpoints/0$FT_STEPS/pretrained_model"
    $PY preempt/reach_field.py --ckpt "$CKPT" --mags 0.05,0.075 \
      --out "p2t/eval_r2_$ARM.jsonl" --log-dir "p2t/eval_r2_${ARM}_logs" \
      --no-flow-samples > "p2t/logs/r2_eval_$ARM.log" 2>&1 || die "eval $ARM failed"
    grep -q "REACH_FIELD_DONE" "p2t/logs/r2_eval_$ARM.log" || die "eval $ARM incomplete"
    mark "r2_eval_$ARM"
  fi
done

echo "P2T_ROUND2_DONE $(date '+%F %T')"
