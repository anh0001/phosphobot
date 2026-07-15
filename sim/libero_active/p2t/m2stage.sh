#!/usr/bin/env bash
# M2 stage — powered G: M2 (100% synthetic) and N1 (100% syn + equal-count canonical)
# each at 3 training seeds, + primary degradation null (base + action-noise sigma) at
# 3 noise seeds. Resumable via sentinels. Feeds the between-seed G statistic.
set -uo pipefail
cd "$(dirname "$0")/.."
VENV="$PWD/.venv"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=1 CUDA_VISIBLE_DEVICES=1
PY="env -u PYTHONPATH $VENV/bin/python"
SENT=p2t/sentinels; mkdir -p "$SENT" p2t/logs
BASE=results/e0_full_expert_100k/seed0/train/checkpoints/100000/pretrained_model
DS_ROOT=p2t/datasets/p2t_mixed
STEPS=20000
NULL_SIGMA="${NULL_SIGMA:-0.15}"   # clean-closest, highest-u_gap (most conservative) null

mark(){ touch "$SENT/$1"; echo "[m2stage] $1 DONE $(date '+%F %T')"; }
die(){ echo "[m2stage] FATAL: $1 $(date '+%F %T')"; exit 1; }

train(){ # $1=arm(M2|N1) $2=seed $3=episodes.json
  local ARM=$1 SEED=$2 EPS_F=$3 OUT="p2t/ft/${1}_seed${2}/train"
  local EPS; EPS=$(cat "$EPS_F")
  env -u PYTHONPATH PATH="$VENV/bin:$PATH" \
    "$VENV/bin/accelerate" launch --num_processes=1 --mixed_precision=bf16 \
    "$VENV/bin/lerobot-train" --policy.path="$BASE" --policy.push_to_hub=false \
    --dataset.repo_id=local/p2t_mixed --dataset.root="$DS_ROOT" --dataset.episodes="$EPS" \
    --policy.optimizer_lr=1e-4 --policy.scheduler_warmup_steps=500 \
    --policy.scheduler_decay_steps=$STEPS --policy.scheduler_decay_lr=2.5e-6 \
    --policy.freeze_vision_encoder=true --policy.train_expert_only=true \
    --policy.train_state_proj=true --policy.compile_model=false \
    --env.type=libero --env.task=libero_spatial --steps=$STEPS --batch_size=32 \
    --save_freq=$STEPS --eval_freq=0 --num_workers=16 --output_dir="$OUT" \
    --seed="$SEED" --wandb.enable=false > "p2t/logs/m2stage_ft_${ARM}_s${SEED}.log" 2>&1
}
reachf(){ # $1=ckpt $2=out-tag
  $PY preempt/reach_field.py --ckpt "$1" --mags 0.05,0.075 \
    --out "p2t/${2}.jsonl" --log-dir "p2t/${2}_logs" --no-flow-samples \
    > "p2t/logs/m2stage_eval_${2}.log" 2>&1
}

# --- M2 + N1 fine-tunes at seeds 1,2 (seed 0: M2 = round-2 M2_pure; N1 seed0 new) ---
for ARM in M2 N1; do
  EPS_F="p2t/$(echo $ARM | tr A-Z a-z)_episodes.json"
  for SEED in 0 1 2; do
    # M2 seed0 already exists as eval_r2_M2_pure (skip its FT+eval, alias later)
    if [ "$ARM" = M2 ] && [ "$SEED" = 0 ]; then continue; fi
    CK="p2t/ft/${ARM}_seed${SEED}/train/checkpoints/0${STEPS}/pretrained_model"
    if [ ! -f "$SENT/m2s_ft_${ARM}_s${SEED}" ]; then
      train "$ARM" "$SEED" "$EPS_F" || die "train $ARM s$SEED"
      ls "$CK"/*.safetensors >/dev/null 2>&1 || die "no ckpt $ARM s$SEED"
      mark "m2s_ft_${ARM}_s${SEED}"
    fi
    if [ ! -f "$SENT/m2s_eval_${ARM}_s${SEED}" ]; then
      reachf "$CK" "eval_m2s_${ARM}_s${SEED}" || die "eval $ARM s$SEED"
      grep -q REACH_FIELD_DONE "p2t/logs/m2stage_eval_eval_m2s_${ARM}_s${SEED}.log" || die "eval incomplete $ARM s$SEED"
      mark "m2s_eval_${ARM}_s${SEED}"
    fi
  done
done

# --- primary degradation null: base + action-noise NULL_SIGMA, at 3 noise seeds ---
for NS in 1 2 3; do
  if [ ! -f "$SENT/m2s_null_ns${NS}" ]; then
    P2T_ACTION_NOISE=$NULL_SIGMA P2T_NOISE_SEED=$NS \
      $PY preempt/reach_field.py --ckpt "$BASE" --mags 0.05,0.075 \
      --out "p2t/eval_null_s${NULL_SIGMA}_ns${NS}.jsonl" \
      --log-dir "p2t/eval_null_s${NULL_SIGMA}_ns${NS}_logs" --no-flow-samples \
      > "p2t/logs/m2stage_null_ns${NS}.log" 2>&1 || die "null ns$NS"
    grep -q REACH_FIELD_DONE "p2t/logs/m2stage_null_ns${NS}.log" || die "null incomplete ns$NS"
    mark "m2s_null_ns${NS}"
  fi
done

echo "M2STAGE_DONE $(date '+%F %T')"
