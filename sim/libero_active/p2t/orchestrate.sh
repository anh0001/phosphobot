#!/usr/bin/env bash
# P2T pipeline orchestrator — chains all stages after the e0 retrain completes.
# Every stage is resumable; a sentinel file under p2t/sentinels/ marks completion.
# Run:  nohup bash p2t/orchestrate.sh > p2t/orchestrate.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."   # sim/libero_active
VENV="$PWD/.venv"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=1
export CUDA_VISIBLE_DEVICES=1
PY="env -u PYTHONPATH $VENV/bin/python"
SENT=p2t/sentinels; mkdir -p "$SENT" p2t/logs
STRONG=results/e0_full_expert_100k/seed0/train/checkpoints/100000/pretrained_model
DS_ROOT=p2t/datasets/p2t_mixed
N_SYN=128
FT_STEPS=20000

stage() { [ -f "$SENT/$1" ]; }
mark()  { touch "$SENT/$1"; echo "[orchestrate] stage $1 DONE $(date '+%F %T')"; }
die()   { echo "[orchestrate] FATAL: $1 $(date '+%F %T')"; exit 1; }

echo "[orchestrate] start $(date '+%F %T')"

# ---- stage 0: wait for the e0 retrain to produce the 100k checkpoint ----
if ! stage s0_ckpt; then
  echo "[orchestrate] waiting for $STRONG ..."
  for i in $(seq 1 1800); do  # up to 30 h (measured rate ~1.8 step/s, not 2.8)
    ls "$STRONG"/*.safetensors >/dev/null 2>&1 && break
    if ! pgrep -f "lerobot-train" >/dev/null 2>&1; then
      ls "$STRONG"/*.safetensors >/dev/null 2>&1 || die "training died without 100k ckpt (see results/e0_retrain_100k.log)"
    fi
    sleep 60
  done
  ls "$STRONG"/*.safetensors >/dev/null 2>&1 || die "timeout waiting for 100k ckpt"
  mark s0_ckpt
fi

# ---- stage 1: clean-success gate on the retrained ckpt (>=60% of 50) ----
if ! stage s1_gate; then
  $PY preempt/eval_clean.py --ckpt "$STRONG" --out p2t/e0_retrain_clean.jsonl \
    > p2t/logs/s1_gate.log 2>&1 || die "eval_clean failed"
  OK=$($PY - <<'EOF'
import json
recs=[json.loads(l) for l in open('p2t/e0_retrain_clean.jsonl')]
n=sum(1 for r in recs if r.get('success')); print(n)
EOF
)
  echo "[orchestrate] clean gate: $OK/50"
  [ "$OK" -ge 30 ] || die "retrained ckpt too weak ($OK/50 < 30) — review before continuing"
  mark s1_gate
fi

# ---- stage 2: reach-field grid on the retrained ckpt (baseline + gain field) ----
if ! stage s2_field; then
  $PY preempt/reach_field.py --ckpt "$STRONG" \
    --out p2t/reach_field_retrain.jsonl --log-dir p2t/reach_field_retrain_logs \
    --no-flow-samples > p2t/logs/s2_field.log 2>&1 || die "reach_field failed"
  grep -q "REACH_FIELD_DONE" p2t/logs/s2_field.log || die "reach_field incomplete"
  mark s2_field
fi

# ---- stage 3: acquisition maps from the NEW field ----
if ! stage s3_maps; then
  $PY p2t/acquisition.py --in p2t/reach_field_retrain.jsonl --n-syn $N_SYN \
    --auto-sharpen --out p2t/acquisition_maps.json \
    > p2t/logs/s3_maps.log 2>&1 || die "acquisition failed"
  mark s3_maps
fi

# ---- stage 4: synthesis, 4 conditions (env-only; no policy) ----
if ! stage s4_synth; then
  for COND in A_gain C_uniform D_failure E_sham; do
    $PY p2t/synthesize_episodes.py --alloc p2t/acquisition_maps.json \
      --condition "$COND" --out-dir p2t/staging --max-attempts 5 \
      $( [ "$COND" = A_gain ] && echo --clean-check 2 ) \
      > "p2t/logs/s4_synth_$COND.log" 2>&1 || die "synthesis $COND failed"
    grep -q "SYNTH_DONE" "p2t/logs/s4_synth_$COND.log" || die "synthesis $COND incomplete"
  done
  mark s4_synth
fi

# ---- stage 5: one mixed dataset (originals + all synthetic) ----
if ! stage s5_dataset; then
  $PY p2t/make_mixed_dataset.py --staging p2t/staging --out-root p2t/datasets \
    --name p2t_mixed > p2t/logs/s5_dataset.log 2>&1 || die "dataset build failed"
  grep -q "DATASET_DONE" p2t/logs/s5_dataset.log || die "dataset incomplete"
  mark s5_dataset
fi

# ---- stage 6: five fine-tunes (matched budget) ----
episodes_for() {  # $1 = condition (B_noaug uses originals only)
  $PY - "$1" <<'EOF'
import json, sys
m = json.load(open("p2t/datasets/p2t_mixed/p2t_manifest.json"))
eps = list(m["originals"])
if sys.argv[1] != "B_noaug":
    eps += m["conditions"][sys.argv[1]]
print(json.dumps(sorted(eps), separators=(",", ":")))
EOF
}
for COND in A_gain B_noaug C_uniform D_failure E_sham; do
  if ! stage "s6_ft_$COND"; then
    EPS=$(episodes_for "$COND")
    OUT="p2t/ft/$COND/train"
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
      --policy.freeze_vision_encoder=true \
      --policy.train_expert_only=true \
      --policy.train_state_proj=true \
      --policy.compile_model=false \
      --env.type=libero --env.task=libero_spatial \
      --steps=$FT_STEPS --batch_size=32 --save_freq=$FT_STEPS --eval_freq=0 \
      --num_workers=16 --output_dir="$OUT" --seed=0 --wandb.enable=false \
      > "p2t/logs/s6_ft_$COND.log" 2>&1 || die "fine-tune $COND failed"
    ls "$OUT/checkpoints/0$FT_STEPS/pretrained_model"/*.safetensors >/dev/null 2>&1 \
      || die "fine-tune $COND produced no ckpt"
    mark "s6_ft_$COND"
  fi
done

# ---- stage 7: reach-field eval per condition (50 mm, 4 dirs, all pairs) ----
for COND in A_gain B_noaug C_uniform D_failure E_sham; do
  if ! stage "s7_eval_$COND"; then
    CKPT="p2t/ft/$COND/train/checkpoints/0$FT_STEPS/pretrained_model"
    $PY preempt/reach_field.py --ckpt "$CKPT" --mags 0.05 \
      --out "p2t/eval_$COND.jsonl" --log-dir "p2t/eval_${COND}_logs" \
      --no-flow-samples > "p2t/logs/s7_eval_$COND.log" 2>&1 || die "eval $COND failed"
    grep -q "REACH_FIELD_DONE" "p2t/logs/s7_eval_$COND.log" || die "eval $COND incomplete"
    mark "s7_eval_$COND"
  fi
done

# ---- stage 8: kill-gate analysis ----
if ! stage s8_analysis; then
  $PY p2t/analyze_conditions.py > p2t/logs/s8_analysis.log 2>&1 || die "analysis failed"
  cat p2t/logs/s8_analysis.log
  mark s8_analysis
fi

echo "P2T_ORCHESTRATE_DONE $(date '+%F %T')"
