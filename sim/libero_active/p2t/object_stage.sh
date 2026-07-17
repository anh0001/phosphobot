#!/usr/bin/env bash
# M3 object pipeline (phases 1-4): synthesis -> dataset -> M2/N1 fine-tune -> readouts.
# Runs on GPU 0 (ACT base occupies GPU 1). E_SUITE=libero_object throughout.
# First pass at 1 seed (cost control); add seeds after the effect is confirmed.
# Degradation null handled separately after M2 competence is known (mirrors spatial).
set -uo pipefail
cd "$(dirname "$0")/.."
VENV="$PWD/.venv"
DEMO=/srv/storage/roboserver1/home/anhar/.cache/huggingface/hub/datasets--yifengzhu-hf--LIBERO-datasets/snapshots/f13aa24a3da8c43c7225569f28c562979fa0e35a/libero_object
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0
export E_SUITE=libero_object P2T_DEMO_DIR="$DEMO" P2T_OBJ_SRC=preempt/pair_meta_libero_object.json
PY="env -u PYTHONPATH E_SUITE=libero_object P2T_DEMO_DIR=$DEMO P2T_OBJ_SRC=preempt/pair_meta_libero_object.json $VENV/bin/python"
SENT=p2t/sentinels; mkdir -p "$SENT" p2t/logs
OBJBASE=results/e0_full_expert_object_100k/seed0/train/checkpoints/100000/pretrained_model
DS=p2t/datasets/p2t_object; STEPS=20000
mark(){ touch "$SENT/$1"; echo "[obj] $1 DONE $(date '+%F %T')"; }
die(){ echo "[obj] FATAL: $1 $(date '+%F %T')"; exit 1; }

# --- phase 1: synthesis (4 conditions) ---
if ! [ -f "$SENT/obj_synth" ]; then
  for C in A_gain C_uniform D_failure E_sham; do
    $PY p2t/synthesize_episodes.py --alloc p2t/acquisition_maps_object.json \
      --condition "$C" --out-dir p2t/staging_object --max-attempts 5 \
      > "p2t/logs/obj_synth_$C.log" 2>&1 || die "synth $C"
    grep -q SYNTH_DONE "p2t/logs/obj_synth_$C.log" || die "synth incomplete $C"
  done
  mark obj_synth
fi

# --- phase 2: dataset build (originals from object suite + synthetic) ---
if ! [ -f "$SENT/obj_dataset" ]; then
  $PY p2t/make_mixed_dataset.py --staging p2t/staging_object --out-root p2t/datasets \
    --name p2t_object > p2t/logs/obj_dataset.log 2>&1 || die "dataset"
  grep -q DATASET_DONE p2t/logs/obj_dataset.log || die "dataset incomplete"
  mark obj_dataset
fi

# episode lists: M2=all synthetic; N1=synthetic + equal-count canonical
$PY - <<PYEOF
import json, numpy as np
m=json.load(open("p2t/datasets/p2t_object/p2t_manifest.json"))
syn=sorted(e for c in m["conditions"].values() for e in c)
orig=m["originals"]; rng=np.random.default_rng(0)
n1o=sorted(rng.choice(orig,size=min(len(syn),len(orig)),replace=False).tolist())
open("p2t/obj_m2_episodes.json","w").write(json.dumps(syn,separators=(',',':')))
open("p2t/obj_n1_episodes.json","w").write(json.dumps(sorted(syn+n1o),separators=(',',':')))
print("obj M2",len(syn),"N1",len(syn)+len(n1o))
PYEOF

# --- phase 3: M2/N1 fine-tunes (seed 0) from object base ---
train(){ local ARM=$1 EPS_F=$2 OUT="p2t/ft/OBJ_${ARM}_seed0/train"
  local EPS; EPS=$(cat "$EPS_F")
  env -u PYTHONPATH PATH="$VENV/bin:$PATH" E_SUITE=libero_object \
    "$VENV/bin/accelerate" launch --num_processes=1 --mixed_precision=bf16 \
    "$VENV/bin/lerobot-train" --policy.path="$OBJBASE" --policy.push_to_hub=false \
    --dataset.repo_id=local/p2t_object --dataset.root="$DS" --dataset.episodes="$EPS" \
    --policy.optimizer_lr=1e-4 --policy.scheduler_warmup_steps=500 \
    --policy.scheduler_decay_steps=$STEPS --policy.scheduler_decay_lr=2.5e-6 \
    --policy.freeze_vision_encoder=true --policy.train_expert_only=true \
    --policy.train_state_proj=true --policy.compile_model=false \
    --env.type=libero --env.task=libero_object --steps=$STEPS --batch_size=32 \
    --save_freq=$STEPS --eval_freq=0 --num_workers=16 --output_dir="$OUT" \
    --seed=0 --wandb.enable=false > "p2t/logs/obj_ft_${ARM}.log" 2>&1
}
reachf(){ $PY preempt/reach_field.py --ckpt "$1" --mags 0.05,0.075 \
    --src preempt/pair_meta_libero_object.json --out "p2t/${2}.jsonl" \
    --log-dir "p2t/${2}_logs" --no-flow-samples > "p2t/logs/obj_eval_${2}.log" 2>&1; }
for ARM in M2 N1; do
  EPS_F="p2t/obj_$(echo $ARM|tr A-Z a-z)_episodes.json"
  CK="p2t/ft/OBJ_${ARM}_seed0/train/checkpoints/0${STEPS}/pretrained_model"
  if ! [ -f "$SENT/obj_ft_${ARM}" ]; then train "$ARM" "$EPS_F" || die "ft $ARM"
    ls "$CK"/*.safetensors >/dev/null 2>&1 || die "no ckpt $ARM"; mark "obj_ft_${ARM}"; fi
  if ! [ -f "$SENT/obj_eval_${ARM}" ]; then reachf "$CK" "eval_obj_${ARM}" || die "eval $ARM"
    grep -q REACH_FIELD_DONE "p2t/logs/obj_eval_eval_obj_${ARM}.log" || die "eval incomplete $ARM"
    mark "obj_eval_${ARM}"; fi
done
echo "OBJECT_STAGE_P1_4_DONE $(date '+%F %T')"
