#!/usr/bin/env bash
# 2nd-suite (libero_object) eval, auto-run when the object ckpt lands.
# E_SUITE=libero_object -> harness_lib uses libero_object envs; --src = the new pair metadata.
set -uo pipefail
cd "$(dirname "$0")/.."
VENV="$PWD/.venv"; CK="results/e0_full_expert_object_100k/seed0/train/checkpoints/100000/pretrained_model"
SRC=preempt/pair_meta_libero_object.json
export E_SUITE=libero_object MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
while true; do
  if ls "$CK"/*.safetensors >/dev/null 2>&1 && ! pgrep -f "object_100k/seed0/train" >/dev/null 2>&1; then break; fi
  sleep 120
done
echo "[objchain] ckpt ready -> clean eval"
env -u PYTHONPATH "$VENV/bin/python" preempt/eval_clean.py --ckpt "$CK" --src "$SRC" \
  --out preempt/e0_eval_object_100k.jsonl > preempt/obj_clean.run.log 2>&1
echo "[objchain] -> E1r-50mm (keystone on suite 2)"
env -u PYTHONPATH "$VENV/bin/python" preempt/reach_field.py --ckpt "$CK" --src "$SRC" \
  --tasks 0,1,2,3,4,5,6,7,8,9 --seeds 1000,1001,1002,1003,1004 \
  --mags 0.05 --dirs px,nx,py,ny --no-flow-samples \
  --out preempt/e1r_object_50mm.jsonl --log-dir preempt/e1r_object_50mm_logs > preempt/obj_e1r.run.log 2>&1
echo "[objchain] -> E2-CT subset (dissociation on suite 2)"
env -u PYTHONPATH "$VENV/bin/python" preempt/object_local_paste.py --ckpt "$CK" --src "$SRC" \
  --pairs-from preempt/e0_eval_object_100k.jsonl --mags 0.05 --dirs px,py \
  --conditions CC,CT,TC --edit-cams image,image2 \
  --out preempt/e2r_object_suite_50mm.jsonl --log-dir preempt/e2r_object_suite_logs > preempt/obj_e2r.run.log 2>&1
echo "OBJ_EVAL_CHAIN_DONE"
