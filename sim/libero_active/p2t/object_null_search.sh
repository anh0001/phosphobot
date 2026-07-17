#!/usr/bin/env bash
cd "$(dirname "$0")/.."
VENV="$PWD/.venv"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=1 CUDA_VISIBLE_DEVICES=1 E_SUITE=libero_object
for S in 0.10 0.15 0.20; do
  [ -f p2t/sentinels/objnull_${S} ] && continue
  P2T_ACTION_NOISE=$S env -u PYTHONPATH E_SUITE=libero_object "$VENV/bin/python" preempt/reach_field.py \
    --ckpt results/e0_full_expert_object_100k/seed0/train/checkpoints/100000/pretrained_model --mags 0.05 --src preempt/pair_meta_libero_object.json \
    --out p2t/objnull_s${S}.jsonl --log-dir p2t/objnull_s${S}_logs --no-flow-samples \
    > p2t/logs/objnull_s${S}.log 2>&1 || { echo "FATAL $S"; exit 1; }
  touch p2t/sentinels/objnull_${S}
  n=$(env -u PYTHONPATH "$VENV/bin/python" -c "import json,numpy as np; r=[json.loads(l) for l in open('p2t/objnull_s${S}.jsonl') if l.strip()]; c=[x['success'] for x in r if x.get('cond')=='clean']; print(f'{np.mean(c):.0%}')")
  echo "[objnull] sigma=$S clean=$n (target object M2 ~42%)"
done
echo OBJNULL_SEARCH_DONE
