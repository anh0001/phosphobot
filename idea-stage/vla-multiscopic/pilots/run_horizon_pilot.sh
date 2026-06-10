#!/usr/bin/env bash
# Pilot P0 for Direction A ("Override, not replan").
# Execution-horizon ablation: vary SmolVLA's open-loop commit horizon n_action_steps
# in {50 (full open-loop chunk), 8, 1 (full closed-loop, re-infer every step)} and
# measure LIBERO success. This UPPER-BOUNDS what any override/replan controller could
# buy on CLEAN (un-perturbed) LIBERO:
#   - n=1 >> n=50  -> closed-loop reactivity helps even on static LIBERO -> headroom for
#                     selective override (capture most of n=1's gain at ~n=50's cost). GREEN.
#   - n=1 ~= n=50  -> no clean headroom -> reactivity only bites under disturbance ->
#                     the perturbation protocol is mandatory (informs the plan). REDIRECT.
# Training-free, reuses LeRobot's own (correct) rollout. Target <= 2 GPU-h.
set -u

REPO=/srv/data/users/anhar/codes/phosphobot
LA="$REPO/sim/libero_active"
CKPT="$LA/results/fixedbuf_random_N20/seed0/train/checkpoints/004000/pretrained_model"
OUTROOT="$REPO/idea-stage/vla-multiscopic/pilots/horizon_ablation"
SUITE=libero_spatial
N_EPISODES=20
SEED=1000

cd "$LA" || exit 2
# shellcheck disable=SC1091
source .venv/bin/activate || { echo "FATAL: venv activate failed"; exit 2; }

mkdir -p "$OUTROOT"
SUMMARY="$OUTROOT/summary.tsv"
printf "n_action_steps\tpc_success\tn_episodes\treturncode\n" > "$SUMMARY"

for N in 50 8 1; do
  OUT="$OUTROOT/h${N}"
  rm -rf "$OUT"   # LeRobot refuses an existing output dir
  echo "=================================================================="
  echo "[pilot] n_action_steps=$N  suite=$SUITE  n_episodes=$N_EPISODES  seed=$SEED"
  echo "=================================================================="
  env -u PYTHONPATH MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    lerobot-eval \
      --policy.path="$CKPT" \
      --policy.n_action_steps="$N" \
      --env.type=libero \
      --env.task="$SUITE" \
      --eval.batch_size=1 \
      --eval.n_episodes="$N_EPISODES" \
      --env.max_parallel_tasks=1 \
      --output_dir="$OUT" \
      --seed="$SEED"
  RC=$?
  PC=$(python3 - "$OUT" <<'PY'
import json,sys,glob,math
out=sys.argv[1]
cands=glob.glob(out+"/**/eval_info.json",recursive=True)
pc=float("nan"); n=0
if cands:
    info=json.load(open(cands[0]))
    agg=info.get("overall") or info.get("aggregated") or info
    pc=float(agg.get("pc_success",float("nan")))
    n=int(agg.get("n_episodes",0)) or len(info.get("per_task",[]))
print(f"{pc}\t{n}")
PY
)
  echo -e "[pilot] n=$N -> pc_success=$PC rc=$RC"
  printf "%s\t%s\t%s\n" "$N" "$PC" "$RC" >> "$SUMMARY"
done

echo "=================================================================="
echo "[pilot] DONE. Summary:"
cat "$SUMMARY"
echo "PILOT_HORIZON_DONE"
