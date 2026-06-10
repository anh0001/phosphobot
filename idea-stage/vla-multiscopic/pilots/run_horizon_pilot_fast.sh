#!/usr/bin/env bash
# Pilot P0 (lean): clean-LIBERO execution-horizon ablation for Direction A.
# All three horizons on the SAME tasks at matched, affordable sample size so n=1
# (re-infer every step) is cheap. n_episodes is PER-TASK in lerobot-eval, so
# N_EPISODES=5 -> 50 rollouts/horizon over libero_spatial's 10 tasks.
set -u
REPO=/srv/data/users/anhar/codes/phosphobot
LA="$REPO/sim/libero_active"
CKPT="$LA/results/fixedbuf_random_N20/seed0/train/checkpoints/004000/pretrained_model"
OUTROOT="$REPO/idea-stage/vla-multiscopic/pilots/horizon_ablation_fast"
SUITE=libero_spatial
N_EPISODES=5
SEED=1000

cd "$LA" || exit 2
# shellcheck disable=SC1091
source .venv/bin/activate || { echo "FATAL: venv activate failed"; exit 2; }
mkdir -p "$OUTROOT"
SUMMARY="$OUTROOT/summary.tsv"
printf "n_action_steps\tpc_success\tn_episodes\treturncode\n" > "$SUMMARY"

for N in 50 8 1; do
  OUT="$OUTROOT/h${N}"; rm -rf "$OUT"
  echo "=================================================================="
  echo "[pilot] n_action_steps=$N suite=$SUITE n_episodes/task=$N_EPISODES seed=$SEED"
  echo "=================================================================="
  env -u PYTHONPATH MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    lerobot-eval --policy.path="$CKPT" --policy.n_action_steps="$N" \
      --env.type=libero --env.task="$SUITE" \
      --eval.batch_size=1 --eval.n_episodes="$N_EPISODES" --env.max_parallel_tasks=1 \
      --output_dir="$OUT" --seed="$SEED"
  RC=$?
  PC=$(python3 - "$OUT" <<'PY'
import json,sys,glob
out=sys.argv[1]; cands=glob.glob(out+"/**/eval_info.json",recursive=True)
pc=float("nan"); n=0
if cands:
    info=json.load(open(cands[0])); agg=info.get("overall") or info.get("aggregated") or info
    pc=float(agg.get("pc_success",float("nan"))); n=int(agg.get("n_episodes",0)) or len(info.get("per_task",[]))
print(f"{pc}\t{n}")
PY
)
  echo "[pilot] n=$N -> pc_success=$PC rc=$RC"
  printf "%s\t%s\t%s\n" "$N" "$PC" "$RC" >> "$SUMMARY"
done
echo "[pilot] DONE. Summary:"; cat "$SUMMARY"
echo "PILOT_HORIZON_DONE"
