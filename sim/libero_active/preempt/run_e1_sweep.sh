#!/usr/bin/env bash
# E1 decisive run: disturbance-magnitude sweep for the override-vs-replan question.
# 10 tasks x 5 seeds, mid-task perturbation (tp-frac=0.15), latency L=8, skip full_replan.
# Two nudge magnitudes (5cm, 10cm) to distinguish "preemption never helps on LIBERO"
# from "we picked one bad nudge size" (the C5 phase-diagram question).
set -u
cd /srv/data/users/anhar/codes/phosphobot/sim/libero_active || exit 2
source .venv/bin/activate || { echo "venv FAIL"; exit 2; }
RUN="env -u PYTHONPATH MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python preempt/e1_oracle_handover.py"
COMMON="--tasks 0,1,2,3,4,5,6,7,8,9 --seeds 1000,1001,1002,1003,1004 --latency 8 --tp-frac 0.15 --no-full-replan"

for D in 0.05 0.10; do
  TAG=${D/./}
  echo "================= E1 sweep delta=$D ================="
  $RUN $COMMON --delta "$D" --out "preempt/e1_sweep_d${TAG}.json"
done

echo "================= SWEEP SUMMARY ================="
python3 - <<'PY'
import json,glob
for f in sorted(glob.glob("preempt/e1_sweep_d*.json")):
    d=json.load(open(f)); cfg=d["config"]; arms=d["aggregate"]["arms"]; kc=d["kill_criterion"]
    print(f"\n# {f}  delta={cfg['delta']} t_p={cfg['perturb_step']} n_pairs={d['aggregate']['n_pairs']} n_clean_succ={d['aggregate']['n_clean_success']}")
    for v in ("clean","open_loop","replan_only","preempt_hold"):
        a=arms.get(v,{})
        print(f"  {v:13s} cond={a.get('conditional_perturbed_success')} raw={a.get('raw_perturbed_success')} macro={a.get('avg_macro_calls')} harm={a.get('stale_action_harm')} fired={a.get('n_perturbations_fired_in_clean_pairs')}")
    print(f"  KILL: cond_gain_pp={kc['cond_gain_pp']} harm_reduction_pct={kc['harm_reduction_pct']} PASSES={kc['PASSES']}")
PY
echo "E1_SWEEP_DONE"
