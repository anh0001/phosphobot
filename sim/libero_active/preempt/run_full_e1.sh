#!/usr/bin/env bash
# Full E1 oracle-handover run: 10 tasks x 5 seeds x 5 variants = 250 rollouts.
# Spec timing: t_p = int(0.35 * 280) = 98.  See REPORT.md "Correctness caveats"
# for why an EARLIER t_p (e.g. --tp-frac 0.14) is recommended to actually
# exercise the override-vs-replan question on libero_spatial.
#
# Run from sim/libero_active:
#   bash preempt/run_full_e1.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
env -u PYTHONPATH MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  python preempt/e1_oracle_handover.py \
    --tasks 0,1,2,3,4,5,6,7,8,9 \
    --seeds 1000,1001,1002,1003,1004 \
    --delta 0.05 --latency 8 --tp-frac 0.35 \
    --out preempt/e1_full_results.json
