"""Stage 1 de-risk: plain chunked rollout on one task with manual chunk control.

Confirms the inference pipeline is wired correctly (success ~20-40% over a few eps)
BEFORE we add perturbations / variants. Mirrors lerobot_eval.rollout step-for-step
but uses predict_action_chunk + our own chunk execution loop on a single in-process
(sync) LiberoEnv, with manual batch-dim insertion so preprocess_observation works.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

# allow `from preempt.harness_lib import ...`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preempt.harness_lib import (
    CKPT,
    build_pipeline,
    make_single_env,
    plain_chunked_rollout,
)


def main() -> None:
    task_id = 0
    n_eps = 4
    base_seed = 1000

    print(f"[stage1] building pipeline from {CKPT}")
    pipe = build_pipeline()
    print("[stage1] pipeline ready. chunk_size=", pipe.policy.config.chunk_size,
          "n_action_steps=", pipe.policy.config.n_action_steps)

    successes = []
    for ep in range(n_eps):
        seed = base_seed + ep
        env = make_single_env(task_id=task_id, episode_index=ep)
        res = plain_chunked_rollout(env, pipe, seed=seed, max_steps_cap=None)
        env.close()
        successes.append(res["success"])
        print(f"[stage1] ep={ep} seed={seed} task='{res['task']}' "
              f"success={res['success']} steps={res['n_steps']} macro_calls={res['macro_calls']}")

    rate = float(np.mean(successes)) if successes else 0.0
    print(f"\n[stage1] success_rate over {n_eps} eps = {rate*100:.1f}%  ({sum(successes)}/{n_eps})")
    print("STAGE1_OK")


if __name__ == "__main__":
    main()
