"""Training-free keystone control: representation failure vs control failure.

Substitute for the privileged pose-token probe (whose training data lacks object
poses). Tests the SAME hypothesis directly: after a 5cm object displacement at
episode start, does the policy reach where the object NOW IS (true), or where it
canonically EXPECTS it (orig = pre-displacement)? Reaching 'orig not true' = the
policy ignores the visually-displaced object = perception/localization failure
(representation), not a control/timing failure.

For each clean-success (task,seed) pair (from e1_recovery_d005.json) we run a
perturbed-at-start rollout with normal chunked execution and log the eef XY
trajectory, then compare closest approach to the true vs original object XY.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preempt.harness_lib import (  # noqa: E402
    build_pipeline, make_single_env, build_policy_batch, execute_action, _refresh_obs,
)
from preempt.perturb import perturb_object_by_name  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402

SRC = "preempt/e1_recovery_d005.json"
SEEDS = [1000, 1001, 1002, 1003, 1004]
TASKS = list(range(10))
DELTA = 0.05
GRASP_XY = 0.030  # within 3cm in XY = "reached" that location
OUT = "preempt/mislocalization_probe.json"


def _eef_xy(raw_obs) -> np.ndarray:
    return np.asarray(raw_obs["robot_state"]["eef"]["pos"], dtype=np.float64)[:2]


def run_logged_perturbed_start(pipe, env, seed, obj_name, delta):
    set_seed(seed)
    pipe.policy.reset()
    raw_obs, info = env.reset(seed=seed)
    task_desc = env.task_description
    pinfo = perturb_object_by_name(env, obj_name, delta=delta)
    raw_obs = _refresh_obs(env)
    obj_orig = np.asarray(pinfo["before_xyz"], dtype=np.float64)[:2]
    obj_true = np.asarray(pinfo["after_xyz"], dtype=np.float64)[:2]

    max_steps = env._max_episode_steps
    n_action_steps = pipe.policy.config.n_action_steps
    chunk = None; chunk_pos = 0; success = False; eef_traj = []
    for step in range(max_steps):
        eef_traj.append(_eef_xy(raw_obs))
        if chunk is None or chunk_pos >= n_action_steps:
            batch = build_policy_batch(pipe, raw_obs, task_desc)
            with torch.inference_mode():
                chunk = pipe.policy.predict_action_chunk(batch)
            chunk_pos = 0
        a = chunk[:, chunk_pos, :]; chunk_pos += 1
        raw_obs, reward, terminated, info = execute_action(pipe, env, a)
        if bool(info.get("is_success", False)):
            success = True
        if terminated:
            break
    eef = np.asarray(eef_traj)
    d_true = float(np.min(np.linalg.norm(eef - obj_true, axis=1)))
    d_orig = float(np.min(np.linalg.norm(eef - obj_orig, axis=1)))
    return {"success": success, "min_d_true": d_true, "min_d_orig": d_orig,
            "reached_true": d_true <= GRASP_XY, "reached_orig": d_orig <= GRASP_XY,
            "closer_to_orig": d_orig < d_true, "obj": obj_name}


def main():
    src = json.load(open(SRC))
    pairs = {}
    for r in src["records"]:
        pairs.setdefault((r["task_id"], r["seed"]), {})[r["variant"]] = r
    clean_keys = [k for k, v in pairs.items() if v.get("clean", {}).get("success")]
    obj_map = {}
    for k, v in pairs.items():
        for vary in ("open_loop", "replan_only", "preempt_hold", "recovery_probe"):
            o = v.get(vary, {}).get("perturb", {}).get("object")
            if o:
                obj_map[k] = o; break

    pipe = build_pipeline()
    recs = []
    for (task_id, seed) in clean_keys:
        obj = obj_map.get((task_id, seed))
        if obj is None:
            continue
        env = make_single_env(task_id=task_id, episode_index=SEEDS.index(seed))
        r = run_logged_perturbed_start(pipe, env, seed, obj, DELTA)
        env.close()
        r.update({"task_id": task_id, "seed": seed})
        recs.append(r)
        print(f"[mislocal] task{task_id} seed{seed} {obj[:20]:20s} succ={r['success']} "
              f"d_true={r['min_d_true']*100:.1f}cm d_orig={r['min_d_orig']*100:.1f}cm "
              f"reached_true={r['reached_true']} reached_orig={r['reached_orig']} closer_orig={r['closer_to_orig']}")

    n = len(recs)
    fail = [r for r in recs if not r["success"]]
    out = {
        "n_clean_success_pairs": n,
        "grasp_xy_thresh_m": GRASP_XY,
        "perturbed_start_success_rate": sum(r["success"] for r in recs) / n if n else None,
        "mean_min_d_true_cm": float(np.mean([r["min_d_true"] for r in recs]) * 100) if n else None,
        "mean_min_d_orig_cm": float(np.mean([r["min_d_orig"] for r in recs]) * 100) if n else None,
        "frac_reached_true": sum(r["reached_true"] for r in recs) / n if n else None,
        "frac_reached_orig": sum(r["reached_orig"] for r in recs) / n if n else None,
        "frac_closer_to_orig_than_true": sum(r["closer_to_orig"] for r in recs) / n if n else None,
        "among_failures": {
            "n": len(fail),
            "frac_closer_to_orig": (sum(r["closer_to_orig"] for r in fail) / len(fail)) if fail else None,
            "frac_reached_orig_not_true": (sum(r["reached_orig"] and not r["reached_true"] for r in fail) / len(fail)) if fail else None,
            "mean_min_d_true_cm": float(np.mean([r["min_d_true"] for r in fail]) * 100) if fail else None,
        },
        "records": recs,
    }
    Path(OUT).write_text(json.dumps(out, indent=2))
    print("\n==== MISLOCALIZATION PROBE ====")
    print(f"n={n}  perturbed_start success={out['perturbed_start_success_rate']*100:.1f}%")
    print(f"mean closest approach: to TRUE obj = {out['mean_min_d_true_cm']:.1f}cm | to ORIG(canonical) = {out['mean_min_d_orig_cm']:.1f}cm")
    print(f"reached TRUE (<3cm): {out['frac_reached_true']*100:.0f}%  |  reached ORIG: {out['frac_reached_orig']*100:.0f}%  |  closer to ORIG than TRUE: {out['frac_closer_to_orig_than_true']*100:.0f}%")
    if fail:
        af = out["among_failures"]
        print(f"among FAILURES (n={af['n']}): closer-to-ORIG={af['frac_closer_to_orig']*100:.0f}%  reached-ORIG-not-TRUE={af['frac_reached_orig_not_true']*100:.0f}%  mean d_true={af['mean_min_d_true_cm']:.1f}cm")
    print("MISLOCAL_PROBE_DONE")


if __name__ == "__main__":
    main()
