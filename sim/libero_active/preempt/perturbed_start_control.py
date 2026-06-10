"""Crux control: is the 5cm-displaced object pose solvable IN PRINCIPLE from scratch?

For each (task,seed) we displace the SAME object the mid-rollout disturbance targeted
(read from e1_recovery_d005.json) but at EPISODE START, then let SmolVLA solve normally.

Interpretation (on the clean-success conditioning pairs):
  perturbed_start >> mid-rollout arms  -> displaced pose IS solvable from scratch; the
      mid-rollout failure is about RECOVERY, not the pose -> "lacks recovery manifold" holds.
  perturbed_start ~= mid-rollout arms  -> the displaced pose is just hard regardless; the
      claim downgrades to object-pose fragility (weaker / known).
"""
from __future__ import annotations
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preempt.harness_lib import build_pipeline, make_single_env, run_variant  # noqa: E402

SRC = "preempt/e1_recovery_d005.json"
SEEDS = [1000, 1001, 1002, 1003, 1004]
TASKS = list(range(10))
DELTA = 0.05
OUT = "preempt/perturbed_start_control.json"


def main() -> None:
    src = json.load(open(SRC))
    # per-(task,seed): clean success + the object the mid-rollout perturbation hit
    pairs: dict[tuple[int, int], dict] = {}
    for r in src["records"]:
        pairs.setdefault((r["task_id"], r["seed"]), {})[r["variant"]] = r
    clean_succ = {k: v["clean"]["success"] for k, v in pairs.items() if "clean" in v}
    obj_map = {}
    for k, v in pairs.items():
        for vary in ("open_loop", "replan_only", "preempt_hold", "recovery_probe"):
            o = v.get(vary, {}).get("perturb", {}).get("object")
            if o:
                obj_map[k] = o
                break

    pipe = build_pipeline()
    recs = []
    for ti, task_id in enumerate(TASKS):
        for si, seed in enumerate(SEEDS):
            key = (task_id, seed)
            obj = obj_map.get(key)
            if obj is None:
                continue
            env = make_single_env(task_id=task_id, episode_index=si)
            r = run_variant(env, pipe, seed=seed, variant="perturbed_start",
                            perturb_step=0, latency=0, delta=DELTA,
                            start_perturb_object=obj)
            env.close()
            recs.append({"task_id": task_id, "seed": seed, "object": obj,
                         "success": r["success"], "n_steps": r["n_steps"],
                         "clean_success": clean_succ.get(key)})
            print(f"[ctrl] task={task_id} seed={seed} obj={obj[:24]:24s} "
                  f"pstart_succ={r['success']} clean_succ={clean_succ.get(key)} steps={r['n_steps']}")

    n = len(recs)
    raw = sum(x["success"] for x in recs) / n if n else float("nan")
    cond_keys = [x for x in recs if x["clean_success"]]
    cond = (sum(x["success"] for x in cond_keys) / len(cond_keys)) if cond_keys else float("nan")
    # mid-rollout arms' conditional success (from source) for direct comparison
    src_arms = src["aggregate"]["arms"]
    out = {
        "n": n,
        "perturbed_start_raw_success": raw,
        "clean_raw_success": src["aggregate"]["arms"]["clean"]["raw_perturbed_success"],
        "perturbed_start_conditional_success": cond,
        "n_clean_success": len(cond_keys),
        "n_clean_success_that_survive_start_displacement": sum(x["success"] for x in cond_keys),
        "compare_conditional": {
            "perturbed_start": cond,
            "open_loop_midroll": src_arms["open_loop"]["conditional_perturbed_success"],
            "recovery_probe_midroll": src_arms["recovery_probe"]["conditional_perturbed_success"],
            "preempt_hold_midroll": src_arms["preempt_hold"]["conditional_perturbed_success"],
        },
        "records": recs,
    }
    Path(OUT).write_text(json.dumps(out, indent=2))
    print("\n==== CRUX CONTROL RESULT ====")
    print(f"perturbed_start: raw={raw*100:.1f}%  conditional(|clean succ)={cond*100:.1f}% "
          f"({out['n_clean_success_that_survive_start_displacement']}/{len(cond_keys)})")
    print(f"vs mid-rollout conditional: open_loop={src_arms['open_loop']['conditional_perturbed_success']*100:.1f}% "
          f"recovery_probe={src_arms['recovery_probe']['conditional_perturbed_success']*100:.1f}%")
    print(f"clean raw success = {out['clean_raw_success']*100:.1f}%")
    print("PSTART_CONTROL_DONE")


if __name__ == "__main__":
    main()
