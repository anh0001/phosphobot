"""E1 oracle-handover harness: override (preempt-hold) vs replan after a stale chunk.

SmolVLA executes 50-step action chunks open-loop. After a mid-rollout disturbance
makes the committed chunk STALE, is it better to HOLD (override) than to keep
executing stale actions while a re-plan computes? This harness runs 5 variants per
(task_id, seed) on a single in-process LIBERO MuJoCo env and reports the conditional
perturbed success and a kill-criterion verdict.

Usage:
  env -u PYTHONPATH MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    python preempt/e1_oracle_handover.py --tasks 0,1 --seeds 1000,1001 --smoke

Full E1 (10 tasks x 5 seeds):
  env -u PYTHONPATH MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    python preempt/e1_oracle_handover.py \
      --tasks 0,1,2,3,4,5,6,7,8,9 --seeds 1000,1001,1002,1003,1004 \
      --out preempt/e1_full_results.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preempt.harness_lib import (  # noqa: E402
    VARIANTS,
    build_pipeline,
    make_single_env,
    run_variant,
)

MAX_STEPS_SPATIAL = 280  # libero_spatial _max_episode_steps


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E1 oracle-handover rollout harness")
    p.add_argument("--tasks", type=str, default="0,1",
                   help="comma-separated task_ids")
    p.add_argument("--seeds", type=str, default="1000,1001",
                   help="comma-separated seeds (each maps to an episode_index too)")
    p.add_argument("--delta", type=float, default=0.05, help="lateral nudge in metres")
    p.add_argument("--latency", type=int, default=8, help="replan latency L (steps)")
    p.add_argument("--tp-frac", type=float, default=0.35,
                   help="perturbation step fraction of max_steps (spec: 0.35)")
    p.add_argument("--out", type=str, default="preempt/e1_results.json")
    p.add_argument("--smoke", action="store_true",
                   help="smoke mode: also measure stale-action harm; small grid")
    p.add_argument("--no-stale-harm", action="store_true",
                   help="disable the (expensive) stale-action-harm secondary metric")
    p.add_argument("--no-full-replan", action="store_true",
                   help="skip the full_replan (n=1) arm — dominates runtime and collapses the chunked policy")
    return p.parse_args()


def _int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip() != ""]


def aggregate(records: list[dict]) -> dict:
    """Aggregate per-arm metrics over (task,seed) pairs.

    PRIMARY: conditional_perturbed_success = mean(success_arm) over pairs where
    clean succeeded.
    """
    # index by (task, seed) -> {variant: record}
    pairs: dict[tuple[int, int], dict] = {}
    for r in records:
        key = (r["task_id"], r["seed"])
        pairs.setdefault(key, {})[r["variant"]] = r

    clean_keys = [k for k, v in pairs.items()
                  if "clean" in v and v["clean"]["success"]]
    all_keys = list(pairs.keys())

    arms = {}
    for variant in VARIANTS:
        succ_all = [pairs[k][variant]["success"] for k in all_keys if variant in pairs[k]]
        succ_cond = [pairs[k][variant]["success"] for k in clean_keys if variant in pairs[k]]
        macro = [pairs[k][variant]["macro_calls"] for k in all_keys if variant in pairs[k]]
        harms = [pairs[k][variant].get("stale_harm") for k in clean_keys
                 if variant in pairs[k] and pairs[k][variant].get("stale_harm") is not None]
        # n perturbations that actually fired (mid-task), among clean-success pairs
        fired = [bool(pairs[k][variant].get("perturb", {}).get("perturbed"))
                 for k in clean_keys if variant in pairs[k]]
        arms[variant] = {
            "raw_perturbed_success": float(np.mean(succ_all)) if succ_all else None,
            "conditional_perturbed_success": float(np.mean(succ_cond)) if succ_cond else None,
            "n_pairs": len(all_keys),
            "n_clean_success": len(clean_keys),
            "avg_macro_calls": float(np.mean(macro)) if macro else None,
            "stale_action_harm": float(np.mean(harms)) if harms else None,
            "n_perturbations_fired_in_clean_pairs": int(sum(fired)),
        }
    return {"arms": arms, "n_clean_success": len(clean_keys), "n_pairs": len(all_keys)}


def kill_criterion(arms: dict) -> dict:
    """preempt_hold worth pursuing iff:
        cond(preempt_hold) - cond(replan_only) >= +10pp
        OR stale_harm reduction >= 25%, at equal macro calls.
    """
    ph = arms.get("preempt_hold", {})
    ro = arms.get("replan_only", {})
    ph_cond = ph.get("conditional_perturbed_success")
    ro_cond = ro.get("conditional_perturbed_success")
    cond_gain_pp = None
    if ph_cond is not None and ro_cond is not None:
        cond_gain_pp = (ph_cond - ro_cond) * 100.0

    ph_harm = ph.get("stale_action_harm")
    ro_harm = ro.get("stale_action_harm")
    harm_reduction_pct = None
    if ph_harm is not None and ro_harm is not None and ro_harm > 0:
        harm_reduction_pct = (ro_harm - ph_harm) / ro_harm * 100.0

    equal_macro = (ph.get("avg_macro_calls") == ro.get("avg_macro_calls"))

    pass_cond = cond_gain_pp is not None and cond_gain_pp >= 10.0
    pass_harm = (harm_reduction_pct is not None and harm_reduction_pct >= 25.0 and equal_macro)
    passes = bool(pass_cond or pass_harm)
    return {
        "cond_gain_pp": cond_gain_pp,
        "harm_reduction_pct": harm_reduction_pct,
        "equal_macro_calls": equal_macro,
        "pass_via_conditional_success": bool(pass_cond),
        "pass_via_harm_reduction": bool(pass_harm),
        "PASSES": passes,
        "threshold_cond_gain_pp": 10.0,
        "threshold_harm_reduction_pct": 25.0,
    }


def main() -> None:
    args = parse_args()
    tasks = _int_list(args.tasks)
    seeds = _int_list(args.seeds)
    perturb_step = int(args.tp_frac * MAX_STEPS_SPATIAL)
    measure_harm = (args.smoke or True) and not args.no_stale_harm

    print(f"[e1] tasks={tasks} seeds={seeds} t_p={perturb_step} L={args.latency} "
          f"delta={args.delta} measure_harm={measure_harm}")
    t0 = time.time()
    pipe = build_pipeline()
    print(f"[e1] pipeline built in {time.time()-t0:.1f}s")

    records: list[dict] = []
    for ti, task_id in enumerate(tasks):
        for si, seed in enumerate(seeds):
            episode_index = si  # one episode per seed slot (fixed init state)
            active_variants = [v for v in VARIANTS
                               if not (args.no_full_replan and v == "full_replan")]
            for variant in active_variants:
                # only measure stale-harm where it is defined (stale window present)
                vm_harm = measure_harm and variant in ("replan_only", "preempt_hold")
                env = make_single_env(task_id=task_id, episode_index=episode_index)
                r = run_variant(
                    env, pipe, seed=seed, variant=variant,
                    perturb_step=perturb_step, latency=args.latency, delta=args.delta,
                    measure_stale_harm=vm_harm,
                )
                env.close()
                rec = {
                    "task_id": task_id, "seed": seed, "episode_index": episode_index,
                    "variant": variant, "success": r["success"], "n_steps": r["n_steps"],
                    "macro_calls": r["macro_calls"],
                    "perturb": {"perturbed": r["perturb"].get("perturbed"),
                                "object": r["perturb"].get("object")},
                    "stale_harm": r.get("stale_harm"),
                }
                records.append(rec)
                print(f"[e1] task={task_id} seed={seed} {variant:13s} "
                      f"success={r['success']} steps={r['n_steps']:3d} "
                      f"macro={r['macro_calls']:3d} perturbed={r['perturb'].get('perturbed')} "
                      f"harm={r.get('stale_harm')}")

    agg = aggregate(records)
    kc = kill_criterion(agg["arms"])

    result = {
        "config": {
            "tasks": tasks, "seeds": seeds, "perturb_step": perturb_step,
            "tp_frac": args.tp_frac, "latency": args.latency, "delta": args.delta,
            "max_steps": MAX_STEPS_SPATIAL, "ckpt": "fixedbuf_random_N20/seed0/004000",
            "suite": "libero_spatial",
        },
        "aggregate": agg,
        "kill_criterion": kc,
        "records": records,
        "wall_time_s": round(time.time() - t0, 1),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print("\n==== AGGREGATE (per arm) ====")
    for v in VARIANTS:
        a = agg["arms"][v]
        print(f"  {v:13s} cond={_fmt(a['conditional_perturbed_success'])} "
              f"raw={_fmt(a['raw_perturbed_success'])} "
              f"macro={a['avg_macro_calls']} harm={a['stale_action_harm']}")
    print(f"\n  n_clean_success={agg['n_clean_success']} / n_pairs={agg['n_pairs']}")
    print("\n==== KILL CRITERION ====")
    print(f"  cond_gain (preempt_hold - replan_only) = {kc['cond_gain_pp']} pp "
          f"(threshold >= +10pp)")
    print(f"  harm_reduction = {kc['harm_reduction_pct']} % "
          f"(threshold >= 25%, equal_macro={kc['equal_macro_calls']})")
    print(f"  PASSES = {kc['PASSES']}")
    print(f"\n[e1] wrote {out_path}")


def _fmt(x) -> str:
    return "  n/a" if x is None else f"{x*100:5.1f}%"


if __name__ == "__main__":
    main()
