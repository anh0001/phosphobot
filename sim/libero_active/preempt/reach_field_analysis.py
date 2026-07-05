"""K1 analysis for the reach-field pilot (a) — DVR statistics + verdict.

Pre-registered (vla-multiscopic-v2 IDEA_REPORT, reviewer-corrected gates):
  PRIMARY endpoint  = pregrasp where defined, else closest-approach-to-true.
  PRIMARY set       = displaced rollouts whose paired clean rollout (this run)
                      succeeded ("clean-conditioned").
  K1 CONTINUE gate  (pre-registered bar, 2026-06-16 — Codex-adjudicated):
                      median cos(e, -d) >= 0.50, AND task-cluster bootstrap
                      lower CI > 0.20, AND signed projection
                      median dot(e,-d)/||d||^2 >= 0.35, AND
                      closer-to-canonical fraction >= 0.65.
  DESCRIPTIVE ONLY (NOT gated): the magnitude-transfer regression slope/R^2 of
                      |e| ~ |d|. The pooled-task R^2 is confounded by per-task
                      canonical geometry (it tanks to ~0.05 even when every task
                      shows a clean canonical pull), so it must not gate the
                      verdict; the signed projection is the correct
                      magnitude-transfer measure. (Old rubric gated R^2>=0.25 and
                      so spuriously printed KILL on a result that passes every
                      pre-registered criterion — fixed here.)
Secondary views: all displaced rollouts; clean-conditioned failures only;
per-magnitude / per-direction / per-task breakdowns; all 3 endpoint defs.

Usage:  python preempt/reach_field_analysis.py [--in preempt/reach_field_results.jsonl]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RNG = np.random.default_rng(0)
N_BOOT = 10_000


def load(path: str) -> tuple[list[dict], dict]:
    recs = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    recs = [r for r in recs if "error" not in r]
    clean_succ = {(r["task_id"], r["seed"]): r["success"]
                  for r in recs if r["cond"] == "clean"}
    disp = [r for r in recs if r["cond"] != "clean"]
    return disp, clean_succ


def endpoint_xy(r: dict, kind: str) -> np.ndarray | None:
    if kind == "primary":
        ep = r["endpoints"]["pregrasp"] or r["endpoints"]["closest"]
    else:
        ep = r["endpoints"][kind]
    return None if ep is None else np.asarray(ep[:2], dtype=np.float64)


def vectors(recs: list[dict], kind: str) -> list[dict]:
    out = []
    for r in recs:
        E = endpoint_xy(r, kind)
        if E is None:
            continue
        T = np.asarray(r["true_xyz"][:2])
        C = np.asarray(r["canonical_xyz"][:2])
        d = np.asarray(r["d_real_xy"])
        if np.linalg.norm(d) < 0.01:  # perturb did not realize
            continue
        e = E - T
        cos = float(np.dot(e, -d) / (np.linalg.norm(e) * np.linalg.norm(d) + 1e-12))
        proj = float(np.dot(e, -d) / (np.dot(d, d) + 1e-12))  # signed projection onto -d
        out.append({
            "task_id": r["task_id"], "seed": r["seed"], "cond": r["cond"],
            "mag_cm": float(np.linalg.norm(d) * 100), "cos": cos, "proj": proj,
            "e_cm": float(np.linalg.norm(e) * 100),
            "d_cm": float(np.linalg.norm(d) * 100),
            "closer_canonical": bool(np.linalg.norm(E - C) < np.linalg.norm(E - T)),
            "small_e": bool(np.linalg.norm(e) < 0.01),
            "success": r["success"],
        })
    return out


def cluster_boot_median_cos(vecs: list[dict]) -> tuple[float, float, float]:
    by_task = defaultdict(list)
    for v in vecs:
        by_task[v["task_id"]].append(v["cos"])
    tasks = list(by_task)
    meds = []
    for _ in range(N_BOOT):
        sample = RNG.choice(len(tasks), size=len(tasks), replace=True)
        pool = [c for i in sample for c in by_task[tasks[i]]]
        meds.append(np.median(pool))
    return (float(np.median([v["cos"] for v in vecs])),
            float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5)))


def slope_r2(vecs: list[dict]) -> tuple[float, float, float]:
    x = np.array([v["d_cm"] for v in vecs])
    y = np.array([v["e_cm"] for v in vecs])
    if len(x) < 3 or np.std(x) < 1e-9:
        return float("nan"), float("nan"), float("nan")
    b, a = np.polyfit(x, y, 1)
    r2 = 1 - np.sum((y - (a + b * x)) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)
    return float(b), float(a), float(r2)


def summarize(name: str, vecs: list[dict]) -> dict | None:
    if not vecs:
        print(f"  {name}: EMPTY")
        return None
    med, lo, hi = cluster_boot_median_cos(vecs)
    b, a, r2 = slope_r2(vecs)
    med_proj = float(np.median([v["proj"] for v in vecs]))
    frac_canon = float(np.mean([v["closer_canonical"] for v in vecs]))
    frac_small = float(np.mean([v["small_e"] for v in vecs]))
    print(f"  {name}: n={len(vecs)}  median cos={med:.3f} [CI {lo:.3f},{hi:.3f}]  "
          f"proj={med_proj:.2f}  closer-canon={frac_canon:.0%}  "
          f"[descriptive: |e|~|d| slope={b:.2f} int {a:.1f}cm R2={r2:.2f}; |e|<1cm={frac_small:.0%}]")
    return {"n": len(vecs), "median_cos": med, "ci": [lo, hi], "median_proj": med_proj,
            "slope": b, "intercept_cm": a, "r2": r2, "frac_closer_canonical": frac_canon,
            "frac_small_e": frac_small}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="preempt/reach_field_results.jsonl")
    ap.add_argument("--out", default="preempt/reach_field_k1.json")
    args = ap.parse_args()

    disp, clean_succ = load(args.inp)
    n_pairs = len(clean_succ)
    n_clean_ok = sum(clean_succ.values())
    print(f"== reach-field K1 ==  displaced rollouts={len(disp)}  pairs={n_pairs} "
          f"(clean success {n_clean_ok})")

    out: dict = {"n_displaced": len(disp), "n_pairs": n_pairs,
                 "n_clean_success_pairs": n_clean_ok, "sets": {}}

    for kind in ("primary", "pregrasp", "closest", "final"):
        print(f"\n-- endpoint = {kind} --")
        all_v = vectors(disp, kind)
        cond_v = [v for v in all_v if clean_succ.get((v["task_id"], v["seed"]))]
        fail_v = [v for v in cond_v if not v["success"]]
        out["sets"][kind] = {
            "all": summarize("all", all_v),
            "clean_conditioned (PRIMARY SET)" if kind == "primary" else "clean_conditioned":
                summarize("clean-conditioned", cond_v),
            "clean_conditioned_failures": summarize("clean-cond failures", fail_v),
        }
        if kind == "primary" and cond_v:
            print("  by magnitude:")
            for m in sorted({round(v_["mag_cm"] * 0.4) / 0.4 for v_ in cond_v}):
                grp = [v_ for v_ in cond_v if abs(v_["mag_cm"] - m) < 1.0]
                if grp:
                    print(f"    ~{m:.1f}cm: n={len(grp)} median cos="
                          f"{np.median([g['cos'] for g in grp]):.3f} "
                          f"median |e|={np.median([g['e_cm'] for g in grp]):.1f}cm "
                          f"disp-succ={np.mean([g['success'] for g in grp]):.0%}")
            print("  by direction:")
            for dn in ("px", "nx", "py", "ny"):
                grp = [v_ for v_ in cond_v if v_["cond"].endswith(dn)]
                if grp:
                    print(f"    {dn}: n={len(grp)} median cos="
                          f"{np.median([g['cos'] for g in grp]):.3f}")

    # ---- K1 verdict on the pre-registered primary set ----
    prim = out["sets"]["primary"].get("clean_conditioned (PRIMARY SET)")
    print("\n==== K1 VERDICT (primary endpoint, clean-conditioned) ====")
    if prim is None:
        print("NO DATA — cannot decide K1")
        verdict = "NO_DATA"
    else:
        checks = {
            "median_cos >= 0.50": prim["median_cos"] >= 0.50,
            "boot lower CI > 0.20 (clearly above chance)": prim["ci"][0] > 0.20,
            "signed projection median dot(e,-d)/||d||^2 >= 0.35": prim["median_proj"] >= 0.35,
            "closer-to-canonical >= 0.65": prim["frac_closer_canonical"] >= 0.65,
        }
        for k, v in checks.items():
            print(f"  [{'PASS' if v else 'FAIL'}] {k}")
        print(f"  [descriptive, NOT gated] |e|~|d| slope={prim['slope']:.2f} "
              f"R2={prim['r2']:.2f} (pooled R2 confounded by task geometry — see header)")
        verdict = "CONTINUE" if all(checks.values()) else "KILL (review breakdowns before final call)"
        print(f"K1: {verdict}")
    out["k1_verdict"] = verdict
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwritten: {args.out}\nREACH_FIELD_K1_DONE")


if __name__ == "__main__":
    main()
