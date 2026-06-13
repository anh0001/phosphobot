"""K2 analysis for the oracle-warp pilot (b) — causal sufficiency of re-anchoring.

Per-arm endpoint statistics against the canonical-anchoring model E ≈ C + w:
  nowarp (v2 grid d50mm records)  E ≈ C        |E-T| ≈ |d|
  warp_full                       E ≈ C+d = T  RESTORATION (K2 primary)
  warp_image / warp_proprio       channel attribution
  warp_random (w = R90 d)         E ≈ C+w      specificity (no restoration)
  warp_clean (no displ, w)        E ≈ C+w      breaks healthy episodes by +w
  sham_zero                       == grid nowarp record (machinery no-op)
  midroll                         post-replan endpoint still ≈ C (mid-rollout tau)

K2 PRIMARY metric (reviewer-corrected): endpoint-error reduction
  R_e = 1 - median|E_warpfull - T| / median|E_nowarp - T|   (>= 0.5 green)
Secondary: paired conditioned-success restoration; specificity margins.

Usage: python preempt/oracle_warp_analysis.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

GRID = "preempt/reach_field_results_v2.jsonl"
WARP = "preempt/oracle_warp_results.jsonl"
OUT = "preempt/oracle_warp_k2.json"
DIRS = {"px": (1.0, 0.0), "nx": (-1.0, 0.0), "py": (0.0, 1.0), "ny": (0.0, -1.0)}


def ep_xy(endpoints: dict) -> np.ndarray | None:
    e = endpoints.get("pregrasp") or endpoints.get("closest")
    return None if e is None else np.asarray(e[:2], dtype=np.float64)


def main() -> None:
    grid = [json.loads(l) for l in Path(GRID).read_text().splitlines()]
    clean = {(r["task_id"], r["seed"]): r for r in grid if r["cond"] == "clean"}
    pairs = sorted(k for k, v in clean.items() if v["success"])

    # nowarp baseline at 5cm from the v2 grid
    nowarp = {}
    for r in grid:
        k = (r["task_id"], r["seed"])
        if k in dict.fromkeys(pairs) and r["cond"].startswith("d50mm_"):
            nowarp[(k, r["cond"].split("_")[1])] = r

    warp = [json.loads(l) for l in Path(WARP).read_text().splitlines()]
    warmups = [r for r in warp if r["arm"] == "warmup_clean"]
    n_wu_match = sum(bool(r.get("grid_match")) for r in warmups)
    print(f"pairs={len(pairs)}  warmups={len(warmups)} (grid_match {n_wu_match})")

    by_arm: dict[str, list[dict]] = defaultdict(list)
    for r in warp:
        if r["arm"] != "warmup_clean":
            by_arm[r["arm"]].append(r)

    def stats(recs, label, pred_of=None, use_post_replan=False):
        d_T, d_C, d_pred, succ, taus = [], [], [], [], []
        for r in recs:
            eps = r["endpoints_post_replan"] if use_post_replan else r["endpoints"]
            E = ep_xy(eps)
            if E is None:
                continue
            C = np.asarray(r["canonical_xyz"][:2])
            T = np.asarray(r["true_xyz"][:2])
            w = np.asarray(r.get("w_xyz", [0.0, 0.0, 0.0])[:2])
            k = (r["task_id"], r["seed"])
            d_T.append(np.linalg.norm(E - T) * 100)
            d_C.append(np.linalg.norm(E - C) * 100)
            pred = pred_of(C, T, w) if pred_of else None
            if pred is not None:
                d_pred.append(np.linalg.norm(E - pred) * 100)
            succ.append(bool(r["success"]))
            d = T - C
            if np.linalg.norm(d) > 0.01 and k in clean:
                Ec = ep_xy(clean[k]["endpoints"])
                if Ec is not None:
                    taus.append(float(np.dot(E - Ec, d / np.linalg.norm(d))) / np.linalg.norm(d))
        out = {
            "n": len(d_T),
            "med_d_true_cm": float(np.median(d_T)) if d_T else None,
            "med_d_canon_cm": float(np.median(d_C)) if d_C else None,
            "med_d_pred_cm": float(np.median(d_pred)) if d_pred else None,
            "success": float(np.mean(succ)) if succ else None,
            "med_tau": float(np.median(taus)) if taus else None,
        }
        print(f"  {label:14s} n={out['n']:3d}  |E-T|={out['med_d_true_cm']:5.1f}  "
              f"|E-C|={out['med_d_canon_cm']:5.1f}  "
              f"|E-pred|={out['med_d_pred_cm'] if out['med_d_pred_cm'] is not None else float('nan'):5.1f}  "
              f"succ={out['success']:.0%}  tau={out['med_tau'] if out['med_tau'] is not None else float('nan'):+.2f}")
        return out

    print("\n== arms (model prediction E = C + w) ==")
    res = {}
    res["nowarp_grid"] = stats(list(nowarp.values()), "nowarp(grid)", lambda C, T, w: C)
    for arm, pred in (("sham_zero", lambda C, T, w: C),
                      ("warp_full", lambda C, T, w: T),
                      ("warp_image", lambda C, T, w: T),
                      ("warp_proprio", lambda C, T, w: T),
                      ("warp_random", lambda C, T, w: C + w),
                      ("warp_clean", lambda C, T, w: C + w)):
        res[arm] = stats(by_arm[arm], arm, pred)
    res["midroll_post"] = stats(by_arm["midroll"], "midroll(post50)", lambda C, T, w: C,
                                use_post_replan=True)

    # sham no-op check vs grid records
    mism = 0
    for r in by_arm["sham_zero"]:
        g = nowarp.get(((r["task_id"], r["seed"]), "px"))
        if g and (g["success"] != r["success"] or g["n_steps"] != r["n_steps"]):
            mism += 1
    print(f"\nsham_zero vs grid nowarp(px): mismatches {mism}/{len(by_arm['sham_zero'])}")

    # ==== K2 verdict ====
    print("\n==== K2 (primary = endpoint-error reduction, warp_full vs nowarp) ====")
    base = res["nowarp_grid"]["med_d_true_cm"]
    full = res["warp_full"]["med_d_true_cm"]
    R_e = 1 - full / base if base else float("nan")
    spec_rand = res["warp_random"]["med_d_true_cm"]
    checks = {
        f"endpoint-error reduction R_e >= 0.5 (got {R_e:.2f}: {base:.1f} -> {full:.1f}cm)": R_e >= 0.5,
        f"specificity: warp_random does NOT restore (|E-T| {spec_rand:.1f} vs nowarp {base:.1f})":
            spec_rand >= 0.7 * base,
        f"sham no-op (mismatches {mism})": mism == 0,
        f"success restoration (warp_full {res['warp_full']['success']:.0%} vs nowarp "
        f"{res['nowarp_grid']['success']:.0%}, clean=100% by conditioning)":
            res["warp_full"]["success"] > res["nowarp_grid"]["success"],
    }
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    verdict = ("GREEN" if all(checks.values()) and R_e >= 0.5 else
               "YELLOW" if R_e >= 0.3 else "RED")
    print(f"K2: {verdict}")
    res["k2"] = {"R_e": R_e, "checks": {k: bool(v) for k, v in checks.items()},
                 "verdict": verdict}
    Path(OUT).write_text(json.dumps(res, indent=2))
    print(f"written: {OUT}\nORACLE_WARP_K2_DONE")


if __name__ == "__main__":
    main()
