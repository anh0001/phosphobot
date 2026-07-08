"""P2T step 5 — kill-gate adjudication across conditions A/B/C/D/E.

Reads p2t/eval_<COND>.jsonl (reach-field @50 mm per fine-tuned ckpt) plus the
retrained-base reach field (50 mm subset) as reference. Reports per condition:
  clean success | displaced success @50 mm | median cos(e,-d) | median proj
and the pre-registered contrasts (displaced success):
  PASS if A > C and A > D, margin >= +5 pp over D, outside the task-cluster
  bootstrap 90% CI; secondary: clean(A) >= clean(B) - 5 pp.

Usage: .venv/bin/python p2t/analyze_conditions.py [--dir p2t]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

CONDS = ["A_gain", "B_noaug", "C_uniform", "D_failure", "E_sham"]
RNG = np.random.default_rng(0)
N_BOOT = 10_000


def load(path: Path) -> dict:
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    recs = [r for r in recs if "error" not in r]
    clean = {(r["task_id"], r["seed"]): r["success"] for r in recs if r["cond"] == "clean"}
    disp = []
    for r in recs:
        if r["cond"] == "clean":
            continue
        ep = r["endpoints"]["pregrasp"] or r["endpoints"]["closest"]
        d = np.asarray(r["d_real_xy"], dtype=np.float64)
        if ep is None or np.linalg.norm(d) < 0.01:
            continue
        e = np.asarray(ep[:2]) - np.asarray(r["true_xyz"][:2])
        disp.append({
            "task_id": r["task_id"], "seed": r["seed"], "success": bool(r["success"]),
            "cos": float(np.dot(e, -d) / (np.linalg.norm(e) * np.linalg.norm(d) + 1e-12)),
            "proj": float(np.dot(e, -d) / (np.dot(d, d) + 1e-12)),
        })
    return {"clean": clean, "disp": disp}


def summary(d: dict) -> dict:
    disp = d["disp"]
    return {
        "clean_succ": float(np.mean(list(d["clean"].values()))) if d["clean"] else float("nan"),
        "disp_succ": float(np.mean([r["success"] for r in disp])) if disp else float("nan"),
        "median_cos": float(np.median([r["cos"] for r in disp])) if disp else float("nan"),
        "median_proj": float(np.median([r["proj"] for r in disp])) if disp else float("nan"),
        "n_disp": len(disp),
    }


def boot_diff(a: list[dict], b: list[dict]) -> tuple[float, float, float]:
    """Task-cluster bootstrap of disp-success difference (a - b); 90% CI."""
    by_task_a, by_task_b = defaultdict(list), defaultdict(list)
    for r in a:
        by_task_a[r["task_id"]].append(r["success"])
    for r in b:
        by_task_b[r["task_id"]].append(r["success"])
    tasks = sorted(set(by_task_a) & set(by_task_b))
    diffs = []
    for _ in range(N_BOOT):
        sample = RNG.choice(len(tasks), size=len(tasks), replace=True)
        va = [x for i in sample for x in by_task_a[tasks[i]]]
        vb = [x for i in sample for x in by_task_b[tasks[i]]]
        diffs.append(np.mean(va) - np.mean(vb))
    point = float(np.mean([r["success"] for r in a]) - np.mean([r["success"] for r in b]))
    return point, float(np.percentile(diffs, 5)), float(np.percentile(diffs, 95))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="p2t")
    args = ap.parse_args()
    base = Path(args.dir)

    data = {}
    for c in CONDS:
        p = base / f"eval_{c}.jsonl"
        if p.exists():
            data[c] = load(p)
    ref_p = base / "reach_field_retrain.jsonl"
    if ref_p.exists():
        ref = load(ref_p)
        ref["disp"] = [r for r in ref["disp"]]  # full grid; report as reference only
        data["base_retrain(all-mags)"] = ref

    print(f"{'condition':<24}{'clean':>7}{'disp@50':>9}{'cos':>7}{'proj':>7}{'n':>6}")
    for c, d in data.items():
        s = summary(d)
        print(f"{c:<24}{s['clean_succ']:>7.0%}{s['disp_succ']:>9.0%}"
              f"{s['median_cos']:>7.2f}{s['median_proj']:>7.2f}{s['n_disp']:>6}")

    if all(c in data for c in CONDS):
        print("\n-- pre-registered contrasts (displaced success, A minus X) --")
        gates = {}
        for other in ("B_noaug", "C_uniform", "D_failure", "E_sham"):
            pt, lo, hi = boot_diff(data["A_gain"]["disp"], data[other]["disp"])
            gates[other] = (pt, lo, hi)
            print(f"  A - {other:<10}: {pt:+.1%}  [90% CI {lo:+.1%}, {hi:+.1%}]")
        a_gt_c = gates["C_uniform"][0] > 0
        a_gt_d = gates["D_failure"][0] >= 0.05 and gates["D_failure"][1] > 0
        clean_ok = (summary(data["A_gain"])["clean_succ"]
                    >= summary(data["B_noaug"])["clean_succ"] - 0.05)
        print("\n==== KILL-GATE VERDICT ====")
        print(f"  [{'PASS' if a_gt_c else 'FAIL'}] A > C (placement matters vs uniform)")
        print(f"  [{'PASS' if a_gt_d else 'FAIL'}] A >= D + 5pp with CI>0 (causal gain beats failure signal)")
        print(f"  [{'PASS' if clean_ok else 'FAIL'}] clean(A) >= clean(B) - 5pp (no in-dist regression)")
        if a_gt_c and a_gt_d and clean_ok:
            print("VERDICT: PASS — gain-targeted acquisition is a mechanism, not a reframing")
        elif not a_gt_d and a_gt_c:
            print("VERDICT: PARTIAL — placement matters but the causal gain signal does "
                  "not beat the generic failure signal (report as negative for the core claim)")
        else:
            print("VERDICT: FAIL — see contrasts above")
    print("ANALYZE_DONE")


if __name__ == "__main__":
    main()
