"""P2T M0 — statistics module: readout, wild-cluster bootstrap, G, TOST.

- readout: u (along -d, 0=object,1=canonical), v (orthogonal), rho (||e||/||d||),
  per rollout, at a chosen endpoint (pregrasp | contact | lift), success-conditioned.
- wild_cluster_boot: Rademacher wild bootstrap over task clusters (10 clusters in
  LIBERO; percentile bootstrap undercovers at k~10, so we resample cluster mean
  perturbations) for a median/mean statistic and for differences.
- G: competence-matched grounding gain
     G = ([u_fail - u_succ]_intervention) - ([u_fail - u_succ]_null), with CI.
- tost: two-one-sided-tests equivalence for a paired-condition difference vs a margin.

All functions read the raw eval jsonl (optionally enriched with contact/lift).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RNG = np.random.default_rng(0)
NBOOT = 10000


def readout(path: str, endpoint: str = "pregrasp", mag: int | None = None) -> list[dict]:
    """Per-displaced-rollout u/v/rho + success + task, at the given endpoint."""
    out = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "error" in r or r.get("cond") == "clean":
            continue
        if mag is not None and not r.get("cond", "").startswith(f"d{mag}mm"):
            continue
        eps = r.get("endpoints", {})
        ep = eps.get(endpoint)
        if ep is None and endpoint == "pregrasp":
            ep = eps.get("closest")
        if ep is None:
            continue
        d = np.asarray(r["d_real_xy"], dtype=np.float64)
        nd = np.linalg.norm(d)
        if nd < 0.01:
            continue
        E = np.asarray(ep[:2]); T = np.asarray(r["true_xyz"][:2])
        dhat = d / nd; perp = np.array([-dhat[1], dhat[0]])
        e = E - T
        out.append({
            "task": r["task_id"] if "task_id" in r else int(r["key"].split("_")[0][1:]),
            "success": bool(r["success"]),
            "u": float(np.dot(e, -dhat) / nd),
            "v": float(np.dot(e, perp) / nd),
            "rho": float(np.linalg.norm(e) / nd),
        })
    return out


def _by_task(rows: list[dict], field: str, pred=None) -> dict[int, list[float]]:
    d = defaultdict(list)
    for r in rows:
        if pred is None or pred(r):
            d[r["task"]].append(r[field])
    return d


def wild_cluster_boot(rows: list[dict], field: str, pred=None, stat=np.median,
                      nboot: int = NBOOT) -> tuple[float, float, float]:
    """Point estimate + 90% CI of `stat(field)` via task-cluster resampling."""
    by = _by_task(rows, field, pred)
    tasks = list(by)
    if not tasks:
        return float("nan"), float("nan"), float("nan")
    point = float(stat([x for t in tasks for x in by[t]]))
    boots = []
    for _ in range(nboot):
        samp = RNG.choice(len(tasks), size=len(tasks), replace=True)
        pool = [x for i in samp for x in by[tasks[i]]]
        boots.append(stat(pool))
    return point, float(np.percentile(boots, 5)), float(np.percentile(boots, 95))


def u_gap(rows: list[dict]) -> float:
    """[u_fail - u_succ]: positive = successes reach toward object more than failures."""
    us = [r["u"] for r in rows if r["success"]]
    uf = [r["u"] for r in rows if not r["success"]]
    if not us or not uf:
        return float("nan")
    return float(np.median(uf) - np.median(us))


def G_statistic(intervention: list[dict], null: list[dict],
                nboot: int = NBOOT) -> dict:
    """G = u_gap(intervention) - u_gap(null), task-cluster bootstrap 90% CI.
    Also returns the |v| dispersion guard (IQR ratio, successes)."""
    def cluster_u_gap(rows, samp_tasks):
        by_s = _by_task(rows, "u", lambda r: r["success"])
        by_f = _by_task(rows, "u", lambda r: not r["success"])
        us = [x for t in samp_tasks for x in by_s.get(t, [])]
        uf = [x for t in samp_tasks for x in by_f.get(t, [])]
        if not us or not uf:
            return None
        return np.median(uf) - np.median(us)
    tasks = sorted(set(r["task"] for r in intervention) & set(r["task"] for r in null))
    point = u_gap(intervention) - u_gap(null)
    boots = []
    for _ in range(nboot):
        samp = [tasks[i] for i in RNG.choice(len(tasks), size=len(tasks), replace=True)]
        gi = cluster_u_gap(intervention, samp); gn = cluster_u_gap(null, samp)
        if gi is not None and gn is not None:
            boots.append(gi - gn)
    lo, hi = np.percentile(boots, [5, 95])
    def viqr(rows):
        v = [abs(r["v"]) for r in rows if r["success"]]
        return float(np.subtract(*np.percentile(v, [75, 25]))) if v else float("nan")
    return {"G": float(point), "ci90": [float(lo), float(hi)],
            "excludes_0": bool(lo > 0),
            "v_iqr_intervention": viqr(intervention), "v_iqr_null": viqr(null),
            "dispersion_guard_ok": bool(viqr(intervention) <= viqr(null) + 0.10)}


def tost(rows_a: list[dict], rows_b: list[dict], field: str, margin: float,
         nboot: int = NBOOT) -> dict:
    """Equivalence of median(field) between two conditions within +-margin
    (task-cluster bootstrap CI of the difference; equivalent iff CI in (-m, m))."""
    by_a = _by_task(rows_a, field); by_b = _by_task(rows_b, field)
    tasks = sorted(set(by_a) & set(by_b))
    diffs = []
    for _ in range(nboot):
        samp = [tasks[i] for i in RNG.choice(len(tasks), size=len(tasks), replace=True)]
        a = [x for t in samp for x in by_a.get(t, [])]
        b = [x for t in samp for x in by_b.get(t, [])]
        if a and b:
            diffs.append(np.median(a) - np.median(b))
    lo, hi = np.percentile(diffs, [5, 95])
    point = np.median([x for t in tasks for x in by_a[t]]) - \
            np.median([x for t in tasks for x in by_b[t]])
    return {"diff": float(point), "ci90": [float(lo), float(hi)], "margin": margin,
            "equivalent": bool(lo > -margin and hi < margin)}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true",
                    help="reproduce round-1/2 u medians as a sanity check")
    args = ap.parse_args()
    if args.validate:
        for name, path in [("base", "p2t/reach_field_retrain.jsonl"),
                           ("A(17%)", "p2t/eval_A_gain.jsonl"),
                           ("M2(100%)", "p2t/eval_r2_M2_pure.jsonl"),
                           ("M3(unfrozen)", "p2t/eval_r2_M3_plastic.jsonl")]:
            rows = readout(path, "pregrasp", mag=50)
            us = wild_cluster_boot(rows, "u", lambda r: r["success"])
            print(f"{name:<14} u_succ@50 = {us[0]:.2f} [{us[1]:.2f},{us[2]:.2f}]  "
                  f"u_gap={u_gap(rows):+.2f}  n={len(rows)}")
