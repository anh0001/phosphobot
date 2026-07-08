"""P2T step 2 — acquisition maps from a reach-field run.

Computes per-cell (task, dir, mag) statistics on the clean-conditioned set
(conventions follow preempt/reach_field_analysis.py):
  deficit_c  = clip(median proj, 0, 1)   # dot(e,-d)/||d||^2; 1 = fully anchored
  failure_c  = 1 - displaced success rate

and turns them into synthesis budgets for the pre-registered conditions:
  A gain-targeted    w ∝ deficit_c
  C uniform          w = 1
  D failure-targeted w ∝ failure_c        (generic/IntervenGen-style signal)
  E sham             w ∝ deficit at the BOTTOM (lowest-deficit cells get the
                       budget; specificity control)

Cells with n < --min-n fall back to the (dir, mag)-pooled deficit/failure.
Prints the pre-registered A↔D Spearman correlation (>0.9 => contrast
underpowered; report and re-specify D openly).

Usage:
  .venv/bin/python p2t/acquisition.py --in preempt/e1r_reach_field_strong100k.jsonl \
      --n-syn 128 --out p2t/acquisition_maps.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

DIRS = ("px", "nx", "py", "ny")


def load_cells(path: str, min_n: int) -> dict:
    recs = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    recs = [r for r in recs if "error" not in r]
    clean_succ = {(r["task_id"], r["seed"]): r["success"]
                  for r in recs if r["cond"] == "clean"}
    cells: dict[tuple, dict] = defaultdict(lambda: {"proj": [], "succ": []})
    for r in recs:
        if r["cond"] == "clean" or not clean_succ.get((r["task_id"], r["seed"])):
            continue
        ep = r["endpoints"]["pregrasp"] or r["endpoints"]["closest"]
        if ep is None:
            continue
        d = np.asarray(r["d_real_xy"], dtype=np.float64)
        if np.linalg.norm(d) < 0.01:
            continue
        E = np.asarray(ep[:2], dtype=np.float64)
        T = np.asarray(r["true_xyz"][:2], dtype=np.float64)
        e = E - T
        proj = float(np.dot(e, -d) / (np.dot(d, d) + 1e-12))
        cond = r["cond"]                      # e.g. d50mm_px
        mag_mm = int(cond.split("mm_")[0][1:])
        dirname = cond.split("_")[-1]
        key = (r["task_id"], dirname, mag_mm)
        cells[key]["proj"].append(proj)
        cells[key]["succ"].append(bool(r["success"]))

    pooled: dict[tuple, dict] = defaultdict(lambda: {"proj": [], "succ": []})
    for (t, dn, m), v in cells.items():
        pooled[(dn, m)]["proj"] += v["proj"]
        pooled[(dn, m)]["succ"] += v["succ"]

    out = {}
    for key, v in cells.items():
        t, dn, m = key
        if len(v["proj"]) >= min_n:
            proj_med, fail = float(np.median(v["proj"])), 1.0 - float(np.mean(v["succ"]))
            src = "cell"
        else:
            pv = pooled[(dn, m)]
            proj_med, fail = float(np.median(pv["proj"])), 1.0 - float(np.mean(pv["succ"]))
            src = "pooled"
        out[key] = {"n": len(v["proj"]), "deficit": float(np.clip(proj_med, 0.0, 1.0)),
                    "failure": fail, "src": src}
    return out


def allocate(weights: dict[tuple, float], n_syn: int) -> dict[tuple, int]:
    total = sum(weights.values())
    if total <= 0:
        weights = {k: 1.0 for k in weights}
        total = float(len(weights))
    raw = {k: n_syn * w / total for k, w in weights.items()}
    alloc = {k: int(np.floor(v)) for k, v in raw.items()}
    rem = n_syn - sum(alloc.values())
    for k in sorted(raw, key=lambda k: raw[k] - alloc[k], reverse=True)[:rem]:
        alloc[k] += 1
    return {k: v for k, v in alloc.items() if v > 0}


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx ** 2).sum() * (ry ** 2).sum() + 1e-12))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--n-syn", type=int, default=128)
    ap.add_argument("--min-n", type=int, default=3)
    ap.add_argument("--out", default="p2t/acquisition_maps.json")
    ap.add_argument("--auto-sharpen", action="store_true",
                    help="pre-registered rule (2026-07-06): if std(deficit) < 0.10 "
                         "(near-uniform field, as on the old e1r ckpt where A "
                         "degenerated to uniform), weight A by deficit**2 instead "
                         "of deficit. Applied openly and logged in the output.")
    args = ap.parse_args()

    cells = load_cells(args.inp, args.min_n)
    keys = sorted(cells)
    deficit = np.array([cells[k]["deficit"] for k in keys])
    failure = np.array([cells[k]["failure"] for k in keys])
    sharpen = bool(args.auto_sharpen and deficit.std() < 0.10)
    a_weight = deficit ** 2 if sharpen else deficit
    if args.auto_sharpen:
        print(f"deficit std={deficit.std():.3f} -> A weight = deficit"
              f"{'**2 (sharpened)' if sharpen else ' (linear)'}")

    rho = spearman(deficit, failure)
    print(f"cells={len(keys)}  deficit: med={np.median(deficit):.2f} "
          f"range=[{deficit.min():.2f},{deficit.max():.2f}]  "
          f"failure: med={np.median(failure):.2f} "
          f"range=[{failure.min():.2f},{failure.max():.2f}]")
    print(f"A<->D Spearman rho = {rho:.3f}  "
          f"({'UNDERPOWERED CONTRAST — re-specify D' if rho > 0.9 else 'contrast OK'})")

    inv = deficit.max() - deficit  # sham: budget where gain is LEAST missing
    maps = {
        "A_gain":    allocate({k: float(w) for k, w in zip(keys, a_weight)}, args.n_syn),
        "C_uniform": allocate({k: 1.0 for k in keys}, args.n_syn),
        "D_failure": allocate({k: cells[k]["failure"] for k in keys}, args.n_syn),
        "E_sham":    allocate({k: float(v) for k, v in zip(keys, inv)}, args.n_syn),
    }
    for name, alloc in maps.items():
        top = sorted(alloc.items(), key=lambda kv: -kv[1])[:5]
        print(f"  {name}: {sum(alloc.values())} eps over {len(alloc)} cells; "
              f"top: {[(f't{k[0]}_{k[1]}_{k[2]}mm', v) for k, v in top]}")

    out = {
        "source": args.inp, "n_syn": args.n_syn, "spearman_A_D": rho,
        "deficit_std": float(deficit.std()), "A_sharpened": sharpen,
        "cells": {f"t{k[0]}_{k[1]}_{k[2]}mm": cells[k] for k in keys},
        "alloc": {name: {f"t{k[0]}_{k[1]}_{k[2]}mm": v for k, v in alloc.items()}
                  for name, alloc in maps.items()},
    }
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
