"""P2T M2-stage — powered G with between-seed variance.

Intervention arms: M2 (100% synthetic) seeds {0,1,2}, N1 (100% syn + 352 canonical)
seeds {0,1,2}. Primary degradation null: base + action-noise sigma=0.15, noise-seeds
{1,2,3}. Per-arm u_gap = median[u_fail] - median[u_succ] @50mm (pregrasp endpoint).

Two levels of inference:
1. Between-seed: mean +- std of per-seed u_gap for each arm; G_seed = mean u_gap(arm)
   - mean u_gap(null); a paired t-like read on 3v3 seeds (report mean, sd, and a
     seed-level bootstrap CI).
2. Pooled task-cluster: pool all seeds, wild-cluster bootstrap of G (as in M1) for CI.

Grounding claimed iff G>0 with CI excluding 0 AND |v|-dispersion guard holds.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p2t.stats import readout, u_gap, G_statistic, wild_cluster_boot  # noqa: E402

SIG = "0.15"
M2 = {0: "p2t/eval_r2_M2_pure.jsonl", 1: "p2t/eval_m2s_M2_s1.jsonl", 2: "p2t/eval_m2s_M2_s2.jsonl"}
N1 = {s: f"p2t/eval_m2s_N1_s{s}.jsonl" for s in (0, 1, 2)}
NULL = {ns: f"p2t/eval_null_s{SIG}_ns{ns}.jsonl" for ns in (1, 2, 3)}


def per_seed_ugap(paths: dict) -> tuple[list[float], list[dict]]:
    gaps, pooled = [], []
    for s, p in paths.items():
        if not Path(p).exists():
            print(f"  [missing] {p}")
            continue
        rows = readout(p, "pregrasp", mag=50)
        if rows:
            gaps.append(u_gap(rows)); pooled += rows
    return gaps, pooled


def summarize(name: str, paths: dict):
    gaps, pooled = per_seed_ugap(paths)
    if not gaps:
        print(f"{name}: NO DATA"); return None, []
    us = wild_cluster_boot(pooled, "u", lambda r: r["success"])
    print(f"{name:<14} u_gap/seed={[round(g,2) for g in gaps]}  "
          f"mean={np.mean(gaps):+.2f} sd={np.std(gaps):.2f}  "
          f"pooled u_succ={us[0]:.2f} n={len(pooled)}")
    return gaps, pooled


def main():
    print("=== per-seed u_gap @50mm (pregrasp) ===")
    m2_g, m2_pool = summarize("M2 (100%syn)", M2)
    n1_g, n1_pool = summarize("N1 (syn+canon)", N1)
    nl_g, nl_pool = summarize(f"null σ={SIG}", NULL)
    if not (m2_g and nl_g):
        print("\n[incomplete — waiting on runs]"); return

    print("\n=== G = u_gap(intervention) - u_gap(null) ===")
    # between-seed: mean diff + seed bootstrap
    def seed_G(ig, ng):
        rng = np.random.default_rng(0)
        boots = [np.mean(rng.choice(ig, len(ig))) - np.mean(rng.choice(ng, len(ng)))
                 for _ in range(10000)]
        return float(np.mean(ig) - np.mean(ng)), float(np.percentile(boots, 5)), float(np.percentile(boots, 95))
    for name, ig, pool in [("M2", m2_g, m2_pool), ("N1", n1_g, n1_pool)]:
        if not ig:
            continue
        sg = seed_G(ig, nl_g)
        pg = G_statistic(pool, nl_pool)
        print(f"{name}: between-seed G={sg[0]:+.2f} CI90[{sg[1]:+.2f},{sg[2]:+.2f}] "
              f"excl0={sg[1] > 0}")
        print(f"    pooled-cluster G={pg['G']:+.2f} CI90[{pg['ci90'][0]:+.2f},"
              f"{pg['ci90'][1]:+.2f}] excl0={pg['excludes_0']} "
              f"|v|guard={pg['dispersion_guard_ok']}")
    print("\nVERDICT: grounding beyond degradation IFF a G CI excludes 0 with |v| guard OK.")


if __name__ == "__main__":
    main()
