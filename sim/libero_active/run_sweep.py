"""PS.6 — Main-results sweep.

Runs the active-query loop across the (method x suite x seed) grid and aggregates the
sample-efficiency curves into one table — the paper's main result.

Usage:
    python run_sweep.py --methods random,conformal --suites libero_spatial --seeds 0,1,2
    python run_sweep.py --full          # all 5 methods x 2 suites x 3 seeds
    python run_sweep.py --dry-run       # print the grid without running
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

from conformal_active.config import LIBERO_SUITES, ExperimentConfig
from conformal_active.active_loop import run_active_loop

ALL_METHODS = ["random", "entropy", "knn", "human_gated", "conformal"]


def build_grid(methods, suites, seeds) -> list[ExperimentConfig]:
    return [
        ExperimentConfig(method=m, suite=s, seed=seed)
        for m, s, seed in itertools.product(methods, suites, seeds)
    ]


def run_sweep(grid: list[ExperimentConfig], results_dir: Path) -> dict:
    results_dir.mkdir(parents=True, exist_ok=True)
    all_runs: list[dict] = []
    for i, cfg in enumerate(grid, 1):
        print(f"[sweep] {i}/{len(grid)} :: {cfg.run_id}", flush=True)
        t0 = time.time()
        try:
            out = run_active_loop(cfg)
            out["wall_s"] = round(time.time() - t0, 1)
        except Exception as e:  # noqa: BLE001 — keep the sweep going; record the failure
            out = {"run_id": cfg.run_id, "method": cfg.method, "suite": cfg.suite,
                   "seed": cfg.seed, "curve": [], "error": repr(e)}
        all_runs.append(out)
        (results_dir / "sweep.json").write_text(json.dumps(all_runs, indent=2))
    return {"n_runs": len(all_runs), "runs": all_runs}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="random,conformal")
    ap.add_argument("--suites", default="libero_spatial")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--results-dir", default="results/ps6_sweep")
    ap.add_argument("--full", action="store_true",
                    help="all 5 methods x 2 suites x 3 seeds (the paper grid)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.full:
        methods, suites, seeds = ALL_METHODS, list(LIBERO_SUITES), [0, 1, 2]
    else:
        methods = [m.strip() for m in args.methods.split(",")]
        suites = [s.strip() for s in args.suites.split(",")]
        seeds = [int(s) for s in args.seeds.split(",")]

    grid = build_grid(methods, suites, seeds)
    print(f"[sweep] grid: {len(grid)} runs "
          f"({len(methods)} methods x {len(suites)} suites x {len(seeds)} seeds)", flush=True)
    if args.dry_run:
        for cfg in grid:
            print("  ", cfg.run_id)
        return 0

    summary = run_sweep(grid, Path(args.results_dir))
    print(f"[sweep] done: {summary['n_runs']} runs -> {args.results_dir}/sweep.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
