"""PS.4 — Offline-pipeline sanity gate.

The single make-or-break checkpoint of Phase S: prove the *collect -> train -> predict
-> succeed* chain works at all, before any active-query work or hardware.

Gate: a SmolVLA-450M fine-tuned (LoRA) on ~40 LIBERO demonstrations reaches >= 80%
success on one LIBERO suite over 20 evaluation rollouts. If it fails, STOP and fix the
data / training / inference path — no active-query cleverness can rescue a broken
collect->train->predict chain.

Usage:
    python offline_sanity.py --suite libero_spatial --n-demos 40 --steps 60000
    python offline_sanity.py --suite libero_spatial --smoke   # tiny run, pipeline check only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from conformal_active.config import TrainConfig
from conformal_active.evaluate import evaluate_checkpoint
from conformal_active.train import train_smolvla_lora

GATE_SUCCESS_PCT = 80.0
GATE_EVAL_EPISODES = 20


def run_offline_sanity(
    *,
    suite: str,
    n_demos: int,
    steps: int,
    results_dir: Path,
    seed: int,
    eval_episodes: int,
) -> dict:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # "Collect": use the first n_demos episodes of the LIBERO suite as the demo budget.
    # (In the active loop these indices are chosen by a query method instead.)
    episodes = list(range(n_demos))
    print(f"[PS.4] suite={suite} demos={n_demos} steps={steps} seed={seed}", flush=True)

    train_out = results_dir / "train"
    train_res = train_smolvla_lora(
        output_dir=train_out,
        episodes=episodes,
        suite=suite,
        cfg=TrainConfig(),
        steps=steps,
        seed=seed,
    )
    if not train_res.ok:
        return _verdict(suite, n_demos, float("nan"), t0, results_dir,
                        passed=False, note=f"training failed rc={train_res.returncode}")

    print(f"[PS.4] training done -> {train_res.checkpoint_dir}", flush=True)

    eval_res = evaluate_checkpoint(
        checkpoint_dir=train_res.checkpoint_dir,
        suite=suite,
        output_dir=results_dir / "eval",
        n_episodes=eval_episodes,
    )
    if not eval_res.ok:
        return _verdict(suite, n_demos, float("nan"), t0, results_dir,
                        passed=False, note=f"eval failed rc={eval_res.returncode}")

    passed = eval_res.pc_success >= GATE_SUCCESS_PCT
    return _verdict(suite, n_demos, eval_res.pc_success, t0, results_dir,
                    passed=passed, note="ok")


def _verdict(suite, n_demos, pc, t0, results_dir, *, passed, note) -> dict:
    verdict = {
        "phase": "PS.4_offline_sanity",
        "suite": suite,
        "n_demos": n_demos,
        "pc_success": pc,
        "gate_pct": GATE_SUCCESS_PCT,
        "passed": passed,
        "note": note,
        "wall_s": round(time.time() - t0, 1),
    }
    (Path(results_dir) / "verdict.json").write_text(json.dumps(verdict, indent=2))
    return verdict


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--n-demos", type=int, default=40)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-episodes", type=int, default=GATE_EVAL_EPISODES)
    ap.add_argument("--results-dir", default="results/ps4_offline_sanity")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run (200 steps, 4 demos, 2 eval eps) — pipeline check only")
    args = ap.parse_args()

    if args.smoke:
        args.steps, args.n_demos, args.eval_episodes = 200, 4, 2
        args.results_dir = "results/ps4_smoke"

    verdict = run_offline_sanity(
        suite=args.suite,
        n_demos=args.n_demos,
        steps=args.steps,
        results_dir=Path(args.results_dir),
        seed=args.seed,
        eval_episodes=args.eval_episodes,
    )
    print(json.dumps(verdict, indent=2), flush=True)

    if args.smoke:
        # Smoke run only checks the pipeline executed; success rate is not the gate.
        ok = not verdict["note"].startswith(("training failed", "eval failed"))
        print(f"[PS.4-smoke] pipeline {'OK' if ok else 'BROKEN'}", flush=True)
        return 0 if ok else 1

    print(f"[PS.4] GATE {'PASSED' if verdict['passed'] else 'FAILED'} "
          f"({verdict['pc_success']:.1f}% vs {GATE_SUCCESS_PCT:.0f}%)", flush=True)
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
