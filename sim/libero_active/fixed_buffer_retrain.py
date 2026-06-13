"""Fixed-buffer LoRA-retrain variance study (paper experiment E1 — "Unreliable by Default").

Freeze a *selected demo buffer* (e.g. the dispersion_quota N=20 buffer) and retrain
LoRA k times with different TRAIN seeds, evaluating each on the SAME paired eval init
states (eval seed fixed = 1000). This isolates optimizer/train-seed variance from
acquisition variance: if the same buffer swings widely across retrains, then single-run
active-demo-selection "wins" are not reliable evidence of sample efficiency.

Outputs per-seed overall + per-task success and the across-seed mean/std/range, so the
headline statistic can be: "retrain spread (std) vs acquisition-method delta vs random".

Usage:
    python fixed_buffer_retrain.py --label quota_N20 \
        --buffer 1310,1315,1327,1363,1388,1394,1405,1406,1409,1434,1437,1456,1518,1521,1523,1575,1579,1587,1631,1673 \
        --seeds 0,1,2,3,4 --suite libero_spatial --steps 4000 --eval-episodes 20
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from conformal_active.config import TrainConfig
from conformal_active.evaluate import evaluate_checkpoint
from conformal_active.train import train_smolvla_lora

EVAL_SEED = 1000  # fixed across all retrains -> paired eval init states


def per_task_success(eval_dir: Path) -> dict[int, float]:
    """Parse per-task success-rate (percent) from LeRobot eval_info.json."""
    info_path = eval_dir / "eval_info.json"
    if not info_path.exists():
        cands = list(eval_dir.rglob("eval_info.json"))
        if not cands:
            return {}
        info_path = cands[0]
    info = json.loads(info_path.read_text())
    out: dict[int, float] = {}
    for item in info.get("per_task", []):
        s = item.get("metrics", {}).get("successes", [])
        if s:
            out[int(item["task_id"])] = round(100.0 * sum(bool(x) for x in s) / len(s), 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--buffer", required=True, help="comma-separated frozen episode indices")
    ap.add_argument("--label", required=True, help="e.g. quota_N20 / random_N20")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--seeds", default="0,1,2,3,4", help="comma-separated TRAIN seeds")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--eval-episodes", type=int, default=20, help="per task; x10 tasks = total rollouts")
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args()

    buffer = [int(x) for x in args.buffer.split(",") if x.strip() != ""]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip() != ""]
    rdir = Path(args.results_dir or f"results/fixedbuf_{args.label}")
    rdir.mkdir(parents=True, exist_ok=True)

    summary: dict = {
        "label": args.label, "suite": args.suite, "buffer": buffer, "n_demos": len(buffer),
        "steps": args.steps, "eval_seed": EVAL_SEED, "train_seeds": seeds, "runs": [],
    }
    print(f"[E1] fixed-buffer retrain: label={args.label} n_demos={len(buffer)} "
          f"seeds={seeds} steps={args.steps}", flush=True)

    def _flush():
        (rdir / "summary.json").write_text(json.dumps(summary, indent=2))

    for seed in seeds:
        rundir = rdir / f"seed{seed}"
        t0 = time.time()
        tr = train_smolvla_lora(
            output_dir=rundir / "train", episodes=buffer, suite=args.suite,
            cfg=TrainConfig(), steps=args.steps, seed=seed,
        )
        if not tr.ok:
            summary["runs"].append({"seed": seed, "ok": False, "note": f"train rc={tr.returncode}"})
            _flush()
            print(f"[E1] {args.label} seed{seed}: TRAIN FAILED rc={tr.returncode}", flush=True)
            continue
        ev = evaluate_checkpoint(
            checkpoint_dir=tr.checkpoint_dir, suite=args.suite,
            output_dir=rundir / "eval", n_episodes=args.eval_episodes, seed=EVAL_SEED,
        )
        if not ev.ok:
            summary["runs"].append({"seed": seed, "ok": False, "note": f"eval rc={ev.returncode}"})
            _flush()
            print(f"[E1] {args.label} seed{seed}: EVAL FAILED rc={ev.returncode}", flush=True)
            continue
        pt = per_task_success(rundir / "eval")
        summary["runs"].append({
            "seed": seed, "ok": True, "pc_success": ev.pc_success,
            "per_task": pt, "wall_s": round(time.time() - t0, 1),
        })
        _flush()
        print(f"[E1] {args.label} seed{seed}: overall={ev.pc_success:.1f}%  "
              f"({round(time.time()-t0)}s)", flush=True)

    oks = [r for r in summary["runs"] if r.get("ok")]
    if oks:
        vals = [r["pc_success"] for r in oks]
        summary["overall_mean"] = round(statistics.mean(vals), 2)
        summary["overall_std"] = round(statistics.pstdev(vals), 2) if len(vals) > 1 else 0.0
        summary["overall_range"] = [min(vals), max(vals)]
        # per-task std across retrains (the interference/variance signal)
        tasks = sorted({t for r in oks for t in r.get("per_task", {})})
        summary["per_task_std"] = {
            str(t): round(statistics.pstdev([r["per_task"][t] for r in oks if t in r["per_task"]]), 1)
            for t in tasks if sum(t in r["per_task"] for r in oks) > 1
        }
    _flush()
    print(f"[E1] {args.label} DONE: mean={summary.get('overall_mean')} "
          f"std={summary.get('overall_std')} range={summary.get('overall_range')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
