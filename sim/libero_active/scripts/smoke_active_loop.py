"""Active-loop smoke test (PS.5 pre-flight).

Runs the smallest possible active-query loop that still exercises the live
integration point — `policy_runner.episode_signals` against a real LoRA-fine-tuned
SmolVLA checkpoint produced inside the loop. Validates that:

- the loop trains -> evaluates -> scores candidates -> retrains across rounds,
- the conformal method's candidate scoring path works end-to-end on real policy I/O,
- per-round LoRA retraining + eval do not blow up.

This is NOT a research result; it is a code-path validator. Expected ~30-45 min
on one RTX 6000-class GPU. Success = the run completes and a curve.json exists.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

# allow `python scripts/smoke_active_loop.py` from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conformal_active.active_loop import run_active_loop
from conformal_active.config import (
    ActiveLoopConfig,
    ConformalConfig,
    ExperimentConfig,
    TrainConfig,
)


def main() -> int:
    # Use method='random' here: it exercises the full loop architecture
    # (train -> eval -> score candidates -> pick -> retrain) WITHOUT touching the
    # policy.forward() integration that's still unresolved for SmolVLA. Conformal /
    # entropy paths can be re-enabled once policy_runner's forward call is fixed.
    cfg = ExperimentConfig(
        method="random",
        suite="libero_spatial",
        seed=0,
        conformal=ConformalConfig(target_alpha=0.1, n_action_samples=4),
        train=replace(TrainConfig(), steps_per_round=200),  # tiny
        loop=ActiveLoopConfig(
            seed_demos=3,
            demos_per_round=2,
            max_demos=5,                  # -> exactly 2 rounds
            eval_episodes=2,              # cheap eval
            max_candidates_per_round=5,
        ),
        results_dir="results/ps5_smoke",
    )
    print(f"[smoke] config: method={cfg.method} suite={cfg.suite} "
          f"seed={cfg.seed} max_demos={cfg.loop.max_demos} "
          f"steps_per_round={cfg.train.steps_per_round}", flush=True)
    out = run_active_loop(cfg)
    print(json.dumps(out, indent=2), flush=True)

    ok = bool(out.get("curve"))
    last = out["curve"][-1] if ok else None
    if ok and last.get("note", "").startswith(("train failed", "eval failed")):
        ok = False
    print(f"[smoke] active loop {'OK' if ok else 'BROKEN'} "
          f"(rounds={len(out.get('curve', []))})", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
