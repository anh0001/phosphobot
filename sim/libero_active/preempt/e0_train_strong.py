"""E0 — train the strong checkpoint: SmolVLA LoRA on ALL libero_spatial demos.

Full-suite multi-task fine-tune (432 episodes, 10 tasks) with intermediate
checkpoints every 5k steps (emergence axis: bias vs training progress).
Third-party hub checkpoints were screened and rejected (0% under our eval
conventions; no reported numbers — see PILOT_LOG.md).

RECIPE NOTE (2026-06-14): the first E0 run used TrainConfig's default
optimizer_lr=1e-3 / scheduler_decay_lr=1e-4 — which is 10x SmolVLA's own preset
(1e-4 / 2.5e-6). It plateaued at ~36-40% clean (weak 20-demo ckpt was 30%; 20k=40%,
40k=36% — no climb), failing the >=65% gate. Diagnosis (confirmed by Codex xhigh):
LR 10x too high on a tiny r=64 adapter -> underfit plateau. Defaults below are now
the CORRECTED recipe: lr=1e-4, decay_lr=2.5e-6 (warmup=1000 / decay_steps=30000 fall
through to the SmolVLA preset unchanged). --eval_freq=0 disables lerobot's in-training
eval, whose env-spawn at step 20k OOM-killed (SIGKILL) the first run.

Usage:
  env -u PYTHONPATH MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    python preempt/e0_train_strong.py [--steps 40000] [--seed 0] \
      [--lr 1e-4] [--decay-lr 2.5e-6] [--out results/e0_strong_lr1e4]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conformal_active.config import TrainConfig  # noqa: E402
from conformal_active.train import train_smolvla_lora  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-freq", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="SmolVLA preset LR (corrected; the 1e-3 default plateaued)")
    ap.add_argument("--decay-lr", type=float, default=2.5e-6,
                    help="SmolVLA preset cosine floor (warmup/decay_steps fall through)")
    ap.add_argument("--out", default="results/e0_strong_lr1e4")
    args = ap.parse_args()

    suite_map = json.load(open("results/suite_episode_map.json"))["libero_spatial"]
    episodes = sorted(int(e) for eps in suite_map.values() for e in eps)
    cfg = replace(TrainConfig(), optimizer_lr=args.lr, scheduler_decay_lr=args.decay_lr)
    print(f"[E0] training on {len(episodes)} libero_spatial episodes, "
          f"steps={args.steps}, save_freq={args.save_freq}, lr={cfg.optimizer_lr}, "
          f"decay_lr={cfg.scheduler_decay_lr}, lora_r={cfg.lora_r}", flush=True)

    res = train_smolvla_lora(
        output_dir=Path(args.out) / f"seed{args.seed}" / "train",
        episodes=episodes,
        suite="libero_spatial",
        cfg=cfg,
        steps=args.steps,
        seed=args.seed,
        extra_args=[f"--save_freq={args.save_freq}", "--num_workers=16", "--eval_freq=0"],
    )
    print(f"[E0] done rc={res.returncode} ckpt={res.checkpoint_dir}", flush=True)
    return res.returncode


if __name__ == "__main__":
    raise SystemExit(main())
