"""E0 — train the strong checkpoint: SmolVLA LoRA on ALL libero_spatial demos.

Full-suite multi-task fine-tune (432 episodes, 10 tasks) with intermediate
checkpoints every 5k steps (emergence axis: bias vs training progress).
Third-party hub checkpoints were screened and rejected (0% under our eval
conventions; no reported numbers — see PILOT_LOG.md).

Usage:
  env -u PYTHONPATH MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    python preempt/e0_train_strong.py [--steps 40000] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conformal_active.config import TrainConfig  # noqa: E402
from conformal_active.train import train_smolvla_lora  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-freq", type=int, default=5000)
    ap.add_argument("--out", default="results/e0_strong_full432")
    args = ap.parse_args()

    suite_map = json.load(open("results/suite_episode_map.json"))["libero_spatial"]
    episodes = sorted(int(e) for eps in suite_map.values() for e in eps)
    print(f"[E0] training on {len(episodes)} libero_spatial episodes, "
          f"steps={args.steps}, save_freq={args.save_freq}", flush=True)

    res = train_smolvla_lora(
        output_dir=Path(args.out) / f"seed{args.seed}" / "train",
        episodes=episodes,
        suite="libero_spatial",
        cfg=TrainConfig(),
        steps=args.steps,
        seed=args.seed,
        extra_args=[f"--save_freq={args.save_freq}", "--num_workers=16"],
    )
    print(f"[E0] done rc={res.returncode} ckpt={res.checkpoint_dir}", flush=True)
    return res.returncode


if __name__ == "__main__":
    raise SystemExit(main())
