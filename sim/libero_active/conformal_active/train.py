"""SmolVLA LoRA fine-tuning — a thin, well-typed wrapper around `lerobot-train`.

We shell out to the battle-tested LeRobot CLI rather than re-implementing the training
loop. The only research-relevant control we add is `episodes`: the active-query loop
trains on a *growing subset* of demonstration episodes (the demo budget), so each round
passes a different episode index list.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import LIBERO_DATASET, TrainConfig


@dataclass(frozen=True)
class TrainResult:
    output_dir: Path
    checkpoint_dir: Path
    returncode: int

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and self.checkpoint_dir.exists()


def _latest_checkpoint(output_dir: Path) -> Path:
    """LeRobot writes checkpoints under <output_dir>/checkpoints/<step>/pretrained_model."""
    ckpt_root = output_dir / "checkpoints"
    if not ckpt_root.exists():
        return output_dir
    steps = sorted((p for p in ckpt_root.iterdir() if p.is_dir()), key=lambda p: p.name)
    if not steps:
        return output_dir
    pretrained = steps[-1] / "pretrained_model"
    return pretrained if pretrained.exists() else steps[-1]


def train_smolvla_lora(
    *,
    output_dir: Path,
    episodes: list[int],
    suite: str,
    cfg: TrainConfig,
    steps: int | None = None,
    dataset_repo_id: str = LIBERO_DATASET,
    seed: int = 0,
    extra_args: list[str] | None = None,
) -> TrainResult:
    """Fine-tune SmolVLA with LoRA on a chosen subset of LIBERO episodes.

    Args:
        output_dir: where LeRobot writes checkpoints/logs.
        episodes: episode indices forming the current demo budget.
        suite: LIBERO suite name (e.g. "libero_spatial").
        cfg: LoRA / optimizer settings.
        steps: training steps; defaults to cfg.steps_per_round.
        seed: training seed.
    """
    # LeRobot creates output_dir itself and refuses to write into an existing one,
    # so we must NOT pre-create it. Only the parent is ensured to exist.
    output_dir = Path(output_dir).resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    n_steps = cfg.steps_per_round if steps is None else steps

    cmd: list[str] = [
        "lerobot-train",
        f"--policy.path={cfg.policy_path}",
        "--policy.push_to_hub=false",  # local-only research runs; avoids needing a hub repo_id
        f"--dataset.repo_id={dataset_repo_id}",
        # compact JSON (no spaces) so the CLI parser sees a single token
        f"--dataset.episodes={json.dumps(episodes, separators=(',', ':'))}",
        "--policy.output_features=null",
        "--policy.input_features=null",
        f"--policy.optimizer_lr={cfg.optimizer_lr}",
        f"--policy.scheduler_decay_lr={cfg.scheduler_decay_lr}",
        "--env.type=libero",
        f"--env.task={suite}",
        f"--steps={n_steps}",
        f"--batch_size={cfg.batch_size}",
        "--peft.method_type=LORA",
        f"--peft.r={cfg.lora_r}",
        f"--peft.lora_alpha={cfg.lora_alpha}",
        f"--output_dir={output_dir}",
        f"--seed={seed}",
        "--wandb.enable=false",
    ]
    if extra_args:
        cmd.extend(extra_args)

    # Log lives beside output_dir (we cannot write inside it before LeRobot creates it).
    log_path = output_dir.parent / f"{output_dir.name}.train.log"
    with log_path.open("w") as log:
        log.write("CMD: " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)

    return TrainResult(
        output_dir=output_dir,
        checkpoint_dir=_latest_checkpoint(output_dir),
        returncode=proc.returncode,
    )
