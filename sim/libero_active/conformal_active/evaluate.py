"""Policy evaluation — a thin wrapper around `lerobot-eval`.

Runs a trained SmolVLA checkpoint on one or more LIBERO suites and parses the
`eval_info.json` LeRobot writes (its `aggregated.pc_success` field).
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvalResult:
    suite: str
    pc_success: float          # percent, 0-100
    n_episodes: int
    output_dir: Path
    returncode: int

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def _read_pc_success(eval_dir: Path) -> tuple[float, int]:
    """Parse aggregated.pc_success from LeRobot's eval_info.json."""
    info_path = eval_dir / "eval_info.json"
    if not info_path.exists():
        # Some LeRobot versions nest it; search one level down.
        candidates = list(eval_dir.rglob("eval_info.json"))
        if not candidates:
            raise FileNotFoundError(f"no eval_info.json under {eval_dir}")
        info_path = candidates[0]
    info = json.loads(info_path.read_text())
    # LeRobot 0.5.x nests aggregated metrics under "overall"; older builds used
    # "aggregated". Fall back across both so the parser is version-robust.
    agg = info.get("overall") or info.get("aggregated") or info
    pc = float(agg.get("pc_success", float("nan")))
    n = int(agg.get("n_episodes", 0)) or len(info.get("per_task", []))
    return pc, n


def evaluate_checkpoint(
    *,
    checkpoint_dir: Path,
    suite: str,
    output_dir: Path,
    n_episodes: int = 20,
    batch_size: int = 1,
    seed: int = 1000,
    task_ids: list[int] | None = None,
) -> EvalResult:
    """Evaluate a checkpoint on a LIBERO suite over `n_episodes` rollouts.

    If `task_ids` is given, evaluation is restricted to those task indices (the PS.4
    single-task gate); otherwise the whole suite is evaluated.
    """
    checkpoint_dir = Path(checkpoint_dir).resolve()
    output_dir = Path(output_dir).resolve()
    # Do not pre-create output_dir — LeRobot scripts may refuse an existing dir.
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "lerobot-eval",
        f"--policy.path={checkpoint_dir}",
        "--env.type=libero",
        f"--env.task={suite}",
        f"--eval.batch_size={batch_size}",
        f"--eval.n_episodes={n_episodes}",
        "--env.max_parallel_tasks=1",
        f"--output_dir={output_dir}",
        f"--seed={seed}",
    ]
    if task_ids is not None:
        cmd.append(f"--env.task_ids={json.dumps(task_ids, separators=(',', ':'))}")

    log_path = output_dir.parent / f"{output_dir.name}.eval.log"
    with log_path.open("w") as log:
        log.write("CMD: " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)

    pc, n = (float("nan"), 0)
    if proc.returncode == 0:
        try:
            pc, n = _read_pc_success(output_dir)
        except FileNotFoundError:
            pass

    return EvalResult(
        suite=suite,
        pc_success=pc,
        n_episodes=n or n_episodes,
        output_dir=output_dir,
        returncode=proc.returncode,
    )
