"""Experiment configuration for the LIBERO active-query pipeline.

All configs are frozen dataclasses (immutable). See ../../refine-logs/EXPERIMENT_PLAN.md
for how these map onto Phase S steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

QueryMethodName = Literal[
    "random", "entropy", "knn", "human_gated", "conformal",
    "dispersion", "dispersion_quota",
]

# LIBERO suites used as the main-results benchmark (PS.6).
LIBERO_SUITES: tuple[str, ...] = ("libero_spatial", "libero_object")

# SmolVLA base policy on the HuggingFace hub.
SMOLVLA_BASE = "lerobot/smolvla_base"
LIBERO_DATASET = "HuggingFaceVLA/libero"


@dataclass(frozen=True)
class ConformalConfig:
    """Conformal action-uncertainty settings.

    target_alpha: desired miscoverage rate -> query rate target (0.1 == query ~10%).
    iqt_gamma:    online step size for Intermittent Quantile Tracking (ConformalDAgger).
    calib_min:    minimum calibration points before the threshold is trusted.
    n_action_samples: action chunks sampled per step to estimate raw dispersion.
    """

    target_alpha: float = 0.1
    iqt_gamma: float = 0.05
    calib_min: int = 20
    n_action_samples: int = 8
    use_iqt: bool = True  # False -> vanilla split conformal (ablation A1)


@dataclass(frozen=True)
class TrainConfig:
    """SmolVLA LoRA fine-tune settings (passed through to `lerobot-train`)."""

    policy_path: str = SMOLVLA_BASE
    lora_r: int = 64
    lora_alpha: int = 64
    optimizer_lr: float = 1e-3
    scheduler_decay_lr: float = 1e-4
    batch_size: int = 32
    # Steps per active round are deliberately small: each round retrains on the
    # growing buffer rather than training once to convergence.
    steps_per_round: int = 4000


@dataclass(frozen=True)
class ActiveLoopConfig:
    """Active-query loop settings (PS.5)."""

    seed_demos: int = 5
    demos_per_round: int = 5
    max_demos: int = 40
    eval_episodes: int = 20  # rollouts per evaluation point
    recalibrate_each_round: bool = True  # conformal must recalibrate after each LoRA update
    # Random subsample of candidates to score each round; None == score the whole pool.
    # Used to bound the cost of policy-inference-based scoring; the smoke and the PS.6
    # sweep both rely on this to stay tractable on a single GPU.
    max_candidates_per_round: int | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level config for one (method, suite, seed) cell of the sweep."""

    method: QueryMethodName
    suite: str
    seed: int
    conformal: ConformalConfig = field(default_factory=ConformalConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    loop: ActiveLoopConfig = field(default_factory=ActiveLoopConfig)
    results_dir: str = "results"

    @property
    def run_id(self) -> str:
        return f"{self.method}__{self.suite}__seed{self.seed}"

    def with_method(self, method: QueryMethodName) -> "ExperimentConfig":
        return replace(self, method=method)
