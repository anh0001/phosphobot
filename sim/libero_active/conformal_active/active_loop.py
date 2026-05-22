"""Active-query loop — PS.5.

One (method, suite, seed) cell of the main-results sweep. Pool-based active demo
selection:

    seed budget (N_0 episodes)
    repeat until budget == max_demos:
        1. LoRA fine-tune SmolVLA on the current budget
        2. evaluate -> record (N, success_rate)            <- the sample-efficiency curve
        3. (conformal only) recollect calibration scores, recalibrate
        4. score every candidate episode under the query method
        5. move the top-scored episodes into the budget

Output: a list of (n_demos, success_rate) points = one curve in the PS.6 main table.

NOTE ON POLICY INFERENCE. Step 4 needs the current policy's action-sample dispersion
on candidate-episode states. That is the one part that touches live SmolVLA inference;
it is factored into `score_candidate_episodes` and `_collect_episode_action_samples` so
it can be validated in isolation once PS.4 passes. Methods that need no policy inference
(random) or only dataset features (knn) are handled without loading the policy.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .evaluate import evaluate_checkpoint
from .oracle import DemoPool, select_episodes
from .query_methods import QueryMethod, build_query_method
from .train import TrainResult, train_smolvla_lora


def _dataset_episode_count(dataset_repo_id: str) -> int:
    """Read the total episode count from the LeRobot dataset metadata."""
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(dataset_repo_id)
    return int(meta.total_episodes)


def score_candidate_episodes(
    method: QueryMethod,
    *,
    candidates: list[int],
    checkpoint_dir: Path,
    dataset_repo_id: str,
    n_action_samples: int,
    rng: np.random.Generator,
) -> dict[int, float]:
    """Score each candidate episode by how informative it is to add.

    random  -> random score (no inference).
    knn     -> dataset state-feature novelty (no policy inference).
    others  -> mean policy action-sample dispersion over the episode's states.

    This is the integration point with live SmolVLA inference; see module docstring.
    """
    if method.name == "random":
        return {idx: float(rng.random()) for idx in candidates}

    # Policy-inference path (entropy / conformal / human_gated / knn-with-features).
    # Implemented lazily so random-method sweeps need no GPU policy load.
    from .policy_runner import episode_signals  # local import: heavy deps

    scores: dict[int, float] = {}
    for idx in candidates:
        signals = episode_signals(
            checkpoint_dir=checkpoint_dir,
            dataset_repo_id=dataset_repo_id,
            episode_index=idx,
            n_action_samples=n_action_samples,
        )
        from .oracle import episode_uncertainty

        scores[idx] = episode_uncertainty(
            method,
            action_samples_per_step=signals.action_samples,
            features_per_step=signals.features,
            oracle_disagreement_per_step=signals.oracle_disagreement,
        )
    return scores


def run_active_loop(cfg: ExperimentConfig) -> dict:
    """Run one full active-query loop. Returns the sample-efficiency curve + metadata."""
    results_dir = Path(cfg.results_dir) / cfg.run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)

    total_eps = _dataset_episode_count(cfg.train.policy_path and "HuggingFaceVLA/libero")
    pool = DemoPool.with_seed(total_eps, cfg.loop.seed_demos, rng)
    method = build_query_method(cfg.method, seed=cfg.seed, conformal=cfg.conformal)

    curve: list[dict] = []
    round_idx = 0
    while True:
        method.reset_round()
        round_dir = results_dir / f"round{round_idx}_N{len(pool.budget)}"

        train_res: TrainResult = train_smolvla_lora(
            output_dir=round_dir / "train",
            episodes=pool.budget,
            suite=cfg.suite,
            cfg=cfg.train,
            seed=cfg.seed,
        )
        if not train_res.ok:
            curve.append({"n_demos": len(pool.budget), "pc_success": float("nan"),
                          "note": f"train failed rc={train_res.returncode}"})
            break

        eval_res = evaluate_checkpoint(
            checkpoint_dir=train_res.checkpoint_dir,
            suite=cfg.suite,
            output_dir=round_dir / "eval",
            n_episodes=cfg.loop.eval_episodes,
            seed=1000 + cfg.seed,
        )
        curve.append({"n_demos": len(pool.budget), "pc_success": eval_res.pc_success,
                       "note": "ok" if eval_res.ok else "eval failed"})
        _dump(results_dir, cfg, curve)  # checkpoint the curve every round

        if len(pool.budget) >= cfg.loop.max_demos:
            break

        # Score candidates and grow the budget.
        scores = score_candidate_episodes(
            method,
            candidates=pool.candidates,
            checkpoint_dir=train_res.checkpoint_dir,
            dataset_repo_id="HuggingFaceVLA/libero",
            n_action_samples=cfg.conformal.n_action_samples,
            rng=rng,
        )
        picked = select_episodes(scores, cfg.loop.demos_per_round)
        for idx in picked:
            method.register_demo_state(np.zeros(1))  # kNN bookkeeping; features set in runner
        pool = pool.add(picked)
        round_idx += 1

    _dump(results_dir, cfg, curve)
    return {"run_id": cfg.run_id, "method": cfg.method, "suite": cfg.suite,
            "seed": cfg.seed, "curve": curve}


def _dump(results_dir: Path, cfg: ExperimentConfig, curve: list[dict]) -> None:
    payload = {
        "run_id": cfg.run_id,
        "method": cfg.method,
        "suite": cfg.suite,
        "seed": cfg.seed,
        "config": {"loop": asdict(cfg.loop), "conformal": asdict(cfg.conformal)},
        "curve": curve,
    }
    (results_dir / "curve.json").write_text(json.dumps(payload, indent=2))
