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

from .config import LIBERO_DATASET, ExperimentConfig
from .evaluate import evaluate_checkpoint
from .oracle import DemoPool, select_episodes
from .query_methods import QueryMethod, build_query_method
from .suite_episodes import suite_episode_indices
from .train import TrainResult, train_smolvla_lora


def score_candidate_episodes(
    method: QueryMethod,
    *,
    candidates: list[int],
    checkpoint_dir: Path,
    dataset_repo_id: str,
    n_action_samples: int,
    rng: np.random.Generator,
    budget_features: list[np.ndarray] | None = None,
) -> dict[int, float]:
    """Score each candidate episode by how informative it is to add (pool-based AL).

    random      -> uniform random (no policy load)
    knn         -> mean L2 distance to the budget's episode features (no policy load)
    entropy     -> raw mean policy loss on the episode
    conformal   -> threshold-normalised policy loss (calibrated by the loop)
    human_gated -> falls back to raw loss in sim (no live human signal)
    """
    if method.name == "random":
        return {idx: float(rng.random()) for idx in candidates}

    from .policy_runner import episode_signals  # heavy deps, loaded lazily

    scores: dict[int, float] = {}
    for idx in candidates:
        sig = episode_signals(
            checkpoint_dir=checkpoint_dir,
            dataset_repo_id=dataset_repo_id,
            episode_index=idx,
            n_action_samples=n_action_samples,
        )
        if method.name == "knn":
            if not budget_features:
                scores[idx] = 1.0  # nothing in the budget yet -> max novelty
            else:
                feats = np.stack(budget_features)
                dists = np.linalg.norm(feats - sig.feature[None, :feats.shape[1]], axis=1)
                scores[idx] = float(np.sort(dists)[: min(5, len(dists))].mean())
        elif method.name == "conformal":
            from .query_methods import ConformalQuery  # narrow runtime cast
            assert isinstance(method, ConformalQuery)
            method.uncertainty.add_calibration(sig.loss_mean)  # opportunistic calib
            raw = sig.loss_mean
            scores[idx] = method.uncertainty.normalized(raw) if method.uncertainty.is_calibrated else raw
        else:  # entropy / human_gated proxy
            scores[idx] = sig.loss_mean
    return scores


def run_active_loop(cfg: ExperimentConfig) -> dict:
    """Run one full active-query loop. Returns the sample-efficiency curve + metadata."""
    results_dir = Path(cfg.results_dir) / cfg.run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)

    suite_eps = suite_episode_indices(cfg.suite, LIBERO_DATASET)
    pool = DemoPool.with_seed(suite_eps, cfg.loop.seed_demos, rng)
    method = build_query_method(cfg.method, seed=cfg.seed, conformal=cfg.conformal)
    budget_features: list[np.ndarray] = []  # populated as candidates are accepted

    curve: list[dict] = []
    round_idx = 0
    while True:
        method.reset_round()
        # The trained checkpoint changes every round; clear the policy-loader cache.
        from .policy_runner import reset_policy_cache
        reset_policy_cache()
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

        # Score candidates and grow the budget. Subsample to bound inference cost.
        candidates = pool.candidates
        if cfg.loop.max_candidates_per_round and len(candidates) > cfg.loop.max_candidates_per_round:
            picked_idx = rng.choice(len(candidates), cfg.loop.max_candidates_per_round, replace=False)
            candidates = [candidates[i] for i in sorted(picked_idx)]
        scores = score_candidate_episodes(
            method,
            candidates=candidates,
            checkpoint_dir=train_res.checkpoint_dir,
            dataset_repo_id=LIBERO_DATASET,
            n_action_samples=cfg.conformal.n_action_samples,
            rng=rng,
            budget_features=budget_features,
        )
        picked = select_episodes(scores, cfg.loop.demos_per_round)
        # Cache the picked episodes' features for next round's kNN scoring.
        if cfg.method == "knn":
            from .policy_runner import episode_signals
            for idx in picked:
                sig = episode_signals(
                    checkpoint_dir=train_res.checkpoint_dir,
                    dataset_repo_id=LIBERO_DATASET,
                    episode_index=idx,
                    n_action_samples=1,
                )
                budget_features.append(sig.feature)
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
