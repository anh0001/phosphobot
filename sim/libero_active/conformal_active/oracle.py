"""Corrective-demo source for the active-query loop (PS.2).

Path A uses **pool-based active demonstration selection**, the cleanest and most
reproducible flavour for a sim-main paper:

- LIBERO ships a fixed pool of expert human demonstrations (one per episode index).
- The active loop holds a *budget* of episode indices already in training.
- Each round, a query method scores the remaining candidate episodes; the highest-scored
  ones are "queried" (added to the budget). The LIBERO demo IS the oracle correction —
  no motion planner needed, and selection is fully deterministic given a seed.

This avoids re-implementing an online robosuite planner while keeping the research
question intact: *which* demonstrations a query method chooses, under a shared budget.

An episode is scored by how uncertain the current policy is across that episode's
states — a high-uncertainty episode is informative to add. `episode_uncertainty`
aggregates per-step scores; the per-step signal comes from the query method.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .query_methods import QueryMethod, StepContext


@dataclass
class DemoPool:
    """The fixed pool of LIBERO demonstration episodes.

    `total_episodes` is read from the LeRobot dataset metadata (see active_loop.py).
    The pool is partitioned into `budget` (in training) and `candidates` (selectable).
    """

    total_episodes: int
    budget: list[int]

    @classmethod
    def with_seed(cls, total_episodes: int, seed_size: int, rng: np.random.Generator) -> "DemoPool":
        """Initialise with a random seed budget of `seed_size` episodes."""
        order = rng.permutation(total_episodes).tolist()
        return cls(total_episodes=total_episodes, budget=sorted(order[:seed_size]))

    @property
    def candidates(self) -> list[int]:
        in_budget = set(self.budget)
        return [i for i in range(self.total_episodes) if i not in in_budget]

    def add(self, episode_indices: list[int]) -> "DemoPool":
        """Return a new pool with `episode_indices` moved into the budget (immutable)."""
        return DemoPool(
            total_episodes=self.total_episodes,
            budget=sorted(set(self.budget) | set(episode_indices)),
        )


def episode_uncertainty(
    method: QueryMethod,
    *,
    action_samples_per_step: list[np.ndarray],
    features_per_step: list[np.ndarray] | None = None,
    oracle_disagreement_per_step: list[float] | None = None,
) -> float:
    """Aggregate a query method's per-step scores over one candidate episode.

    Args:
        method: the query strategy under test.
        action_samples_per_step: list of (K, T, A) arrays — the current policy's
            sampled action chunks at each state of the candidate episode.
        features_per_step / oracle_disagreement_per_step: optional per-step signals
            required by the kNN and human-gated methods respectively.

    Returns:
        Mean per-step score. Higher == this episode is more informative to add.
    """
    n = len(action_samples_per_step)
    if n == 0:
        return 0.0
    scores = []
    for t in range(n):
        ctx = StepContext(
            step=t,
            action_samples=action_samples_per_step[t],
            feature=None if features_per_step is None else features_per_step[t],
            oracle_disagreement=(
                None if oracle_disagreement_per_step is None
                else oracle_disagreement_per_step[t]
            ),
        )
        scores.append(method.score(ctx))
    return float(np.mean(scores))


def select_episodes(
    candidate_scores: dict[int, float],
    n_to_select: int,
) -> list[int]:
    """Pick the `n_to_select` highest-scoring candidate episodes (the demo budget step).

    Ties are broken by lowest episode index for determinism.
    """
    ranked = sorted(candidate_scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return sorted(idx for idx, _ in ranked[:n_to_select])
