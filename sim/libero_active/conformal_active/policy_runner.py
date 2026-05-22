"""Live SmolVLA inference for candidate-episode scoring (PS.5 integration point).

`episode_signals` loads a candidate LIBERO demonstration episode and, for each state,
samples K action chunks from the current policy. The query methods turn those samples
into per-step scores (dispersion / conformal uncertainty / oracle disagreement).

This is the one module that depends on a trained SmolVLA checkpoint, so it can only be
exercised after the PS.4 offline-sanity gate is green. Until then it is import-light:
`active_loop.py` imports it lazily and the `random` query method never touches it.

Design notes
- SmolVLA is a flow-matching policy: sampling K chunks == K forward passes with K noise
  seeds. We toggle the policy's RNG between calls to get genuine sample spread.
- `oracle_disagreement` compares the policy's mean action to the dataset's recorded
  expert action at the same state — the privileged signal the human_gated proxy uses.
- `features` are the policy vision-encoder embeddings, used by the kNN method.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class EpisodeSignals:
    """Per-step signals for one candidate episode, consumed by query methods."""

    action_samples: list[np.ndarray]              # each (K, T, A)
    features: list[np.ndarray] | None             # each (D,)
    oracle_disagreement: list[float] | None       # each scalar ||policy_a - expert_a||


def _load_policy(checkpoint_dir):
    """Load a trained SmolVLA policy from a LeRobot checkpoint directory."""
    from lerobot.policies.factory import make_policy_from_pretrained  # type: ignore

    policy = make_policy_from_pretrained(str(checkpoint_dir))
    policy.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return policy.to(device), device


def _load_episode_frames(dataset_repo_id: str, episode_index: int, stride: int):
    """Yield (observation_batch, expert_action) for a strided subset of episode frames."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore

    ds = LeRobotDataset(dataset_repo_id, episodes=[episode_index])
    for i in range(0, len(ds), stride):
        yield ds[i]


@torch.no_grad()
def episode_signals(
    *,
    checkpoint_dir,
    dataset_repo_id: str,
    episode_index: int,
    n_action_samples: int,
    frame_stride: int = 8,
) -> EpisodeSignals:
    """Sample K action chunks per state across a candidate episode.

    The result feeds `oracle.episode_uncertainty`. Frame striding keeps candidate
    scoring affordable when the pool is large (LIBERO episodes are long).

    IMPLEMENTATION STATUS: structurally complete and written against the LeRobot
    policy/dataset API, but must be validated against a real checkpoint once PS.4
    passes — the exact obs-batch keys and the SmolVLA sampling entry point can vary
    by LeRobot minor version. Verify with `tests/test_policy_runner.py` (to add)
    against the PS.4 checkpoint before running the PS.6 sweep.
    """
    policy, device = _load_policy(checkpoint_dir)

    action_samples: list[np.ndarray] = []
    features: list[np.ndarray] = []
    oracle_disagreement: list[float] = []

    for frame in _load_episode_frames(dataset_repo_id, episode_index, frame_stride):
        obs = {
            k: v.unsqueeze(0).to(device)
            for k, v in frame.items()
            if isinstance(v, torch.Tensor) and k.startswith(("observation", "image"))
        }
        expert_action = np.asarray(frame["action"], dtype=np.float64)

        # K action-chunk samples from the flow-matching policy.
        chunks = []
        for k in range(n_action_samples):
            policy.reset()
            torch.manual_seed(1000 * episode_index + k)
            chunk = policy.predict_action_chunk(obs)  # (1, T, A)
            chunks.append(chunk.squeeze(0).float().cpu().numpy())
        samples = np.stack(chunks, axis=0)            # (K, T, A)
        action_samples.append(samples)

        mean_first_action = samples.mean(axis=0)[0]   # mean of K, first horizon step
        oracle_disagreement.append(
            float(np.linalg.norm(mean_first_action - expert_action[: mean_first_action.shape[0]]))
        )
        features.append(samples.reshape(-1))          # cheap embedding proxy for kNN

    return EpisodeSignals(
        action_samples=action_samples,
        features=features,
        oracle_disagreement=oracle_disagreement,
    )
