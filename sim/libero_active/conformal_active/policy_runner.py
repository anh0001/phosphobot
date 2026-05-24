"""Per-episode uncertainty signal for candidate scoring (PS.5 integration point).

Pool-based active IL: each round we score every candidate episode by "how much
would adding it help?" We use **mean policy training-loss on the episode** as the
raw uncertainty signal — the standard pool-based AL proxy and far simpler than
reconstructing SmolVLA's full inference preprocessor stack just to sample action
chunks. Conformal calibration on these per-episode losses gives the calibrated
query decision the conformal method needs.

Loads the per-round LoRA-fine-tuned checkpoint via LeRobot's real 0.5.x API:
`PreTrainedConfig.from_pretrained` + `make_policy(cfg, ds_meta=...)`. Pulling the
dataset metadata at policy-construction time lets LeRobot infer feature shapes
without us having to wire up an env config.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch


@dataclass
class EpisodeSignals:
    """Per-episode signals consumed by the active loop.

    `loss_mean` is the raw uncertainty score (higher == more informative).
    `feature` is a cheap embedding for kNN-coverage scoring (we use the mean of
    the episode's proprioception so no separate vision encoder is needed).
    """

    loss_mean: float
    feature: np.ndarray


@lru_cache(maxsize=2)
def _load_policy_and_meta(checkpoint_dir: str, dataset_repo_id: str):
    """Cached policy/preprocessor/meta loader — one (checkpoint, dataset) pair per round.

    Returns (policy, preprocessor, ds_meta). The preprocessor is essential: SmolVLA's
    forward() expects already-tokenized language (`observation.language.tokens`) and
    normalised images, both produced by the saved processor pipeline.
    """
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(checkpoint_dir)
    cfg.pretrained_path = checkpoint_dir  # so the LoRA adapter loads
    ds_meta = LeRobotDatasetMetadata(dataset_repo_id)
    policy = make_policy(cfg=cfg, ds_meta=ds_meta)
    policy.eval()
    preprocessor, _ = make_pre_post_processors(policy_cfg=cfg, pretrained_path=checkpoint_dir)
    return policy, preprocessor, ds_meta


def reset_policy_cache() -> None:
    """Call between active-loop rounds — the checkpoint changes each retrain."""
    _load_policy_and_meta.cache_clear()


@torch.no_grad()
def episode_signals(
    *,
    checkpoint_dir,
    dataset_repo_id: str,
    episode_index: int,
    n_action_samples: int = 1,   # unused in the loss-based flow; kept for API symmetry
    frame_stride: int = 8,
    max_frames: int = 12,
) -> EpisodeSignals:
    """Compute mean policy loss + a cheap feature embedding for one episode."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    policy, preprocessor, _ = _load_policy_and_meta(str(checkpoint_dir), dataset_repo_id)
    device = next(policy.parameters()).device

    ds = LeRobotDataset(dataset_repo_id, episodes=[episode_index])
    indices = list(range(0, len(ds), frame_stride))[:max_frames]
    if not indices:
        return EpisodeSignals(loss_mean=0.0, feature=np.zeros(1, dtype=np.float32))

    losses: list[float] = []
    feat_parts: list[np.ndarray] = []
    for i in indices:
        frame = ds[i]
        # Build batch (add a batch dim for tensors, keep scalars/strings as-is).
        batch = {
            k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else v)
            for k, v in frame.items()
        }
        # The preprocessor tokenises language, normalises images, and moves to device.
        batch = preprocessor(batch)
        out = policy.forward(batch)
        loss = out["loss"] if isinstance(out, dict) else out[0]
        losses.append(float(loss.detach().cpu()))
        if "observation.state" in frame and isinstance(frame["observation.state"], torch.Tensor):
            feat_parts.append(frame["observation.state"].detach().cpu().float().numpy())

    feature = np.mean(np.stack(feat_parts), axis=0) if feat_parts else np.zeros(1, dtype=np.float32)
    return EpisodeSignals(loss_mean=float(np.mean(losses)), feature=feature.astype(np.float32))
