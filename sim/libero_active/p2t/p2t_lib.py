"""P2T shared library — dataset access + state derivation for the replay engine.

Dataset: HuggingFaceVLA/libero (v3, local HF cache). libero_spatial episodes are
listed per task in results/suite_episode_map.json (task_id str -> episode ids).
Raw parquet access bisected by episode_index (the meta `data/file_index` column
does NOT match the on-disk file packing — verified 2026-07-06).

State convention (matches LiberoProcessorStep): [eef_pos(3), axisangle(3),
gripper_qpos(2)], quat in (x,y,z,w).
"""
from __future__ import annotations

import functools
import glob
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO = Path(__file__).resolve().parents[1]
SUITE_MAP = REPO / "results/suite_episode_map.json"


@functools.lru_cache(maxsize=1)
def _snapshot_dir() -> str:
    pat = os.path.expanduser(
        "~/.cache/huggingface/lerobot/hub/datasets--HuggingFaceVLA--libero/snapshots/*/")
    return sorted(glob.glob(pat))[0]


@functools.lru_cache(maxsize=1)
def spatial_episode_map() -> dict[int, list[int]]:
    m = json.load(open(SUITE_MAP))["libero_spatial"]
    return {int(k): sorted(int(e) for e in v) for k, v in m.items()}


@functools.lru_cache(maxsize=1)
def _file_episode_index() -> list[tuple[str, int, int]]:
    """(path, ep_min, ep_max) per data parquet, in episode order."""
    files = sorted(glob.glob(_snapshot_dir() + "data/chunk-*/*.parquet"))
    out = []
    for f in files:
        d = pd.read_parquet(f, columns=["episode_index"])
        out.append((f, int(d.episode_index.min()), int(d.episode_index.max())))
    return out

_EP_CACHE: dict[str, pd.DataFrame] = {}


def episode_frames(ep: int) -> pd.DataFrame:
    """All rows of one episode (state, action, images as encoded bytes)."""
    for f, mn, mx in _file_episode_index():
        if mn <= ep <= mx:
            if f not in _EP_CACHE:
                if len(_EP_CACHE) > 4:
                    _EP_CACHE.clear()
                _EP_CACHE[f] = pd.read_parquet(f)
            d = _EP_CACHE[f]
            return d[d.episode_index == ep].sort_values("frame_index")
    raise KeyError(f"episode {ep} not found in data files")


def decode_image(cell) -> np.ndarray:
    """Decode a parquet image cell (dict with 'bytes' or raw bytes) to HxWx3 uint8."""
    from PIL import Image
    raw = cell["bytes"] if isinstance(cell, dict) else cell
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))


def quat2axisangle(quat_xyzw: np.ndarray) -> np.ndarray:
    """(4,) xyzw -> (3,) axis-angle; mirrors LiberoProcessorStep._quat2axisangle."""
    q = np.asarray(quat_xyzw, dtype=np.float64)
    w = float(np.clip(q[3], -1.0, 1.0))
    den = np.sqrt(max(1.0 - w * w, 0.0))
    if den <= 1e-10:
        return np.zeros(3)
    return (q[:3] / den) * (2.0 * np.arccos(w))


def env_state8(raw_obs: dict) -> np.ndarray:
    """8-D dataset-convention state from a LiberoEnv raw obs."""
    rs = raw_obs["robot_state"]
    return np.concatenate([
        np.asarray(rs["eef"]["pos"], dtype=np.float64),
        quat2axisangle(np.asarray(rs["eef"]["quat"], dtype=np.float64)),
        np.asarray(rs["gripper"]["qpos"], dtype=np.float64),
    ])


def gripper_crossings(actions: np.ndarray) -> tuple[int | None, int | None]:
    """(t_grasp, t_release) from the gripper channel (open<=0 -> close>0 and back)."""
    g = actions[:, 6]
    close = np.where((g[1:] > 0.0) & (g[:-1] <= 0.0))[0]
    t_grasp = int(close[0]) + 1 if len(close) else None
    if t_grasp is None:
        return None, None
    opens = np.where((g[t_grasp + 1:] <= 0.0) & (g[t_grasp:-1] > 0.0))[0]
    t_release = (t_grasp + 1 + int(opens[0])) if len(opens) else None
    return t_grasp, t_release
