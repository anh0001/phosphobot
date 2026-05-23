"""Map a LIBERO suite name to its episode indices in the combined dataset.

`HuggingFaceVLA/libero` is the *combined* dataset: 1693 episodes spanning 40 tasks
across all four suites. `--env.task=libero_spatial` only controls the eval *env* — the
`--dataset.episodes` budget must independently be drawn from that suite's episodes, or
the policy trains on the wrong suite entirely (the PS.4 v1 bug: trained on libero_10,
evaluated on libero_spatial -> 0%).

This module matches each dataset episode's task language string against the canonical
task list of each LIBERO suite, and caches the result to JSON.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from .config import LIBERO_DATASET

_CACHE_PATH = Path(__file__).parent.parent / "results" / "suite_episode_map.json"


def _build_suite_map(dataset_repo_id: str) -> dict[str, dict[str, list[int]]]:
    """Build {suite: {str(task_id): [episode_index, ...]}} by matching task strings.

    Episodes within a suite are interleaved across tasks in the combined dataset, so
    we record the per-task grouping (needed for single-task and stratified selection).
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from libero.libero import benchmark

    meta = LeRobotDatasetMetadata(dataset_repo_id)
    bench = benchmark.get_benchmark_dict()

    suite_map: dict[str, dict[str, list[int]]] = {}
    for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
        s = bench[suite]()
        lang_to_task = {s.get_task(i).language.strip().lower(): i for i in range(s.n_tasks)}
        per_task: dict[str, list[int]] = {str(i): [] for i in range(s.n_tasks)}
        for r in meta.episodes:
            tid = lang_to_task.get(r["tasks"][0].strip().lower())
            if tid is not None:
                per_task[str(tid)].append(int(r["episode_index"]))
        suite_map[suite] = {k: sorted(v) for k, v in per_task.items()}
    return suite_map


@lru_cache(maxsize=1)
def _suite_map_cached(dataset_repo_id: str) -> dict[str, dict[str, list[int]]]:
    if _CACHE_PATH.exists():
        cached = json.loads(_CACHE_PATH.read_text())
        if cached.get("_dataset") == dataset_repo_id and cached.get("_schema") == 2:
            return {k: v for k, v in cached.items() if not k.startswith("_")}
    suite_map = _build_suite_map(dataset_repo_id)
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_PATH.write_text(
        json.dumps({"_dataset": dataset_repo_id, "_schema": 2, **suite_map}, indent=2)
    )
    return suite_map


def suite_episode_indices(suite: str, dataset_repo_id: str = LIBERO_DATASET) -> list[int]:
    """All episode indices belonging to `suite` (every task), sorted."""
    suite_map = _suite_map_cached(dataset_repo_id)
    if suite not in suite_map:
        raise ValueError(f"unknown suite '{suite}'; known: {sorted(suite_map)}")
    eps = sorted(e for task_eps in suite_map[suite].values() for e in task_eps)
    if not eps:
        raise RuntimeError(f"no episodes matched suite '{suite}' in {dataset_repo_id}")
    return eps


def task_episode_indices(
    suite: str, task_id: int, dataset_repo_id: str = LIBERO_DATASET
) -> list[int]:
    """Episode indices for a single task within `suite` — used by the PS.4 gate."""
    suite_map = _suite_map_cached(dataset_repo_id)
    if suite not in suite_map:
        raise ValueError(f"unknown suite '{suite}'; known: {sorted(suite_map)}")
    eps = suite_map[suite].get(str(task_id), [])
    if not eps:
        raise RuntimeError(f"no episodes for suite '{suite}' task {task_id}")
    return eps
