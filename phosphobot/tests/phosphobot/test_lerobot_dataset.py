import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from phosphobot.models.dataset import Observation, Step
from phosphobot.models.lerobot_dataset import (
    EpisodesModel,
    EpisodesStatsModel,
    LeRobotDataset,
    LeRobotEpisode,
    StatsModel,
    TasksModel,
)


def _make_step(timestamp: float, joints_position: list[float]) -> Step:
    return Step(
        observation=Observation(
            joints_position=np.array(joints_position, dtype=np.float32),
            timestamp=timestamp,
            language_instruction="collect training data",
            main_image=np.zeros((1, 1, 3), dtype=np.uint8),
        )
    )


@pytest.mark.parametrize("format_version", ["lerobot_v2", "lerobot_v2.1"])
def test_stats_timestamps_match_normalized_parquet_timestamps(
    tmp_path, format_version: str
):
    dataset_path = tmp_path / format_version / "timestamp_alignment"
    dataset_manager = LeRobotDataset(path=str(dataset_path))
    dataset_manager.info_model = SimpleNamespace(total_frames=0)
    dataset_manager.episodes_model = EpisodesModel(episodes=[])
    dataset_manager.tasks_model = TasksModel(tasks=[])
    if format_version == "lerobot_v2.1":
        dataset_manager.episodes_stats_model = EpisodesStatsModel()
    else:
        dataset_manager.stats_model = StatsModel()

    episode = LeRobotEpisode(
        steps=[],
        metadata={
            "episode_index": 0,
            "format": format_version,
            "dataset_name": dataset_manager.dataset_name,
            "task_index": 0,
        },
        dataset_manager=dataset_manager,
        freq=15,
        codec="mp4v",
        target_size=(1, 1),
    )

    raw_timestamps = [0.0, 4.75, 9.5, 53.06]
    joints_positions = [
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
    ]
    for timestamp, joints_position in zip(raw_timestamps, joints_positions):
        step = _make_step(timestamp=timestamp, joints_position=joints_position)
        episode.update_previous_step(step)
        asyncio.run(episode.append_step(step))

    parquet_model = episode._convert_to_le_robot_episode_model()
    expected_timestamps = [frame_index / episode.freq for frame_index in range(4)]

    assert parquet_model.timestamp == pytest.approx(expected_timestamps)

    if format_version == "lerobot_v2.1":
        assert dataset_manager.episodes_stats_model is not None
        timestamp_stats = dataset_manager.episodes_stats_model.episodes_stats[
            0
        ].stats.timestamp
    else:
        assert dataset_manager.stats_model is not None
        timestamp_stats = dataset_manager.stats_model.timestamp

    assert np.asarray(timestamp_stats.min).item() == pytest.approx(
        expected_timestamps[0]
    )
    assert np.asarray(timestamp_stats.max).item() == pytest.approx(
        expected_timestamps[-1]
    )
    assert np.asarray(timestamp_stats.max).item() != pytest.approx(raw_timestamps[-1])
