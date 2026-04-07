import time
from unittest.mock import create_autospec

import numpy as np
import pytest
from fastapi import BackgroundTasks
from scipy.spatial.transform import Rotation as R

from phosphobot.configs import config
from phosphobot.endpoints import control
from phosphobot.hardware import PiperHardware, SO100Hardware, get_sim
from phosphobot.hardware.base import BaseManipulator
from phosphobot.models import MoveAbsoluteRequest, RelativeEndEffectorPosition
from phosphobot.types import SimulationMode


class FakeRCM:
    def __init__(self, robot: BaseManipulator):
        self.robot = robot

    async def get_robot(self, robot_id: int) -> BaseManipulator:
        return self.robot


@pytest.fixture
def so100() -> SO100Hardware:
    config.SIM_MODE = SimulationMode.headless
    sim = get_sim()
    sim.reset()
    return SO100Hardware(only_simulation=True)


@pytest.fixture
def piper() -> PiperHardware:
    config.SIM_MODE = SimulationMode.headless
    sim = get_sim()
    sim.reset()
    return PiperHardware(only_simulation=True)


@pytest.mark.asyncio
async def test_move_to_absolute_position_uses_synced_fk() -> None:
    robot = create_autospec(BaseManipulator, instance=True)
    robot.name = "test-manipulator"
    robot.initial_position = np.zeros(3)
    robot.initial_orientation_rad = np.zeros(3)

    fk_sync_calls: list[bool] = []

    def forward_kinematics(sync_robot_pos: bool = False):
        fk_sync_calls.append(sync_robot_pos)
        return np.zeros(3), np.zeros(3)

    robot.forward_kinematics.side_effect = forward_kinematics

    await control.move_to_absolute_position(
        query=MoveAbsoluteRequest(
            x=0,
            y=0,
            z=0,
            rx=None,
            ry=None,
            rz=None,
            max_trials=1,
            position_tolerance=1e-6,
            orientation_tolerance=1e-6,
        ),
        background_tasks=BackgroundTasks(),
        rcm=FakeRCM(robot),
    )

    assert fk_sync_calls == [True]
    robot.move_robot_absolute.assert_not_called()


@pytest.mark.asyncio
async def test_move_relative_uses_synced_fk_for_controller_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robot = create_autospec(BaseManipulator, instance=True)
    robot.name = "test-manipulator"
    robot.initial_position = np.zeros(3)
    robot.initial_orientation_rad = np.zeros(3)

    synced_position = np.array([0.20, -0.04, 0.31])
    unsynced_position = np.array([0.99, 0.99, 0.99])
    current_orientation = np.array([0.0, 0.0, 0.0])

    fk_sync_calls: list[bool] = []

    def forward_kinematics(sync_robot_pos: bool = False):
        fk_sync_calls.append(sync_robot_pos)
        position = synced_position if sync_robot_pos else unsynced_position
        return position.copy(), current_orientation.copy()

    robot.forward_kinematics.side_effect = forward_kinematics

    captured_query: dict[str, MoveAbsoluteRequest] = {}

    async def fake_move_to_absolute_position(
        query: MoveAbsoluteRequest,
        background_tasks: BackgroundTasks,
        robot_id: int = 0,
        rcm: FakeRCM | None = None,
    ) -> None:
        captured_query["query"] = query

    monkeypatch.setattr(
        control, "move_to_absolute_position", fake_move_to_absolute_position
    )

    await control.move_relative(
        data=RelativeEndEffectorPosition(
            x=1,
            y=-2,
            z=3,
            rx=0,
            ry=0,
            rz=0,
            open=None,
        ),
        background_tasks=BackgroundTasks(),
        rcm=FakeRCM(robot),
    )

    query = captured_query["query"]

    assert fk_sync_calls == [True]
    assert query.x == pytest.approx(21.0)
    assert query.y == pytest.approx(-6.0)
    assert query.z == pytest.approx(34.0)


@pytest.mark.asyncio
async def test_move_relative_without_orientation_delta_uses_position_only_ik(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robot = create_autospec(BaseManipulator, instance=True)
    robot.name = "test-manipulator"
    robot.initial_position = np.zeros(3)
    robot.initial_orientation_rad = np.array([0.4, -0.2, 0.1])

    current_position = np.array([0.20, -0.04, 0.31])
    current_orientation = np.array([0.25, -0.15, 0.05])

    def forward_kinematics(sync_robot_pos: bool = False):
        return current_position.copy(), current_orientation.copy()

    robot.forward_kinematics.side_effect = forward_kinematics

    captured_query: dict[str, MoveAbsoluteRequest] = {}

    async def fake_move_to_absolute_position(
        query: MoveAbsoluteRequest,
        background_tasks: BackgroundTasks,
        robot_id: int = 0,
        rcm: FakeRCM | None = None,
    ) -> None:
        captured_query["query"] = query

    monkeypatch.setattr(
        control, "move_to_absolute_position", fake_move_to_absolute_position
    )

    await control.move_relative(
        data=RelativeEndEffectorPosition(
            x=1,
            y=-2,
            z=3,
            rx=None,
            ry=None,
            rz=None,
            open=None,
        ),
        background_tasks=BackgroundTasks(),
        rcm=FakeRCM(robot),
    )

    query = captured_query["query"]

    assert query.x == pytest.approx(21.0)
    assert query.y == pytest.approx(-6.0)
    assert query.z == pytest.approx(34.0)
    assert query.rx is None
    assert query.ry is None
    assert query.rz is None


def test_inverse_kinematics_prefers_current_robot_joint_seed(
    so100: SO100Hardware,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preferred_seed = np.array([0.15, 0.9, -0.4, 0.2, 0.1, -0.2])
    sim_seed = np.array([0.0, 0.2, -0.1, 0.05, -0.05, 0.1])

    so100.set_simulation_positions(sim_seed)
    so100.is_connected = True

    def fake_read_joints_position(
        unit: str = "rad",
        source: str = "robot",
        **kwargs: object,
    ) -> np.ndarray:
        if source == "robot":
            return preferred_seed
        return BaseManipulator.read_joints_position(
            so100,
            unit=unit,
            source=source,
            **kwargs,
        )

    captured_kwargs: dict[str, object] = {}

    def fake_inverse_kinematics(**kwargs: object) -> list[float]:
        captured_kwargs.update(kwargs)
        return [0.0] * len(so100.lower_joint_limits)

    monkeypatch.setattr(so100, "read_joints_position", fake_read_joints_position)
    monkeypatch.setattr(so100.sim, "inverse_kinematics", fake_inverse_kinematics)

    so100.inverse_kinematics(np.array([0.15, 0.0, 0.15]), None)

    rest_poses = np.array(captured_kwargs["rest_poses"], dtype=float)
    assert np.allclose(rest_poses[np.array(so100.actuated_joints)], preferred_seed)
    assert not np.allclose(rest_poses[np.array(so100.actuated_joints)], sim_seed)


def test_piper_end_effector_index_still_points_to_link6(piper: PiperHardware) -> None:
    link_info = piper.sim.get_joint_info(
        robot_id=piper.p_robot_id,
        joint_index=piper.END_EFFECTOR_LINK_INDEX,
    )
    link_name = (
        link_info[12].decode("utf-8")
        if isinstance(link_info[12], (bytes, bytearray))
        else link_info[12]
    )

    assert piper.END_EFFECTOR_LINK_INDEX == 5
    assert link_name == "link6"


def test_piper_forward_inverse_kinematics_round_trip_ready_pose(
    piper: PiperHardware,
) -> None:
    ready_position = np.array(piper.READY_POSITION, dtype=float)
    piper.set_simulation_positions(ready_position)
    piper.sim.step(steps=600)
    time.sleep(0.1)

    position, orientation = piper.forward_kinematics()
    recovered = piper.inverse_kinematics(
        position,
        R.from_euler("xyz", orientation).as_quat(),
    )

    piper.set_simulation_positions(np.array(recovered, dtype=float))
    piper.sim.step(steps=600)
    recovered_position, recovered_orientation = piper.forward_kinematics()

    assert np.allclose(recovered_position, position, atol=1e-4)
    assert np.allclose(recovered_orientation, orientation, atol=1e-4)


@pytest.mark.asyncio
async def test_piper_small_lateral_round_trip_returns_to_start_y(
    piper: PiperHardware,
) -> None:
    ready_position = np.array(piper.READY_POSITION, dtype=float)
    piper.set_simulation_positions(ready_position)
    piper.sim.step(steps=600)
    time.sleep(0.1)

    start_position, start_orientation = piper.forward_kinematics()

    await piper.move_robot_absolute(
        target_position=start_position + np.array([0.0, 0.01, 0.0]),
        target_orientation_rad=start_orientation,
    )
    piper.sim.step(steps=600)
    left_position, _ = piper.forward_kinematics()

    await piper.move_robot_absolute(
        target_position=start_position,
        target_orientation_rad=start_orientation,
    )
    piper.sim.step(steps=600)
    final_position, _ = piper.forward_kinematics()

    assert np.isfinite(left_position).all()
    assert abs(final_position[1] - start_position[1]) < 0.01
