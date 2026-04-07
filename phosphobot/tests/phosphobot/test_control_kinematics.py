import asyncio
import time
from unittest.mock import create_autospec

import numpy as np
import pybullet as p
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

    # FK is called with sync=True: once to resolve the current orientation
    # (when rx/ry/rz are None) and once inside the world-frame move helper.
    assert all(s is True for s in fk_sync_calls)
    assert len(fk_sync_calls) >= 1
    robot.move_robot_absolute.assert_not_called()


@pytest.mark.asyncio
async def test_move_relative_uses_synced_fk_for_controller_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify /move/relative computes target = current + delta (no double offset)."""
    robot = create_autospec(BaseManipulator, instance=True)
    robot.name = "test-manipulator"
    robot.initial_position = np.zeros(3)
    robot.initial_orientation_rad = np.zeros(3)

    synced_position = np.array([0.20, -0.04, 0.31])
    current_orientation = np.array([0.0, 0.0, 0.0])

    fk_sync_calls: list[bool] = []

    def forward_kinematics(sync_robot_pos: bool = False):
        fk_sync_calls.append(sync_robot_pos)
        return synced_position.copy(), current_orientation.copy()

    robot.forward_kinematics.side_effect = forward_kinematics

    captured_targets: list[tuple[np.ndarray, np.ndarray]] = []

    original_helper = control._execute_world_frame_move

    async def fake_execute_world_frame_move(
        robot: object,
        target_position: np.ndarray,
        target_orientation_rad: np.ndarray,
        **kwargs: object,
    ) -> None:
        captured_targets.append((target_position.copy(), target_orientation_rad.copy()))

    monkeypatch.setattr(
        control, "_execute_world_frame_move", fake_execute_world_frame_move
    )

    await control.move_relative(
        data=RelativeEndEffectorPosition(
            x=1,       # 0.01 m
            y=-2,      # -0.02 m
            z=3,       # 0.03 m
            rx=0,
            ry=0,
            rz=0,
            open=None,
        ),
        background_tasks=BackgroundTasks(),
        rcm=FakeRCM(robot),
    )

    # First FK call must be synced
    assert fk_sync_calls[0] is True

    # Target = current + delta, NOT initial + current + delta
    target_pos, _ = captured_targets[0]
    expected_pos = synced_position + np.array([0.01, -0.02, 0.03])
    np.testing.assert_allclose(target_pos, np.round(expected_pos, 3), atol=1e-6)


@pytest.mark.asyncio
async def test_move_relative_without_orientation_delta_holds_current_orientation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When no rotation delta is supplied, target orientation = current orientation."""
    robot = create_autospec(BaseManipulator, instance=True)
    robot.name = "test-manipulator"
    robot.initial_position = np.zeros(3)
    robot.initial_orientation_rad = np.array([0.4, -0.2, 0.1])

    current_position = np.array([0.20, -0.04, 0.31])
    current_orientation = np.array([0.25, -0.15, 0.05])

    def forward_kinematics(sync_robot_pos: bool = False):
        return current_position.copy(), current_orientation.copy()

    robot.forward_kinematics.side_effect = forward_kinematics

    captured_targets: list[tuple[np.ndarray, np.ndarray]] = []

    async def fake_execute_world_frame_move(
        robot: object,
        target_position: np.ndarray,
        target_orientation_rad: np.ndarray,
        **kwargs: object,
    ) -> None:
        captured_targets.append((target_position.copy(), target_orientation_rad.copy()))

    monkeypatch.setattr(
        control, "_execute_world_frame_move", fake_execute_world_frame_move
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

    target_pos, target_orient = captured_targets[0]

    # Position: current + delta
    expected_pos = current_position + np.array([0.01, -0.02, 0.03])
    np.testing.assert_allclose(target_pos, np.round(expected_pos, 3), atol=1e-6)

    # Orientation: held at current (no delta supplied)
    np.testing.assert_allclose(target_orient, np.round(current_orientation, 3), atol=1e-6)


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


@pytest.mark.asyncio
async def test_move_relative_no_double_initial_offset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: /move/relative must NOT add initial_position to the target.

    The old code path routed through /move/absolute which added
    initial_position a second time, producing target = initial + (current + delta)
    instead of the correct target = current + delta.
    """
    robot = create_autospec(BaseManipulator, instance=True)
    robot.name = "test-manipulator"
    # Non-zero initial_position is key: this is what got doubled before the fix
    robot.initial_position = np.array([0.15, -0.05, 0.30])
    robot.initial_orientation_rad = np.array([0.1, 0.0, 0.0])

    current_world_position = np.array([0.20, -0.04, 0.31])
    current_orientation = np.array([0.1, 0.0, 0.0])

    def forward_kinematics(sync_robot_pos: bool = False):
        return current_world_position.copy(), current_orientation.copy()

    robot.forward_kinematics.side_effect = forward_kinematics

    captured_targets: list[np.ndarray] = []

    async def fake_execute(
        robot: object,
        target_position: np.ndarray,
        target_orientation_rad: np.ndarray,
        **kwargs: object,
    ) -> None:
        captured_targets.append(target_position.copy())

    monkeypatch.setattr(control, "_execute_world_frame_move", fake_execute)

    delta_y_cm = -1.0  # -0.01 m

    await control.move_relative(
        data=RelativeEndEffectorPosition(
            x=0, y=delta_y_cm, z=0, rx=None, ry=None, rz=None, open=None
        ),
        background_tasks=BackgroundTasks(),
        rcm=FakeRCM(robot),
    )

    target = captured_targets[0]
    expected = current_world_position + np.array([0.0, -0.01, 0.0])
    np.testing.assert_allclose(target, np.round(expected, 3), atol=1e-6)


def test_forward_kinematics_sync_uses_immediate_reset(
    so100: SO100Hardware,
) -> None:
    """FK sync must use resetJointState (immediate) not setJointMotorControlArray."""
    # Set sim joints to one pose
    zero_pose = np.zeros(len(so100.actuated_joints))
    so100.set_simulation_positions(zero_pose)

    # Record the position at zero
    pos_zero, _ = so100.forward_kinematics(sync_robot_pos=False)

    # Now set a different pose via the immediate sync path
    offset_pose = np.array([0.3, 0.5, -0.2, 0.1, -0.1, 0.2])
    so100.is_connected = True

    # Mock read_joints_position to return the offset pose
    original_read = so100.read_joints_position

    def fake_read(unit: str = "rad", source: str = "robot", **kwargs: object):
        if source == "robot":
            return offset_pose
        return original_read(unit=unit, source=source, **kwargs)

    so100.read_joints_position = fake_read  # type: ignore[assignment]

    # Call FK with sync — should get the offset pose immediately without stepping
    pos_synced, _ = so100.forward_kinematics(sync_robot_pos=True)

    # The synced position should differ from zero (i.e. the joints moved)
    assert not np.allclose(pos_synced, pos_zero, atol=1e-4)


def test_sync_joints_immediate_does_not_drift_after_stepping(
    so100: SO100Hardware,
) -> None:
    """After sync_joints_immediate, background stepping must not pull joints back."""
    offset_pose = np.array([0.3, 0.5, -0.2, 0.1, -0.1, 0.2])
    so100.is_connected = True

    original_read = so100.read_joints_position

    def fake_read(unit: str = "rad", source: str = "robot", **kwargs: object):
        if source == "robot":
            return offset_pose
        return original_read(unit=unit, source=source, **kwargs)

    so100.read_joints_position = fake_read  # type: ignore[assignment]

    # Sync to the offset pose
    pos_synced, _ = so100.forward_kinematics(sync_robot_pos=True)

    # Now run many simulation steps (background stepping uses stepSimulation)
    for _ in range(200):
        p.stepSimulation()

    # Read FK again *without* re-syncing — the pose should stay near the
    # synced position because the motor targets were aligned.
    pos_after_stepping, _ = so100.forward_kinematics(sync_robot_pos=False)

    np.testing.assert_allclose(pos_after_stepping, pos_synced, atol=1e-3)


@pytest.mark.asyncio
async def test_absolute_and_relative_share_motion_lock() -> None:
    """Both /move/absolute and /move/relative must serialize through the same lock."""
    robot = create_autospec(BaseManipulator, instance=True)
    robot.name = "test-manipulator"
    robot.initial_position = np.zeros(3)
    robot.initial_orientation_rad = np.zeros(3)

    call_order: list[str] = []

    def forward_kinematics(sync_robot_pos: bool = False):
        return np.zeros(3), np.zeros(3)

    robot.forward_kinematics.side_effect = forward_kinematics

    rcm = FakeRCM(robot)

    # Retrieve the lock that both endpoints will use
    lock = control._get_robot_motion_lock(robot)

    # Acquire the lock externally so any locked code path will block
    await lock.acquire()

    abs_started = asyncio.Event()
    abs_done = asyncio.Event()
    rel_started = asyncio.Event()
    rel_done = asyncio.Event()

    async def run_absolute() -> None:
        abs_started.set()
        await control.move_to_absolute_position(
            query=MoveAbsoluteRequest(
                x=0, y=0, z=0, rx=0, ry=0, rz=0,
                max_trials=1, position_tolerance=1e-6, orientation_tolerance=1e-6,
            ),
            background_tasks=BackgroundTasks(),
            rcm=rcm,
        )
        call_order.append("absolute")
        abs_done.set()

    async def run_relative() -> None:
        rel_started.set()
        await control.move_relative(
            data=RelativeEndEffectorPosition(
                x=0, y=0, z=0, rx=None, ry=None, rz=None, open=None,
            ),
            background_tasks=BackgroundTasks(),
            rcm=rcm,
        )
        call_order.append("relative")
        rel_done.set()

    task_abs = asyncio.create_task(run_absolute())
    task_rel = asyncio.create_task(run_relative())

    # Wait for both to have started (they'll be blocked on the lock)
    await abs_started.wait()
    await rel_started.wait()

    # Give them a moment to reach the lock
    await asyncio.sleep(0.05)

    # Neither should have completed yet
    assert not abs_done.is_set()
    assert not rel_done.is_set()

    # Release the lock — they should run one at a time
    lock.release()

    await asyncio.gather(task_abs, task_rel)

    # Both completed
    assert abs_done.is_set()
    assert rel_done.is_set()
    # Both ran (order doesn't matter, just that both serialized)
    assert set(call_order) == {"absolute", "relative"}
