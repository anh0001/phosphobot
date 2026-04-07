import asyncio
import json
import traceback
from copy import copy
from typing import List, Optional, cast

import httpx
import json_numpy  # type: ignore
import numpy as np
import serial
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
)
from loguru import logger
from supabase_auth.types import Session as SupabaseSession

from phosphobot.ai_control import CustomAIControlSignal, setup_ai_control
from phosphobot.camera import AllCameras, get_all_cameras
from phosphobot.control_signal import ControlSignal
from phosphobot.hardware.base import BaseManipulator
from phosphobot.leader_follower import RobotPair, start_leader_follower_loop
from phosphobot.models import (
    AIControlStatusResponse,
    AIStatusResponse,
    AppControlData,
    CalibrateResponse,
    EndEffectorPosition,
    EndEffectorReadRequest,
    FeedbackRequest,
    JointsReadRequest,
    JointsReadResponse,
    JointsWriteRequest,
    MoveAbsoluteRequest,
    RelativeEndEffectorPosition,
    RobotConfigResponse,
    RobotConnectionRequest,
    RobotConnectionResponse,
    SpawnStatusResponse,
    StartAIControlRequest,
    StartLeaderArmControlRequest,
    StartServerRequest,
    StatusResponse,
    TeleopSettings,
    TeleopSettingsRequest,
    TemperatureReadResponse,
    TemperatureWriteRequest,
    TorqueControlRequest,
    TorqueReadResponse,
    UDPServerInformationResponse,
    VoltageReadResponse,
)
from phosphobot.robot import (
    RemotePhosphobot,
    RobotConnectionManager,
    SO100Hardware,
    get_rcm,
)
from phosphobot.supabase import get_client, user_is_logged_in
from phosphobot.teleoperation import (
    TeleopManager,
    UDPServer,
    get_teleop_manager,
    get_udp_server,
)
from phosphobot.utils import background_task_log_exceptions, get_tokens

# This is used to send numpy arrays as JSON to OpenVLA server
json_numpy.patch()

router = APIRouter(tags=["control"])


# Object that controls the global /auto, /gravity state in a thread safe way
signal_ai_control = CustomAIControlSignal()
signal_gravity_control = ControlSignal()
signal_leader_follower = ControlSignal()
signal_vr_control = ControlSignal()


def _get_robot_motion_lock(robot: object) -> asyncio.Lock:
    lock = getattr(robot, "_motion_lock", None)
    if lock is None:
        lock = asyncio.Lock()
        setattr(robot, "_motion_lock", lock)
    return cast(asyncio.Lock, lock)


def _read_control_forward_kinematics(
    robot: object, sync: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Read FK from the robot state used for control decisions.

    When *sync* is True (the default), the simulation is first synchronised
    with the real motor positions so the returned pose reflects the physical
    robot – important at the start of a relative move.

    Set *sync=False* for cheap follow-up reads (e.g. residual checks inside a
    retry loop) where an extra hardware round-trip would add latency without
    meaningfully improving accuracy.
    """
    sync_robot_pos = sync and isinstance(robot, BaseManipulator)
    robot_with_fk = cast(BaseManipulator, robot)
    return robot_with_fk.forward_kinematics(sync_robot_pos=sync_robot_pos)


async def _execute_world_frame_move(
    robot: object,
    target_position: np.ndarray,
    target_orientation_rad: np.ndarray,
    position_tolerance: float,
    orientation_tolerance: float,
    max_trials: int,
) -> None:
    """
    Move *robot* to an already-world-frame target pose via IK with retry.

    Both ``target_position`` and ``target_orientation_rad`` must be expressed
    in the pybullet world frame — no controller-relative offset is applied
    here.  This helper is shared by the public ``/move/absolute`` and
    ``/move/relative`` endpoints so the conversion from user coordinates to
    world coordinates only needs to happen once in each caller.
    """
    current_position, current_orientation = _read_control_forward_kinematics(
        robot, sync=True
    )

    position_residual = float(np.linalg.norm(current_position - target_position))
    orientation_residual = float(
        np.linalg.norm(current_orientation - target_orientation_rad)
    )

    num_trials = 0
    while (
        position_residual > position_tolerance
        or orientation_residual > orientation_tolerance
    ) and num_trials <= max_trials - 1:
        if num_trials > 0:
            await asyncio.sleep(0.03 + 0.2 / (num_trials + 1))

        num_trials += 1
        await robot.move_robot_absolute(
            target_position=target_position,
            target_orientation_rad=target_orientation_rad,
        )
        current_position, current_orientation = _read_control_forward_kinematics(
            robot, sync=True
        )
        position_residual = float(np.linalg.norm(current_position - target_position))
        orientation_residual = float(
            np.linalg.norm(current_orientation - target_orientation_rad)
        )


@router.post(
    "/move/init",
    response_model=StatusResponse,
    summary="Initialize Robot",
    description="Initialize the robot to its initial position before starting the teleoperation.",
)
async def move_init(
    robot_id: Optional[int] = None,
    teleop_manager: TeleopManager = Depends(get_teleop_manager),
) -> StatusResponse:
    """
    Initialize the robot to its initial position before starting the teleoperation.
    """
    await teleop_manager.move_init(robot_id=robot_id)
    return StatusResponse()


# HTTP POST endpoint
@router.post(
    "/move/teleop",
    response_model=StatusResponse,
    summary="Teleoperation Control",
)
async def move_teleop_post(
    control_data: AppControlData,
    robot_id: Optional[int] = None,
    teleop_manager: TeleopManager = Depends(get_teleop_manager),
) -> StatusResponse:
    teleop_manager.robot_id = robot_id
    await teleop_manager.process_control_data(control_data)
    return StatusResponse()


# WebSocket endpoint
@router.websocket("/move/teleop/ws")
async def move_teleop_ws(
    websocket: WebSocket,
    rcm: RobotConnectionManager = Depends(get_rcm),
    teleop_manager: TeleopManager = Depends(get_teleop_manager),
) -> None:
    teleop_manager.robot_id = None

    if not await rcm.robots:
        raise HTTPException(status_code=400, detail="No robot connected")

    await websocket.accept()

    signal_vr_control.start()
    try:
        while True:
            data = await websocket.receive_text()
            try:
                control_data = AppControlData.model_validate_json(data)
                await teleop_manager.process_control_data(control_data)
                await teleop_manager.send_status_updates(websocket)
            except json.JSONDecodeError as e:
                logger.error(f"WebSocket JSON error: {e}")

    except WebSocketDisconnect:
        logger.warning("WebSocket client disconnected")

    signal_vr_control.stop()


@router.post("/move/teleop/udp", response_model=UDPServerInformationResponse)
async def move_teleop_udp(
    udp_server: UDPServer = Depends(get_udp_server),
    teleop_manager: TeleopManager = Depends(get_teleop_manager),
) -> UDPServerInformationResponse:
    """
    Start a UDP server to send and receive teleoperation data to the robot.
    """
    teleop_manager.robot_id = None
    udp_server_info = await udp_server.init()
    return udp_server_info


@router.post("/move/teleop/udp/stop", response_model=StatusResponse)
async def stop_teleop_udp(
    udp_server: UDPServer = Depends(get_udp_server),
) -> StatusResponse:
    """
    Stop the UDP server main loop.
    """
    udp_server.stop()
    return StatusResponse()


@router.post(
    "/move/absolute",
    response_model=StatusResponse,
    summary="Move to Absolute Position",
    description="Move the robot to an absolute position specified by the end-effector (in centimeters and degrees). "
    + "Make sure to call `/move/init` before using this endpoint.",
)
async def move_to_absolute_position(
    query: MoveAbsoluteRequest,
    background_tasks: BackgroundTasks,
    robot_id: int = 0,
    rcm: RobotConnectionManager = Depends(get_rcm),
) -> StatusResponse:
    """
    Data: position
    Update the robot position based on the received data
    """
    robot = await rcm.get_robot(robot_id)
    motion_lock = _get_robot_motion_lock(robot)

    async with motion_lock:
        # Divide by 100 to convert from cm to m
        query.x = query.x / 100 if query.x is not None else 0
        query.y = query.y / 100 if query.y is not None else 0
        query.z = query.z / 100 if query.z is not None else 0

        if hasattr(robot, "control_gripper") and query.open is not None:
            # If the robot has a control_gripper method, use it to open/close the gripper
            background_tasks.add_task(
                background_task_log_exceptions(robot.control_gripper),
                open_command=query.open,
            )

        initial_position = getattr(robot, "initial_position", None)
        initial_orientation_rad = getattr(robot, "initial_orientation_rad", None)
        if initial_position is None or initial_orientation_rad is None:
            await robot.move_to_initial_position()
            initial_position = getattr(robot, "initial_position", None)
            initial_orientation_rad = getattr(robot, "initial_orientation_rad", None)
            if initial_position is None or initial_orientation_rad is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Robot {robot.name} .move_to_initial_position() did not set initial position or orientation: {initial_position=}, {initial_orientation_rad=}",
                )

        if hasattr(robot, "forward_kinematics"):
            # Convert controller-relative coordinates to world frame
            target_controller_position = np.array([query.x, query.y, query.z])
            target_position = initial_position + target_controller_position

            # angle
            if query.rx is not None and query.ry is not None and query.rz is not None:
                if robot.name == "so100":
                    # We invert rx and ry
                    target_controller_orientation = np.array([query.ry, query.rx, query.rz])
                else:
                    target_controller_orientation = np.array([query.rx, query.ry, query.rz])

                # Convert from degrees to radians
                target_controller_orientation_rad = np.deg2rad(
                    target_controller_orientation
                )

                target_orientation_rad = (
                    initial_orientation_rad + target_controller_orientation_rad
                )
            else:
                # Maintain the current EEF orientation when no orientation delta
                # is provided.  Without this, the IK solver has unconstrained
                # orientation and will drift the base/wrist joints instead of
                # translating the EEF.
                current_position, current_orientation = _read_control_forward_kinematics(
                    robot, sync=True
                )
                target_orientation_rad = current_orientation

            await _execute_world_frame_move(
                robot=robot,
                target_position=target_position,
                target_orientation_rad=target_orientation_rad,
                position_tolerance=query.position_tolerance,
                orientation_tolerance=query.orientation_tolerance,
                max_trials=query.max_trials,
            )
        else:
            # Otherwise, run the move_robot_absolute method directly
            if query.rx is not None:
                query.rx = np.deg2rad(query.rx)
            if query.ry is not None:
                query.ry = np.deg2rad(query.ry)
            if query.rz is not None:
                query.rz = np.deg2rad(query.rz)
            await robot.move_robot_absolute(
                target_position=np.array([query.x, query.y, query.z]),
                target_orientation_rad=np.array([query.rx, query.ry, query.rz]),
            )

        return StatusResponse()


@router.post(
    "/move/relative",
    response_model=StatusResponse,
    summary="Move to Relative Position",
    description="Move the robot to a relative position based on received delta values (in centimeters and degrees).",
)
async def move_relative(
    data: RelativeEndEffectorPosition,
    background_tasks: BackgroundTasks,
    robot_id: int = 0,
    rcm: RobotConnectionManager = Depends(get_rcm),
) -> StatusResponse:
    """
    Data: The delta sent by OpenVLA for example
    """

    robot = await rcm.get_robot(robot_id)
    motion_lock = _get_robot_motion_lock(robot)

    async with motion_lock:
        # Convert units to meters
        data.x = data.x / 100 if data.x is not None else None
        data.y = data.y / 100 if data.y is not None else None
        data.z = data.z / 100 if data.z is not None else None

        if (
            data.x is None
            and data.y is None
            and data.z is None
            and data.rx is None
            and data.ry is None
            and data.rz is None
            and data.open is not None
        ):
            if hasattr(robot, "control_gripper"):
                # If the robot has a control_gripper method, use it to open/close the gripper
                robot.control_gripper(open_command=data.open)
                return StatusResponse()

        if hasattr(robot, "move_robot_relative"):
            # If the robot has a move_robot_relative method, use it
            target_orientation_rad = np.array(
                [
                    np.deg2rad(u) if u is not None else None
                    for u in [data.rx, data.ry, data.rz]
                ]
            )
            await robot.move_robot_relative(
                target_position=np.array([data.x, data.y, data.z]),
                target_orientation_rad=target_orientation_rad,
            )
            if hasattr(robot, "control_gripper") and data.open is not None:
                # If the robot has a control_gripper method, use it to open/close the gripper
                robot.control_gripper(open_command=data.open)
            return StatusResponse()

        # Fall back to IK-based absolute move if no native move_robot_relative
        if not hasattr(robot, "forward_kinematics"):
            raise HTTPException(
                status_code=400,
                detail="Robot doesn't support move_robot_relative method or forward_kinematics method",
            )

        open_cmd = data.open if data.open is not None else None

        delta_position = np.array([data.x, data.y, data.z])
        has_orientation_delta = any(v is not None for v in (data.rx, data.ry, data.rz))
        delta_orientation_euler_degrees = np.array([data.rx, data.ry, data.rz])

        # Read the current world-frame pose (synced with real motors)
        current_position_raw, current_orientation_raw = _read_control_forward_kinematics(
            robot, sync=True
        )
        piper_diagnostics_enabled = (
            isinstance(robot, BaseManipulator) and getattr(robot, "name", None) == "agilex-piper"
        )
        pre_joint_vector: Optional[np.ndarray] = None
        if piper_diagnostics_enabled:
            try:
                pre_joint_vector = np.array(
                    robot.read_joints_position(unit="rad", source="robot"),
                    dtype=float,
                )
            except Exception as exc:
                logger.debug(f"[move_relative PIPER_DIAG] Failed to read pre joints: {exc}")

        current_position = np.round(current_position_raw.copy(), 3)
        current_orientation = np.round(current_orientation_raw.copy(), 3)

        # Replace None values with 0
        delta_position = np.array([0 if v is None else v for v in delta_position])

        # Compute world-frame target = current + delta (no initial_position added)
        target_position = np.round(current_position + delta_position, 3)

        # Orientation: apply delta if provided, otherwise hold current
        if has_orientation_delta:
            delta_orientation_euler_degrees = np.array(
                [0 if v is None else v for v in delta_orientation_euler_degrees]
            )
            target_orientation_rad = current_orientation + np.deg2rad(
                delta_orientation_euler_degrees
            )
        else:
            target_orientation_rad = current_orientation

        logger.debug(
            f"[move_relative DEBUG] "
            f"delta_pos=[{delta_position[0]:.4f}, {delta_position[1]:.4f}, {delta_position[2]:.4f}] "
            f"current_pos=[{current_position[0]:.4f}, {current_position[1]:.4f}, {current_position[2]:.4f}] "
            f"target_pos=[{target_position[0]:.4f}, {target_position[1]:.4f}, {target_position[2]:.4f}] "
            f"has_orientation_delta={has_orientation_delta}"
        )

        if hasattr(robot, "control_gripper") and open_cmd is not None:
            background_tasks.add_task(
                background_task_log_exceptions(robot.control_gripper),
                open_command=open_cmd,
            )

        # Call the world-frame helper directly — no controller-relative
        # conversion, so the delta is applied exactly once.
        await _execute_world_frame_move(
            robot=robot,
            target_position=target_position,
            target_orientation_rad=target_orientation_rad,
            position_tolerance=1e-3,
            orientation_tolerance=1e-3,
            max_trials=1,
        )

        if piper_diagnostics_enabled:
            post_position_raw: Optional[np.ndarray] = None
            post_joint_vector: Optional[np.ndarray] = None
            try:
                post_position_raw, _ = _read_control_forward_kinematics(robot, sync=True)
            except Exception as exc:
                logger.debug(
                    f"[move_relative PIPER_DIAG] Failed to read post FK pose: {exc}"
                )

            try:
                post_joint_vector = np.array(
                    robot.read_joints_position(unit="rad", source="robot"),
                    dtype=float,
                )
            except Exception as exc:
                logger.debug(
                    f"[move_relative PIPER_DIAG] Failed to read post joints: {exc}"
                )

            if post_position_raw is not None:
                measured_delta_position = post_position_raw - current_position_raw
                logger.debug(
                    "[move_relative PIPER_DIAG] "
                    f"requested_delta_pos=[{delta_position[0]:.4f}, {delta_position[1]:.4f}, {delta_position[2]:.4f}] "
                    f"measured_sync_fk_delta=[{measured_delta_position[0]:.4f}, {measured_delta_position[1]:.4f}, {measured_delta_position[2]:.4f}] "
                    f"pre_sync_pos={np.round(current_position_raw, 4).tolist()} "
                    f"post_sync_pos={np.round(post_position_raw, 4).tolist()} "
                    f"pre_joints={(np.round(pre_joint_vector, 4).tolist() if pre_joint_vector is not None else None)} "
                    f"post_joints={(np.round(post_joint_vector, 4).tolist() if post_joint_vector is not None else None)}"
                )

        return StatusResponse()


@router.post(
    "/move/hello",
    response_model=StatusResponse,
    summary="Make the robot say hello (test endpoint)",
    description="Make the robot say hello by opening and closing its gripper. (Test endpoint)",
)
async def say_hello(
    robot_id: int = 0,
    rcm: RobotConnectionManager = Depends(get_rcm),
) -> StatusResponse:
    """
    Make the robot say hello by opening and closing its gripper.
    """
    robot = await rcm.get_robot(robot_id)

    if not hasattr(robot, "control_gripper"):
        raise HTTPException(
            status_code=400,
            detail="Robot does not support gripper control",
        )

    # Open and close the gripper
    robot.control_gripper(open_command=1)
    await asyncio.sleep(0.5)
    robot.control_gripper(open_command=0.5)
    await asyncio.sleep(0.5)
    robot.control_gripper(open_command=1)
    await asyncio.sleep(0.5)
    robot.control_gripper(open_command=0)

    return StatusResponse()


@router.post(
    "/move/sleep",
    response_model=StatusResponse,
    summary="Put the robot to its sleep position",
    description="Put the robot to its sleep position by giving direct instructions to joints. This function disables the torque.",
)
async def move_sleep(
    robot_id: int = 0,
    rcm: RobotConnectionManager = Depends(get_rcm),
) -> StatusResponse:
    """
    Put the robot to its sleep position.
    """
    robot = await rcm.get_robot(robot_id)
    await robot.move_to_sleep()
    return StatusResponse()


@router.post(
    "/move/home",
    response_model=StatusResponse,
    summary="Move robot to home (zero) position",
    description="Move the arm joints to zero. Gripper is left unchanged.",
)
async def move_home(
    robot_id: int = 0,
    rcm: RobotConnectionManager = Depends(get_rcm),
) -> StatusResponse:
    robot = await rcm.get_robot(robot_id)
    if not isinstance(robot, BaseManipulator):
        raise HTTPException(status_code=400, detail="Robot is not a manipulator")
    await robot.move_to_home_position()
    return StatusResponse()


@router.post(
    "/move/ready",
    response_model=StatusResponse,
    summary="Move robot to ready (operating) position",
    description="Move the arm to the saved or default ready pose.",
)
async def move_ready(
    robot_id: int = 0,
    rcm: RobotConnectionManager = Depends(get_rcm),
) -> StatusResponse:
    robot = await rcm.get_robot(robot_id)
    if not isinstance(robot, BaseManipulator):
        raise HTTPException(status_code=400, detail="Robot is not a manipulator")
    await robot.move_to_ready_position()
    return StatusResponse()


@router.post(
    "/robot/config/ready-pose",
    response_model=StatusResponse,
    summary="Save ready pose",
    description="Save the current arm joint angles (or provided angles) as the robot's ready pose.",
)
async def save_ready_pose(
    robot_id: int = 0,
    angles: Optional[List[float]] = None,
    rcm: RobotConnectionManager = Depends(get_rcm),
) -> StatusResponse:
    """
    If *angles* is omitted, the current arm joint positions are captured.
    The ready pose is persisted to the robot's local config JSON.
    """
    robot = await rcm.get_robot(robot_id)
    if not isinstance(robot, BaseManipulator):
        raise HTTPException(status_code=400, detail="Robot is not a manipulator")

    arm_joint_count = len(robot.SERVO_IDS) - 1
    if angles is None:
        # Read current arm-only joint positions (exclude gripper)
        if not hasattr(robot, "read_joints_position"):
            raise HTTPException(
                status_code=400,
                detail="Robot does not support reading joint positions",
            )
        all_positions = robot.read_joints_position(unit="rad", source="robot")
        if len(all_positions) < arm_joint_count:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Expected at least {arm_joint_count} arm joints when reading "
                    f"robot {robot.name}, got {len(all_positions)}."
                ),
            )
        angles = [float(a) for a in all_positions[:arm_joint_count]]
    elif len(angles) != arm_joint_count:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Ready pose for robot {robot.name} expects {arm_joint_count} arm "
                f"joints, got {len(angles)}."
            ),
        )

    robot.init_config()
    if robot.config is None:
        raise HTTPException(
            status_code=400,
            detail="Robot config not loaded. Calibrate the robot first.",
        )

    robot.config.ready_pose_rad = angles
    robot.config.save_local(serial_id=robot.SERIAL_ID)
    return StatusResponse()


@router.post(
    "/end-effector/read",
    response_model=EndEffectorPosition,
    summary="Read End-Effector Position",
    description="Retrieve the position, orientation, and open status of the robot's end effector. Only available for manipulators.",
)
async def end_effector_read(
    query: Optional[EndEffectorReadRequest] = None,
    robot_id: int = 0,
    rcm: RobotConnectionManager = Depends(get_rcm),
) -> EndEffectorPosition:
    """
    Get the position, orientation, and open status of the end effector.
    """
    robot = await rcm.get_robot(robot_id)
    if query is None:
        query = EndEffectorReadRequest(only_gripper=False, sync=False)

    if not isinstance(robot, BaseManipulator):
        raise HTTPException(
            status_code=400,
            detail="Robot is not a manipulator and does not have an end effector",
        )

    if query.only_gripper:
        if not hasattr(robot, "closing_gripper_value"):
            raise HTTPException(
                status_code=400,
                detail=f"Robot {robot.name} does not have a .closing_gripper_value attribute to read the gripper state.",
            )
        return EndEffectorPosition(
            x=None,
            y=None,
            z=None,
            rx=None,
            ry=None,
            rz=None,
            open=robot.closing_gripper_value,
        )

    initial_position = getattr(robot, "initial_position", None)
    initial_orientation_rad = getattr(robot, "initial_orientation_rad", None)
    if initial_position is None or initial_orientation_rad is None:
        raise HTTPException(
            status_code=400,
            detail=f"Before using /end-effector/read you need to call /move/init?robot_id={robot_id} to initialize the robot's position and orientation.",
        )

    position, orientation, open_status = robot.get_end_effector_state(sync=query.sync)
    # Remove the initial position and orientation (used to zero the robot)
    position = position - initial_position
    orientation = orientation - initial_orientation_rad

    x, y, z = position
    rx, ry, rz = orientation

    # Convert position to centimeters
    x *= 100

    # Convert to degrees
    rx = np.rad2deg(rx)
    ry = np.rad2deg(ry)
    rz = np.rad2deg(rz)

    return EndEffectorPosition(x=x, y=y, z=z, rx=rx, ry=ry, rz=rz, open=open_status)


@router.post(
    "/voltage/read",
    response_model=VoltageReadResponse,
    summary="Read Voltage",
    description="Read the current voltage of the robot's motors.",
)
async def read_voltage(
    robot_id: int = 0,
    rcm: RobotConnectionManager = Depends(get_rcm),
) -> VoltageReadResponse:
    """
    Read voltage of the robot.
    """
    robot = await rcm.get_robot(robot_id)
    if not hasattr(robot, "current_voltage"):
        raise HTTPException(
            status_code=400,
            detail="Robot does not support reading voltage",
        )

    voltage = robot.current_voltage()
    return VoltageReadResponse(
        current_voltage=voltage.tolist() if voltage is not None else None,
    )


@router.post(
    "/temperature/read",
    response_model=TemperatureReadResponse,
    summary="Read Temperature",
    description="Read the current Temperature and maximum Temperature of the robot's motors.",
)
async def read_temperature(
    robot_id: int = 0,
    rcm: RobotConnectionManager = Depends(get_rcm),
) -> TemperatureReadResponse:
    """
    Read temperature of the robot.
    """
    robot = await rcm.get_robot(robot_id)
    if not hasattr(robot, "current_temperature"):
        raise HTTPException(
            status_code=400,
            detail="Robot does not support reading temperature",
        )

    temperature = robot.current_temperature()

    return TemperatureReadResponse(
        current_max_Temperature=temperature,
    )


@router.post(
    "/temperature/write",
    response_model=StatusResponse,
    summary="Write the Maximum Temperature for Joints",
    description="Set the robot's maximum temperature for motors..",
)
async def write_temperature(
    request: TemperatureWriteRequest,
    robot_id: int = 0,
    rcm: RobotConnectionManager = Depends(get_rcm),
) -> StatusResponse:
    """
    Set the robot's maximum temperature for motors.
    """
    robot = await rcm.get_robot(robot_id)
    if not hasattr(robot, "set_maximum_temperature"):
        raise HTTPException(
            status_code=400,
            detail="Robot does not support setting motor temperature",
        )
    robot.set_maximum_temperature(
        maximum_temperature_target=request.maximum_temperature
    )
    return StatusResponse()


@router.post(
    "/torque/read",
    response_model=TorqueReadResponse,
    summary="Read Torque",
    description="Read the current torque of the robot's joints.",
)
async def read_torque(
    robot_id: int = 0,
    rcm: RobotConnectionManager = Depends(get_rcm),
) -> TorqueReadResponse:
    """
    Read torque of the robot.
    """
    robot = await rcm.get_robot(robot_id)

    if not hasattr(robot, "current_torque"):
        raise HTTPException(
            status_code=400,
            detail="Robot does not support reading torque",
        )

    current_torque = robot.current_torque()

    # Replace NaN values with None and convert to list
    current_torque = [
        float(torque) if not np.isnan(torque) else 0 for torque in current_torque
    ]

    return TorqueReadResponse(current_torque=current_torque)


@router.post(
    "/torque/toggle",
    response_model=StatusResponse,
    summary="Toggle Torque",
    description="Enable or disable the torque of the robot.",
)
async def toggle_torque(
    request: TorqueControlRequest,
    robot_id: Optional[int] = None,
    rcm: RobotConnectionManager = Depends(get_rcm),
) -> StatusResponse:
    """
    Enable or disable the torque of the robot.
    """

    if robot_id is not None:
        robot = await rcm.get_robot(robot_id)

        if not hasattr(robot, "enable_torque") or not hasattr(robot, "disable_torque"):
            raise HTTPException(
                status_code=400,
                detail="Robot does not support torque control",
            )

        if request.torque_status:
            robot.enable_torque()
        else:
            robot.disable_torque()
        return StatusResponse()

    # If no robot_id is provided, toggle torque for all robots
    for robot in await rcm.robots:
        if not hasattr(robot, "enable_torque") or not hasattr(robot, "disable_torque"):
            logger.warning(
                f"Robot {robot.name} does not support torque control. Skipping."
            )
            continue
        else:
            if request.torque_status:
                robot.enable_torque()
            else:
                robot.disable_torque()

    return StatusResponse()


@router.post(
    "/joints/read",
    response_model=JointsReadResponse,
    summary="Read Joint Positions",
    description="Read the current positions of the robot's joints in radians and motor units.",
)
async def read_joints(
    request: Optional[JointsReadRequest] = None,
    robot_id: int = 0,
    rcm: RobotConnectionManager = Depends(get_rcm),
) -> JointsReadResponse:
    """
    Read joint position.
    """
    if request is None:
        request = JointsReadRequest(unit="rad", joints_ids=None, source="robot")

    robot = await rcm.get_robot(robot_id)

    if not hasattr(robot, "read_joints_position"):
        raise HTTPException(
            status_code=400,
            detail="Robot does not support reading joint positions",
        )

    current_units_position = robot.read_joints_position(
        unit=request.unit, joints_ids=request.joints_ids, source=request.source
    )
    # Replace NaN values with None and convert to list
    current_units_position = [
        float(angle) if not np.isnan(angle) else None
        for angle in current_units_position
    ]

    return JointsReadResponse(
        angles=current_units_position,
        unit=request.unit,
    )


@router.post(
    "/joints/write",
    response_model=StatusResponse,
    summary="Write Joint Positions",
    description="Move the robot's joints to the specified angles.",
)
async def write_joints(
    request: JointsWriteRequest,
    robot_id: int = 0,
    rcm: RobotConnectionManager = Depends(get_rcm),
) -> StatusResponse:
    """
    Move the robot's joints to the specified angles.
    """
    robot = await rcm.get_robot(robot_id)
    if not hasattr(robot, "write_joint_positions"):
        raise HTTPException(
            status_code=400,
            detail="Robot does not support writing joint positions",
        )

    robot = cast(BaseManipulator, robot)
    robot.write_joint_positions(
        angles=request.angles, unit=request.unit, joints_ids=request.joints_ids
    )

    return StatusResponse()


@router.post(
    "/calibrate",
    response_model=CalibrateResponse,
    summary="Calibrate Robot",
    description="Start the calibration sequence for the robot.",
)
async def calibrate(
    robot_id: int = 0,
    rcm: RobotConnectionManager = Depends(get_rcm),
) -> CalibrateResponse:
    """
    Starts the calibration sequence of the robot.

    This endpoints disable torque. Move the robot to the positions you see in the simulator and call this endpoint again, until the calibration is complete.
    """
    robot = await rcm.get_robot(robot_id)

    if (
        not hasattr(robot, "calibrate")
        or not hasattr(robot, "calibration_current_step")
        or not hasattr(robot, "calibration_max_steps")
    ):
        raise HTTPException(
            status_code=400,
            detail="Robot does not support calibration",
        )

    if not robot.is_connected:
        raise HTTPException(status_code=400, detail="Robot is not connected")

    try:
        status, message = await robot.calibrate()
        current_step = robot.calibration_current_step
        total_nb_steps = robot.calibration_max_steps
        if status == "success":
            current_step = total_nb_steps
    except Exception as e:
        status = "error"
        current_step = getattr(robot, "calibration_current_step", 0)
        total_nb_steps = getattr(robot, "calibration_max_steps", 0)
        message = f"Calibration step {current_step}/{total_nb_steps} failed: {e}"

    return CalibrateResponse(
        calibration_status=status,
        message=message,
        current_step=current_step,
        total_nb_steps=total_nb_steps,
    )


@router.post(
    "/move/leader/start",
    response_model=StatusResponse,
    summary="Use the leader arm to control the follower arm",
    description="Use the leader arm to control the follower arm.",
)
async def start_leader_follower(
    request: StartLeaderArmControlRequest,
    background_tasks: BackgroundTasks,
    rcm: RobotConnectionManager = Depends(get_rcm),
) -> StatusResponse:
    """
    Endpoint to start the leader-follower control.
    The first robot is the leader with gravity compensation enabled,
    and the second robot follows the leader's joint positions.
    """
    if signal_leader_follower.is_in_loop():
        raise HTTPException(
            status_code=400,
            detail="Leader-follower control is already running. To stop it, call /move/leader/stop",
        )

    # Parse the robot IDs from the request
    robot_pairs: list[RobotPair] = []
    for i, robot_pair in enumerate(request.robot_pairs):
        if robot_pair.follower_id is None:
            raise HTTPException(
                status_code=400,
                detail=f"Follower ID is required for robot pair {i}.",
            )
        if robot_pair.leader_id is None:
            raise HTTPException(
                status_code=400,
                detail=f"Leader ID is required for robot pair {i}.",
            )
        leader = await rcm.get_robot(robot_pair.leader_id)
        follower = await rcm.get_robot(robot_pair.follower_id)

        if request.enable_gravity_compensation:
            # Only local SO100
            if not isinstance(leader, SO100Hardware):
                raise HTTPException(
                    status_code=400,
                    detail=f"Leader must be an instance of SO100Hardware for robot pair {i}.",
                )
            if not isinstance(follower, SO100Hardware):
                raise HTTPException(
                    status_code=400,
                    detail=f"Follower must be an instance of SO100Hardware for robot pair {i}.",
                )
        else:
            valid_robot_types = (BaseManipulator, RemotePhosphobot)
            if not isinstance(leader, valid_robot_types):
                raise HTTPException(
                    status_code=400,
                    detail=f"Leader must be an instance of {valid_robot_types} for robot pair {i}.",
                )
            if not isinstance(follower, valid_robot_types):
                raise HTTPException(
                    status_code=400,
                    detail=f"Follower must be an instance of {valid_robot_types} for robot pair {i}.",
                )

        # TODO: Eventually add more config options individual for each pair
        robot_pairs.append(RobotPair(leader=leader, follower=follower))

    # Create control signal for managing the leader-follower operation
    signal_leader_follower.start()

    # Add background task to run the control loop
    background_tasks.add_task(
        start_leader_follower_loop,
        robot_pairs=robot_pairs,
        control_signal=signal_leader_follower,
        invert_controls=request.invert_controls,
        enable_gravity_compensation=request.enable_gravity_compensation,
        compensation_values=request.gravity_compensation_values,
    )

    return StatusResponse(message="Leader-follower control started")


@router.post(
    "/move/leader/stop",
    response_model=StatusResponse,
    summary="Stop the leader-follower control",
    description="Stop the leader-follower control.",
)
async def stop_leader_follower(
    rcm: RobotConnectionManager = Depends(get_rcm),
) -> StatusResponse:
    """
    Stop the leader-follower
    """
    if not signal_leader_follower.is_in_loop():
        return StatusResponse(
            status="error", message="Leader-follower control is not running"
        )

    signal_leader_follower.stop()
    return StatusResponse(message="Stopping leader-follower control")


@router.post("/gravity/start", response_model=StatusResponse)
async def start_gravity(
    background_tasks: BackgroundTasks,
    robot_id: int = 0,
    rcm: RobotConnectionManager = Depends(get_rcm),
) -> StatusResponse:
    """
    Enable gravity compensation for the robot.
    """
    if signal_gravity_control.is_in_loop():
        raise HTTPException(
            status_code=400, detail="Gravity control is already running"
        )

    if len(await rcm.robots) == 0:
        raise HTTPException(status_code=400, detail="No robot connected")

    robot = await rcm.get_robot(robot_id)
    if not isinstance(robot, SO100Hardware):
        raise HTTPException(
            status_code=400, detail="Gravity compensation is only for SO-100 robot"
        )

    signal_gravity_control.start()

    # Add background task to run the control loop
    background_tasks.add_task(
        background_task_log_exceptions(robot.gravity_compensation_loop),
        control_signal=signal_gravity_control,
    )
    return StatusResponse()


@router.post(
    "/gravity/stop",
    response_model=StatusResponse,
    summary="Stop the gravity compensation",
    description="Stop the gravity compensation.",
)
async def stop_gravity_compensation(
    rcm: RobotConnectionManager = Depends(get_rcm),
) -> StatusResponse:
    """
    Stop the gravity compensation for all robots.
    """
    if not signal_gravity_control.is_in_loop():
        return StatusResponse(status="error", message="Gravity control is not running")

    signal_gravity_control.stop()
    return StatusResponse(message="Stopping gravity control")


@router.post(
    "/ai-control/status",
    response_model=AIStatusResponse,
    summary="Get the status of the auto control by AI",
    description="Get the status of the auto control by AI.",
)
async def fetch_auto_control_status() -> AIStatusResponse:
    """
    Fetch the status of the auto control by AI
    """
    return AIStatusResponse(id=signal_ai_control.id, status=signal_ai_control.status)


@router.post(
    "/ai-control/spawn",
    response_model=SpawnStatusResponse,
    summary="Start an inference server",
    description="Start an inference server and return the server info.",
)
async def spawn_inference_server(
    query: StartServerRequest,
    rcm: RobotConnectionManager = Depends(get_rcm),
    all_cameras: AllCameras = Depends(get_all_cameras),
    session: SupabaseSession = Depends(user_is_logged_in),
) -> SpawnStatusResponse:
    """
    Start an inference server and return the server info.
    """

    supabase_client = await get_client()
    await supabase_client.auth.get_user()

    robots_to_control = copy(await rcm.robots)
    for robot in await rcm.robots:
        if (
            hasattr(robot, "SERIAL_ID")
            and query.robot_serials_to_ignore is not None
            and robot.SERIAL_ID in query.robot_serials_to_ignore
        ):
            robots_to_control.remove(robot)
        if not isinstance(robot, BaseManipulator):
            logger.warning(
                f"Robot {robot.name} is not a manipulator and is not supported for AI control. Skipping."
            )
            robots_to_control.remove(robot)

    assert all(isinstance(robot, BaseManipulator) for robot in robots_to_control), (
        "All robots must be manipulators for AI control"
    )

    # Get the modal host and port here
    _, _, server_info = await setup_ai_control(
        robots=robots_to_control,  # type: ignore
        all_cameras=all_cameras,
        model_id=query.model_id,
        init_connected_robots=False,
        model_type=query.model_type,
        ai_control_signal_id=signal_ai_control.id,
    )

    return SpawnStatusResponse(message="ok", server_info=server_info)


@router.post(
    "/ai-control/start",
    response_model=AIControlStatusResponse,
    summary="Start the auto control by AI",
    description="Start the auto control by AI.",
)
async def start_ai_control(
    query: StartAIControlRequest,
    background_tasks: BackgroundTasks,
    rcm: RobotConnectionManager = Depends(get_rcm),
    all_cameras: AllCameras = Depends(get_all_cameras),
    session: SupabaseSession = Depends(user_is_logged_in),
) -> AIControlStatusResponse:
    """
    Start the auto control by AI
    """

    if signal_leader_follower.is_in_loop():
        raise HTTPException(
            status_code=400,
            detail="Leader-follower control is running. Stop it before starting AI control.",
        )
    if signal_gravity_control.is_in_loop():
        raise HTTPException(
            status_code=400,
            detail="Gravity compensation is running. Stop it before starting AI control.",
        )

    if signal_ai_control.is_in_loop():
        return AIControlStatusResponse(
            status="error",
            message="Auto control is already running",
            ai_control_signal_id=signal_ai_control.id,
            ai_control_signal_status=signal_ai_control.status,
            server_info=None,
        )

    signal_ai_control.new_id()
    signal_ai_control.start()

    supabase_client = await get_client()
    user = await supabase_client.auth.get_user()
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    await (
        supabase_client.table("ai_control_sessions")
        .upsert(
            {
                "id": signal_ai_control.id,
                "user_id": user.user.id,
                "user_email": user.user.email,
                "model_type": query.model_type,
                "model_id": query.model_id,
                "prompt": query.prompt,
                "checkpoint": query.checkpoint,
                "status": "waiting",
            }
        )
        .execute()
    )

    robots_to_control = copy(await rcm.robots)
    for robot in await rcm.robots:
        if (
            hasattr(robot, "SERIAL_ID")
            and query.robot_serials_to_ignore is not None
            and robot.SERIAL_ID in query.robot_serials_to_ignore
        ):
            robots_to_control.remove(robot)
        if not isinstance(robot, BaseManipulator):
            logger.warning(
                f"Robot {robot.name} is not a manipulator and is not supported for AI control. Skipping."
            )
            robots_to_control.remove(robot)

    assert all(isinstance(robot, BaseManipulator) for robot in robots_to_control), (
        "All robots must be manipulators for AI control"
    )

    # Get the modal host and port here
    model, model_spawn_config, server_info = await setup_ai_control(
        robots=robots_to_control,  # type: ignore
        all_cameras=all_cameras,
        model_type=query.model_type,
        model_id=query.model_id,
        cameras_keys_mapping=query.cameras_keys_mapping,
        ai_control_signal_id=signal_ai_control.id,
        verify_cameras=query.verify_cameras,
        checkpoint=query.checkpoint,
    )

    # Add a flag: successful setup
    await (
        supabase_client.table("ai_control_sessions")
        .update(
            {
                "setup_success": True,
                "server_id": server_info.server_id,
            }
        )
        .eq("id", signal_ai_control.id)
        .execute()
    )

    background_tasks.add_task(
        model.control_loop,
        robots=robots_to_control,
        control_signal=signal_ai_control,
        prompt=query.prompt,
        all_cameras=all_cameras,
        model_spawn_config=model_spawn_config,
        speed=query.speed,
        cameras_keys_mapping=query.cameras_keys_mapping,
        detect_instruction=query.prompt,
        selected_camera_id=query.selected_camera_id,
        angle_format=query.angle_format,
        min_angle=query.min_angle,
        max_angle=query.max_angle,
    )

    return AIControlStatusResponse(
        status="ok",
        message=f"Starting AI control with id: {signal_ai_control.id}",
        server_info=server_info,
        ai_control_signal_id=signal_ai_control.id,
        ai_control_signal_status="waiting",
    )


@router.post(
    "/ai-control/stop",
    response_model=StatusResponse,
    summary="Stop the auto control by AI",
    description="Stop the auto control by AI.",
)
async def stop_ai_control(
    background_tasks: BackgroundTasks,
    rcm: RobotConnectionManager = Depends(get_rcm),
    session: SupabaseSession = Depends(user_is_logged_in),
) -> StatusResponse:
    """
    Stop the auto control by AI
    """

    tokens = get_tokens()

    # Call the /stop endpoint in Modal
    @background_task_log_exceptions
    async def stop_modal() -> None:
        async with httpx.AsyncClient(timeout=5) as client:
            await client.post(
                url=f"{tokens.MODAL_API_URL}/stop",
                headers={
                    "Authorization": f"Bearer {session.access_token}",
                    "Content-Type": "application/json",
                },
            )

    background_tasks.add_task(stop_modal)

    if not signal_ai_control.is_in_loop():
        return StatusResponse(message="Auto control is not running")

    signal_ai_control.stop()

    return StatusResponse(message="Stopped AI control")


@router.post(
    "/ai-control/pause",
    response_model=StatusResponse,
    summary="Pause the auto control by AI",
    description="Pause the auto control by AI.",
)
async def pause_ai_control(
    rcm: RobotConnectionManager = Depends(get_rcm),
) -> StatusResponse:
    """
    Pause the auto control by AI
    """
    if not signal_ai_control.is_in_loop():
        return StatusResponse(message="Auto control is not running")

    signal_ai_control.status = "paused"
    return StatusResponse(message="Pausing AI control")


@router.post(
    "/ai-control/resume",
    response_model=StatusResponse,
    summary="Resume the auto control by AI",
    description="Resume the auto control by AI.",
)
async def resume_ai_control(
    rcm: RobotConnectionManager = Depends(get_rcm),
) -> StatusResponse:
    """
    Resume the auto control by AI
    """
    if signal_ai_control.status == "running":
        return StatusResponse(message="AI control is already running")

    signal_ai_control.status = "running"
    return StatusResponse(message="Resuming AI control")


@router.post(
    "/ai-control/feedback",
    response_model=StatusResponse,
    summary="Feedback about the AI control session",
)
async def feedback_ai_control(
    request: FeedbackRequest,
    session: SupabaseSession = Depends(user_is_logged_in),
) -> StatusResponse:
    supabase_client = await get_client()

    await (
        supabase_client.table("ai_control_sessions")
        .update({"feedback": request.feedback})
        .eq("id", request.ai_control_id)
        .execute()
    )

    return StatusResponse(message="Feedback sent")


@router.post("/robot/add-connection", response_model=RobotConnectionResponse)
async def add_robot_connection(
    query: RobotConnectionRequest,
    rcm: RobotConnectionManager = Depends(get_rcm),
) -> RobotConnectionResponse:
    """
    Manually add a robot connection to the robot manager.
    Useful for adding robot that are accessible only via WiFi, for example.
    """
    try:
        robot_id, robot = await rcm.add_connection(
            robot_name=query.robot_name,
            connection_details=query.connection_details,
        )
        return RobotConnectionResponse(
            status="ok",
            message=f"Robot connection to {query.robot_name} added",
            robot_id=robot_id,
        )
    except serial.SerialException as e:
        if "Access is denied" in str(e):
            logger.warning(
                f"Failed to add robot connection: {e}\n{traceback.format_exc()}"
            )
            raise HTTPException(
                status_code=403,
                detail=f"Permission error: {e}. If you're on Windows, try running with WSL (Windows Subsystem for Linux) or phosphobot has the authorization to use the serial port.",
            )
        elif "Permission denied" in str(e):
            logger.warning(
                f"Failed to add robot connection: {e}\n{traceback.format_exc()}"
            )
            raise HTTPException(
                status_code=403,
                detail=f"Permission denied: {e}. If you're on Linux, try running phosphobot as sudo (`sudo phosphobot`) and ensure that your user has access to the serial port.",
            )
        else:
            logger.error(
                f"Failed to add robot connection (SerialException): {e}\n{traceback.format_exc()}"
            )
            raise HTTPException(
                status_code=400, detail=f"Failed to add robot connection: {e}"
            )
    except Exception as e:
        logger.error(f"Failed to add robot connection: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=400, detail=f"Failed to add robot connection: {e}"
        )


@router.post("/robot/disconnect", response_model=StatusResponse)
async def disconnect_robot(
    robot_id: int = 0,
    rcm: RobotConnectionManager = Depends(get_rcm),
) -> StatusResponse:
    """
    Manually add a robot connection to the robot manager.
    Useful for adding robot that are accessible only via WiFi, for example.
    """
    try:
        await rcm.remove_connection(robot_id=robot_id)
        return StatusResponse(
            status="ok",
            message=f"Robot connection to {robot_id} removed successfully",
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(
            f"Failed to remove robot connection {robot_id}: {e}\n{traceback.format_exc()}"
        )
        raise HTTPException(
            status_code=400, detail=f"Failed to remove robot connection {robot_id}: {e}"
        )


@router.post("/robot/config", response_model=RobotConfigResponse)
async def get_robot_config(
    robot_id: int = 0,
    rcm: RobotConnectionManager = Depends(get_rcm),
) -> RobotConfigResponse:
    """
    Get the configuration of the robot.
    """
    robot = await rcm.get_robot(robot_id)

    if isinstance(robot, BaseManipulator) or isinstance(robot, RemotePhosphobot):
        config = robot.config
        return RobotConfigResponse(
            robot_id=robot_id,
            name=robot.name,
            config=config,
            gripper_joint_index=robot.GRIPPER_JOINT_INDEX,
            servo_ids=robot.SERVO_IDS,
            resolution=robot.RESOLUTION,
        )

    raise HTTPException(
        status_code=400,
        detail=f"Robot {robot.name} does not support configuration retrieval.",
    )


@router.post(
    "/teleop/settings/read",
    response_model=TeleopSettings,
    summary="Read Teleop Settings",
    description="Get current teleoperation settings.",
)
async def read_teleop_settings(
    teleop_manager: TeleopManager = Depends(get_teleop_manager),
) -> TeleopSettings:
    """
    Get current teleoperation settings.
    """
    return teleop_manager.settings


@router.post(
    "/teleop/settings",
    response_model=StatusResponse,
    summary="Update Teleop Settings",
    description="Update teleoperation settings such as VR scaling factor.",
)
async def update_teleop_settings(
    settings: TeleopSettingsRequest,
    teleop_manager: TeleopManager = Depends(get_teleop_manager),
) -> StatusResponse:
    """
    Update teleoperation settings.
    """
    settings_dict = settings.model_dump()

    for attr_name, value in settings_dict.items():
        if hasattr(teleop_manager, attr_name):
            setattr(teleop_manager, attr_name, value)
        else:
            logger.warning(
                f"Attempted to set non-existent attribute '{attr_name}' on TeleopManager"
            )

    return StatusResponse()
