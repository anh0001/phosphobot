import hashlib
import asyncio
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any, List, Literal, Optional, Union

import numpy as np
import pybullet as p
from loguru import logger
from piper_sdk import C_PiperInterface_V2

from phosphobot.hardware.base import BaseManipulator
from phosphobot.models import BaseRobotConfig
from phosphobot.models.robot import RobotConfigStatus
from phosphobot.utils import get_resources_path, is_running_on_linux


class PiperHardware(BaseManipulator):
    name = "agilex-piper"
    device_name = "agilex-piper"

    # Default to the official new-firmware model (>= S-V1.6-3) and keep a
    # vendored copy of AgileX's old-firmware URDF for firmware < S-V1.6-3.
    DEFAULT_URDF_FILE_PATH = str(
        get_resources_path() / "urdf" / "piper" / "urdf" / "piper.urdf"
    )
    LEGACY_URDF_FILE_PATH = str(
        get_resources_path() / "urdf" / "piper" / "urdf" / "piper_old.urdf"
    )
    URDF_FILE_PATH = DEFAULT_URDF_FILE_PATH
    URDF_VARIANT_ENV_VAR = "PHOSPHOBOT_PIPER_URDF_VARIANT"
    DH_OFFSET_ENV_VAR = "PHOSPHOBOT_PIPER_DH_IS_OFFSET"

    AXIS_ORIENTATION = [0, 0, 0, 1]

    END_EFFECTOR_LINK_INDEX = 5
    GRIPPER_JOINT_INDEX = 6
    # Prefer link6 because it exists in both official old/new URDF variants and
    # gives us a consistent link frame across firmware generations. The newer
    # URDF also exposes a fixed tcp link, but that is a fixed descendant of
    # link6 rather than a distinct kinematic branch.
    END_EFFECTOR_LINK_CANDIDATES = ("link6", "tcp", "piper_tcp")
    EXPECTED_DEFAULT_URDF_JOINT_NAMES = (
        "joint6",
        "joint6_to_gripper_base",
        "joint6_to_tcp",
        "joint7",
        "joint8",
    )
    EXPECTED_LEGACY_URDF_JOINT_NAMES = (
        "joint6",
        "joint6_to_gripper_base",
        "joint7",
        "joint8",
    )
    LEGACY_FIRMWARE_CUTOFF = (1, 6, 3)

    SERVO_IDS = [1, 2, 3, 4, 5, 6, 7]

    RESOLUTION = 360 * 1000  # In 0.001 degree

    SLEEP_POSITION = [0, 0, 0, 0, 0, 0]
    time_to_sleep: float = 1.8
    CALIBRATION_POSITION = [0, 0, 0, 0, 0, 0]
    # Default ready (operating) pose for the arm joints (radians, no gripper).
    READY_POSITION = [0.0, 1.2, -0.2, 0.0, -0.8, 0.0]
    pose_command_repetitions: int = 4
    pose_command_interval_s: float = 0.05

    is_object_gripped = False
    is_moving = False
    robot_connected = False

    GRIPPER_MAX_ANGLE = 99  # In degree
    ENABLE_GRIPPER = 0x01
    DISABLE_GRIPPER = 0x00

    GRIPPER_SERVO_ID = 7
    # When using the set zero of gripper control, we observe that current position is set to -1800 and not to zero
    GRIPPER_ZERO_POSITION = -1800
    # Strength with which the gripper will close. Similar to the gripping threshold value of other robots,
    GRIPPER_EFFORT = 600

    calibration_max_steps: int = 2

    # Reference: https://github.com/agilexrobotics/piper_sdk/blob/master/asserts/V2/INTERFACE_V2.MD#jointctrl
    #  |joint_name|     limit(rad)     |    limit(angle)    |
    # |----------|     ----------     |     ----------     |
    # |joint1    |   [-2.6179, 2.6179]  |    [-150.0, 150.0] |
    # |joint2    |   [0, 3.14]        |    [0, 180.0]      |
    # |joint3    |   [-2.967, 0]      |    [-170, 0]       |
    # |joint4    |   [-1.745, 1.745]  |    [-100.0, 100.0] |
    # |joint5    |   [-1.22, 1.22]    |    [-70.0, 70.0]   |
    # |joint6    |   [-2.09439, 2.09439]|    [-120.0, 120.0] |
    piper_limits_rad: dict = {
        1: {"min_angle_limit": -2.6179, "max_angle_limit": 2.6179},
        2: {"min_angle_limit": 0, "max_angle_limit": 3.14},
        3: {"min_angle_limit": -2.967, "max_angle_limit": 0},
        4: {"min_angle_limit": -1.745, "max_angle_limit": 1.745},
        5: {"min_angle_limit": -1.047, "max_angle_limit": 1.047},
        6: {"min_angle_limit": -2.09439, "max_angle_limit": 2.0943},
    }
    piper_limits_degrees: dict = {
        1: {"min_angle_limit": -150.0, "max_angle_limit": 150.0},
        2: {"min_angle_limit": 0, "max_angle_limit": 180.0},
        3: {"min_angle_limit": -170, "max_angle_limit": 0},
        4: {"min_angle_limit": -100.0, "max_angle_limit": 100.0},
        5: {"min_angle_limit": -60.0, "max_angle_limit": 60.0},
        6: {"min_angle_limit": -120.0, "max_angle_limit": 120.0},
    }

    def __init__(
        self,
        can_name: str = "can0",
        only_simulation: bool = False,
        axis: Optional[List[float]] = None,
    ) -> None:
        self.can_name = can_name
        self.URDF_FILE_PATH = self._resolve_urdf_file_path()
        super().__init__(
            only_simulation=only_simulation, axis=axis, enable_self_collision=True
        )
        self.SERIAL_ID = can_name
        self.is_torqued = False
        # Track the last commanded move mode, but do not assume the hardware
        # actually stayed there across resets/reconnects.
        self._current_move_mode = -1
        self._resolved_end_effector_link_name = "link6"
        self.END_EFFECTOR_LINK_INDEX = self._resolve_end_effector_link_index()
        self._init_sim_gripper()
        self._log_runtime_model_diagnostics()

    @classmethod
    def _resolve_urdf_file_path(cls, firmware_version: Optional[str] = None) -> str:
        variant = os.getenv(cls.URDF_VARIANT_ENV_VAR, "").strip().lower()

        if variant in {"legacy", "piper_old", "piper_legacy_local", "local"}:
            logger.warning(
                f"Using legacy Piper URDF from {cls.LEGACY_URDF_FILE_PATH} because "
                f"{cls.URDF_VARIANT_ENV_VAR}={variant!r}."
            )
            return cls.LEGACY_URDF_FILE_PATH

        if variant in {"default", "agilex", "new", "piper"}:
            return cls.DEFAULT_URDF_FILE_PATH

        if variant not in {"", "auto"}:
            logger.warning(
                f"Unknown {cls.URDF_VARIANT_ENV_VAR}={variant!r}; falling back to "
                f"auto-selection."
            )

        # Auto-select based on firmware version when no explicit override is set
        if firmware_version is not None:
            parsed = cls._parse_firmware_version(firmware_version)
            if parsed is not None:
                if parsed < cls.LEGACY_FIRMWARE_CUTOFF:
                    logger.info(
                        f"Auto-selected legacy URDF for firmware '{firmware_version}' "
                        f"(< S-V{cls.LEGACY_FIRMWARE_CUTOFF[0]}.{cls.LEGACY_FIRMWARE_CUTOFF[1]}-{cls.LEGACY_FIRMWARE_CUTOFF[2]})."
                    )
                    return cls.LEGACY_URDF_FILE_PATH
                else:
                    logger.info(
                        f"Auto-selected default (new) URDF for firmware '{firmware_version}' "
                        f"(>= S-V{cls.LEGACY_FIRMWARE_CUTOFF[0]}.{cls.LEGACY_FIRMWARE_CUTOFF[1]}-{cls.LEGACY_FIRMWARE_CUTOFF[2]})."
                    )
                    return cls.DEFAULT_URDF_FILE_PATH

        return cls.DEFAULT_URDF_FILE_PATH

    @classmethod
    def _resolve_dh_is_offset(cls, firmware_version: Optional[str] = None) -> int:
        env_val = os.getenv(cls.DH_OFFSET_ENV_VAR, "").strip().lower()
        if env_val in {"0", "false", "no", "old"}:
            return 0
        if env_val in {"1", "true", "yes", "new"}:
            return 1

        # Auto-select based on firmware: old firmware = 0, new firmware = 1
        if firmware_version is not None:
            parsed = cls._parse_firmware_version(firmware_version)
            if parsed is not None:
                return 0 if parsed < cls.LEGACY_FIRMWARE_CUTOFF else 1

        # Default to 1 (new firmware) to match SDK default
        return 1

    def _init_sim_gripper(self) -> None:
        """Discover and cache gripper joints + limits in simulation."""

        self.gripper_initial_angle = None

        # Initialize gripper-related attributes with None/empty defaults
        self._gripper_joint_indices = []
        self._gripper_closed_positions = []
        self._gripper_open_positions = []
        self._gripper_joint_limits = []  # NEW: Cache joint limits

        try:
            name2idx = {
                self.sim.get_joint_info(robot_id=self.p_robot_id, joint_index=i)[
                    1
                ].decode("utf-8"): i
                for i in range(self.num_joints)
            }

            # Try to find joint7 and joint8 first
            if "joint7" in name2idx and "joint8" in name2idx:
                self._gripper_joint_indices = [name2idx["joint7"], name2idx["joint8"]]
            else:
                # fallback: last 2 prismatic joints
                found = [
                    i
                    for i in range(self.num_joints)
                    if self.sim.get_joint_info(robot_id=self.p_robot_id, joint_index=i)[
                        2
                    ]
                    == 1
                ]
                if len(found) >= 2:
                    self._gripper_joint_indices = found[-2:]
                    logger.debug(
                        f"Guessed prismatic joints: {self._gripper_joint_indices}"
                    )
                else:
                    logger.warning("Failed to auto-detect gripper joints; disabled")
                    return
        except Exception as e:
            logger.error(f"pybullet joint lookup failed: {e}")
            return

        # Cache joint limits and positions in one pass
        closed, opened, limits = [], [], []
        for jidx in self._gripper_joint_indices:
            info = self.sim.get_joint_info(robot_id=self.p_robot_id, joint_index=jidx)
            lower, upper = float(info[8]), float(info[9])

            # Cache the limits for later use
            limits.append((lower, upper))

            # Determine closed/open positions
            closed_pos, open_pos = (
                (upper, lower) if abs(upper) < abs(lower) else (lower, upper)
            )

            # Apply limits immediately
            closed.append(float(np.clip(closed_pos, lower, upper)))
            opened.append(float(np.clip(open_pos, lower, upper)))

        self._gripper_closed_positions = closed
        self._gripper_open_positions = opened
        self._gripper_joint_limits = limits  # NEW: Store cached limits

        logger.debug(
            f"Init gripper: {self._gripper_joint_indices=} {closed=} {opened=} {limits=}"
        )

    def _get_joint_records(self) -> list[dict[str, Union[int, str]]]:
        joint_records: list[dict[str, Union[int, str]]] = []
        for joint_index in range(self.num_joints):
            joint_info = self.sim.get_joint_info(
                robot_id=self.p_robot_id, joint_index=joint_index
            )
            joint_records.append(
                {
                    "index": joint_index,
                    "joint_name": joint_info[1].decode("utf-8"),
                    "joint_type": int(joint_info[2]),
                    "child_link": joint_info[12].decode("utf-8"),
                }
            )
        return joint_records

    def _joint_type_name(self, joint_type: int) -> str:
        joint_type_names = {
            p.JOINT_REVOLUTE: "revolute",
            p.JOINT_PRISMATIC: "prismatic",
            p.JOINT_FIXED: "fixed",
            p.JOINT_PLANAR: "planar",
            p.JOINT_POINT2POINT: "point2point",
        }
        return joint_type_names.get(joint_type, f"unknown({joint_type})")

    def _resolve_end_effector_link_index(self) -> int:
        joint_records = self._get_joint_records()
        for preferred_link_name in self.END_EFFECTOR_LINK_CANDIDATES:
            for joint_record in joint_records:
                child_link = str(joint_record["child_link"])
                if child_link != preferred_link_name:
                    continue
                self._resolved_end_effector_link_name = child_link
                if child_link != "link6":
                    logger.info(
                        f"Piper: resolved end-effector link '{child_link}' at joint index "
                        f"{joint_record['index']}."
                    )
                return int(joint_record["index"])

        logger.warning(
            f"Piper: failed to resolve an end-effector link from "
            f"{self.END_EFFECTOR_LINK_CANDIDATES}; falling back to link index "
            f"{self.END_EFFECTOR_LINK_INDEX}."
        )
        return self.END_EFFECTOR_LINK_INDEX

    def _log_runtime_model_diagnostics(self) -> None:
        urdf_path = Path(self.URDF_FILE_PATH)
        urdf_mtime = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(urdf_path.stat().st_mtime)
        )
        urdf_sha = hashlib.sha256(urdf_path.read_bytes()).hexdigest()[:12]
        joint_records = self._get_joint_records()
        joint_snapshot = [
            (
                record["index"],
                record["joint_name"],
                record["child_link"],
                self._joint_type_name(int(record["joint_type"])),
            )
            for record in joint_records
        ]
        actuated_joint_names = [
            str(joint_records[joint_index]["joint_name"])
            for joint_index in self.actuated_joints
            if joint_index < len(joint_records)
        ]

        logger.info(
            "Piper model diagnostics: "
            f"urdf='{urdf_path}' mtime='{urdf_mtime}' sha256='{urdf_sha}' "
            f"eef_link='{self._resolved_end_effector_link_name}' "
            f"eef_link_index={self.END_EFFECTOR_LINK_INDEX} "
            f"actuated_joints={self.actuated_joints} "
            f"actuated_joint_names={actuated_joint_names} "
            f"gripper_joint_indices={self._gripper_joint_indices}"
        )
        logger.debug(f"Piper joint chain: {joint_snapshot}")

        joint_names = {str(record["joint_name"]) for record in joint_records}
        expected_joint_names: Optional[tuple[str, ...]] = None
        validation_label: Optional[str] = None
        if urdf_path.resolve() == Path(self.DEFAULT_URDF_FILE_PATH).resolve():
            expected_joint_names = self.EXPECTED_DEFAULT_URDF_JOINT_NAMES
            validation_label = "default"
        elif urdf_path.resolve() == Path(self.LEGACY_URDF_FILE_PATH).resolve():
            expected_joint_names = self.EXPECTED_LEGACY_URDF_JOINT_NAMES
            validation_label = "legacy"

        if expected_joint_names is not None and validation_label is not None:
            missing_joint_names = sorted(set(expected_joint_names) - joint_names)
            if missing_joint_names:
                logger.error(
                    f"Piper {validation_label} URDF validation failed: "
                    f"missing expected joints {missing_joint_names}. "
                    f"Loaded URDF: {urdf_path}"
                )
            if self._resolved_end_effector_link_name not in self.END_EFFECTOR_LINK_CANDIDATES:
                logger.error(
                    f"Piper {validation_label} URDF validation failed: end effector "
                    f"not resolved to a known link. Resolved "
                    f"'{self._resolved_end_effector_link_name}' at index "
                    f"{self.END_EFFECTOR_LINK_INDEX}."
                )

    @classmethod
    def _parse_firmware_version(
        cls, firmware_version: Optional[str]
    ) -> Optional[tuple[int, int, int]]:
        if firmware_version is None:
            return None
        match = re.search(r"S-V(\d+)\.(\d+)-(\d+)", firmware_version)
        if match is None:
            return None
        return tuple(int(group) for group in match.groups())

    def _log_firmware_urdf_guidance(self) -> None:
        firmware_version = self._parse_firmware_version(self.firmware_version)
        if firmware_version is None:
            logger.warning(
                f"Piper: could not parse firmware version '{self.firmware_version}' "
                "for URDF guidance."
            )
            return

        using_default_urdf = (
            Path(self.URDF_FILE_PATH).resolve()
            == Path(self.DEFAULT_URDF_FILE_PATH).resolve()
        )
        if firmware_version < self.LEGACY_FIRMWARE_CUTOFF and using_default_urdf:
            logger.warning(
                "Piper firmware/URDF mismatch: firmware "
                f"'{self.firmware_version}' predates S-V1.6-3, while the active URDF is "
                f"the new-firmware model ('{self.URDF_FILE_PATH}'). "
                "AgileX piper_ros docs recommend the legacy DH model for firmware "
                "< S-V1.6-3 (see piper_description_old.urdf in agilexrobotics/piper_ros). "
                "The server normally auto-selects the matching URDF; this warning "
                "usually means an explicit override is forcing the default model."
            )
        elif firmware_version >= self.LEGACY_FIRMWARE_CUTOFF and not using_default_urdf:
            logger.warning(
                "Piper firmware/URDF mismatch: firmware "
                f"'{self.firmware_version}' is at or after S-V1.6-3, while the active URDF "
                f"is the legacy model ('{self.URDF_FILE_PATH}'). "
                "AgileX piper_ros docs recommend the new DH model for firmware "
                ">= S-V1.6-3 (see piper_description.urdf in agilexrobotics/piper_ros). "
                "The server normally auto-selects the matching URDF; this warning "
                "usually means an explicit override is forcing the legacy model."
            )

    def _reload_urdf(self) -> None:
        """Remove the current PyBullet body and reload with the updated URDF."""
        import pybullet as pb

        # Retrieve the current base position before removing
        base_pos, base_orn = pb.getBasePositionAndOrientation(self.p_robot_id)
        preserved_joint_positions: list[float] = []
        if getattr(self, "actuated_joints", None):
            preserved_joint_positions = self.sim.get_joints_states(
                robot_id=self.p_robot_id,
                joint_indices=self.actuated_joints,
            )
        pb.removeBody(self.p_robot_id)

        self.p_robot_id, num_joints, actuated_joints = self.sim.load_urdf(
            urdf_path=self.URDF_FILE_PATH,
            axis=list(base_pos),
            axis_orientation=list(base_orn),
            use_fixed_base=True,
            enable_self_collision=True,
        )
        self.num_joints = num_joints
        self.actuated_joints = actuated_joints
        self.num_actuated_joints = len(actuated_joints)
        self.gripper_servo_id = self.SERVO_IDS[-1]

        # Rebuild IK data structures
        joint_infos = [
            self.sim.get_joint_info(self.p_robot_id, i) for i in range(num_joints)
        ]
        self.ik_joint_indices = [
            i for i, info in enumerate(joint_infos) if info[2] != p.JOINT_FIXED
        ]
        self.ik_joint_index_lookup = {
            joint_index: idx for idx, joint_index in enumerate(self.ik_joint_indices)
        }
        self.lower_joint_limits = [joint_infos[i][8] for i in self.ik_joint_indices]
        self.upper_joint_limits = [joint_infos[i][9] for i in self.ik_joint_indices]

        # Re-resolve EEF link and gripper for new URDF
        self._resolved_end_effector_link_name = "link6"
        self.END_EFFECTOR_LINK_INDEX = self._resolve_end_effector_link_index()
        self._init_sim_gripper()
        if preserved_joint_positions:
            self.sim.sync_joints_immediate(
                robot_id=self.p_robot_id,
                joint_indices=self.actuated_joints[
                    : min(len(self.actuated_joints), len(preserved_joint_positions))
                ],
                target_positions=preserved_joint_positions[
                    : min(len(self.actuated_joints), len(preserved_joint_positions))
                ],
            )
        self._log_runtime_model_diagnostics()

    @classmethod
    def from_can_port(cls, can_name: str = "can0") -> Optional["PiperHardware"]:
        try:
            piper = cls(can_name=can_name)
            return piper
        except Exception as e:
            logger.warning(e)
            return None

    async def connect(self) -> None:
        """
        Setup the robot.
        can_number : 0 if only one robot is connected, 1 to connect to second robot
        """
        self.is_connected = False

        logger.info(f"Connecting to Agilex Piper on {self.can_name}")

        if not is_running_on_linux():
            logger.warning("Robot can only be connected on a Linux machine.")
            return

        try:
            proc = subprocess.Popen(
                [
                    "bash",
                    str(get_resources_path() / "agilex_can_activate.sh"),
                    self.can_name,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if proc.stdout is None or proc.stderr is None:
                logger.error("Failed to start the CAN activation script.")
                return

            try:
                stdout_data, stderr_data = proc.communicate(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                logger.warning(
                    "CAN activation script timed out (sudo may require a password). "
                    "Run: sudo visudo and add a NOPASSWD entry for ip and ethtool, "
                    "or run: sudo -v before starting the server."
                )
                return

            for line in stdout_data.splitlines():
                logger.debug("[can-script] " + line)
            for line in stderr_data.splitlines():
                logger.error("[can-script] " + line)

            if proc.returncode != 0:
                logger.warning(f"Script exited with exit code: {proc.returncode}")
                return
        except subprocess.CalledProcessError as e:
            logger.warning(
                f"CAN Activation Failed!\nError: {e}\nOutput:\n{e.stdout}\nErrors:\n{e.stderr}"
            )
            return

        logger.debug(f"Attempting to connect to Agilex Piper on {self.can_name}")
        self.motors_bus = C_PiperInterface_V2(
            can_name=self.can_name, judge_flag=True, can_auto_init=True
        )
        await asyncio.sleep(0.1)
        # Check if CAN bus is OK
        is_ok = self.motors_bus.isOk()
        if not is_ok:
            logger.warning(
                f"Could not connect to Agilex Piper on {self.can_name}: CAN bus is not OK."
            )
            return

        self.motors_bus.ConnectPort(can_init=True)
        if not self._enable_arm_with_retry():
            logger.warning(
                f"Could not enable Agilex Piper on {self.can_name} after connecting."
            )
            return

        # Start by resetting the control mode (useful if arm stuck in teaching mode)
        self.motors_bus.MotionCtrl_1(0x02, 0, 0)  # 恢复
        self.motors_bus.MotionCtrl_2(0, 0, 0, 0x00)  # 位置速度模式
        # Reset the gripper
        self.motors_bus.GripperCtrl(0, 1000, 0x00, 0)
        await asyncio.sleep(1.5)

        self.firmware_version = self.motors_bus.GetPiperFirmwareVersion()
        logger.info(
            f"Connected to Agilex Piper on {self.can_name} with firmware version {self.firmware_version}"
        )

        desired_dh = self._resolve_dh_is_offset(self.firmware_version)
        logger.info(
            f"Piper firmware '{self.firmware_version}' prefers SDK dh_is_offset={desired_dh}. "
            "Keeping the existing SDK session to avoid disrupting joint control."
        )

        # Auto-select URDF if no explicit override was set
        correct_urdf = self._resolve_urdf_file_path(self.firmware_version)
        if Path(correct_urdf).resolve() != Path(self.URDF_FILE_PATH).resolve():
            logger.info(
                f"Piper: auto-switching URDF from '{self.URDF_FILE_PATH}' to "
                f"'{correct_urdf}' based on firmware '{self.firmware_version}'."
            )
            self.URDF_FILE_PATH = correct_urdf
            # Reload the URDF into PyBullet
            self._reload_urdf()

        self._log_firmware_urdf_guidance()

        self.motors_bus.ArmParamEnquiryAndConfig(
            param_setting=0x01,
            # data_feedback_0x48x=0x02,
            end_load_param_setting_effective=0,
            set_end_load=0x0,
        )
        await asyncio.sleep(0.1)
        # First, start standby mode (ctrl_mode=0x00). Then, switch to CAN command control mode (ctrl_mode=0x01)
        # Source: https://static.generation-robots.com/media/agilex-piper-user-manual.pdf
        self.motors_bus.MotionCtrl_2(
            ctrl_mode=0x00, move_mode=0x01, move_spd_rate_ctrl=100, is_mit_mode=0x00
        )
        await asyncio.sleep(0.1)
        self.motors_bus.MotionCtrl_2(
            ctrl_mode=0x01,
            move_mode=0x01,
            move_spd_rate_ctrl=100,
            is_mit_mode=0x00,
            installation_pos=0x01,
            # installation_pos:
            # 0x01: Horizontal installation
            # 0x02: Left side installation
            # 0x03: Right side installation
        )
        await asyncio.sleep(0.2)
        self.is_torqued = True

        self.init_config()
        self.is_connected = True

    def get_default_base_robot_config(
        self, voltage: str, raise_if_none: bool = False
    ) -> Union[BaseRobotConfig, None]:
        return BaseRobotConfig(
            name=self.name,
            servos_voltage=12.0,
            servos_offsets=[0] * len(self.SERVO_IDS),
            servos_calibration_position=[1e-6] * len(self.SERVO_IDS),
            # Piper SDK joint readings are signed millidegrees. Keep the
            # fallback config neutral and let saved calibration data override it.
            servos_offsets_signs=[1.0] * len(self.SERVO_IDS),
            gripping_threshold=4500,
            non_gripping_threshold=500,
        )

    def disconnect(self) -> None:
        """
        Disconnect the robot.
        """

        if not self.is_connected:
            return

        # Reset the control mode
        self.motors_bus.MotionCtrl_1(0x02, 0, 0)  # 恢复
        self.motors_bus.MotionCtrl_2(0, 0, 0, 0x00)  # 位置速度模式
        time.sleep(0.1)

        # Disconnect
        self.motors_bus.DisconnectPort()
        self.is_connected = False
        self.is_torqued = False

    def _enable_arm_with_retry(
        self, timeout_s: float = 2.0, poll_interval_s: float = 0.05
    ) -> bool:
        """Enable the arm using the retry loop shown in the official SDK demos."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.motors_bus.EnablePiper():
                return True
            time.sleep(poll_interval_s)
        return False

    def init_config(self) -> None:
        """
        Load the config file.
        Try saved per-robot config first, then fall back to built-in defaults.
        """
        saved = BaseRobotConfig.from_serial_id(serial_id=self.SERIAL_ID, name=self.name)
        if saved is not None:
            self.config = saved
            logger.success(
                "Loaded Piper config from saved file with "
                f"offsets={saved.servos_offsets} signs={saved.servos_offsets_signs}"
            )
            return
        self.config = self.get_default_base_robot_config(voltage="24v")
        logger.warning(
            "Piper is using fallback joint conversion defaults because no saved "
            f"calibration config was found for '{self.SERIAL_ID}'. "
            f"offsets={self.config.servos_offsets if self.config else None} "
            f"signs={self.config.servos_offsets_signs if self.config else None}"
        )

    def enable_torque(self) -> None:
        if not self.is_connected:
            return
        self.is_torqued = self._enable_arm_with_retry()
        if not self.is_torqued:
            logger.warning("Piper EnablePiper() did not report success.")

    def disable_torque(self) -> None:
        # Disable torque
        if not self.is_connected:
            return
        self.motors_bus.DisableArm(7)
        # Disable the gripper with no change of zero position
        self.motors_bus.GripperCtrl(0, self.GRIPPER_EFFORT, self.DISABLE_GRIPPER, 0)
        self.is_torqued = False

    def read_motor_torque(self, servo_id: int) -> Optional[float]:
        """
        Read the torque of a motor

        raise: Exception if the routine has not been implemented
        """
        if servo_id >= self.GRIPPER_SERVO_ID:
            gripper_state = self.motors_bus.GetArmGripperMsgs().gripper_state
            return gripper_state.grippers_effort
        else:
            return 100 if self.is_torqued else 0

    def read_motor_voltage(self, servo_id: int) -> Optional[float]:
        """
        Read the voltage of a motor

        raise: Exception if the routine has not been implemented
        """
        # Not implemented
        return None

    def write_motor_position(self, servo_id: int, units: int, **kwargs: Any) -> None:
        """
        Move the motor to the specified position.

        Args:
            servo_id: The ID of the motor to move.
            units: The position to move the motor to. This is in the range 0 -> (self.RESOLUTION -1).
        Each position is mapped to an angle.
        """
        # If servo_id is 7 (gripper), write the gripper command
        if servo_id == self.GRIPPER_SERVO_ID:
            self.write_gripper_command(units)
            return

        # Otherwise, we need to write the position to the motor. We can only write all motors at once.
        current_position = self.read_joints_position(unit="motor_units", source="robot")
        # The last position is the gripper, so we drop it
        current_position = current_position[:-1]

        # Override the position of the specified servo_id
        current_position[servo_id - 1] = units

        # Clamp the position in the allowed range for the motors using self.piper_limits
        if servo_id in self.piper_limits_degrees:
            min_limit = self.piper_limits_degrees[servo_id]["min_angle_limit"] * 1000
            max_limit = self.piper_limits_degrees[servo_id]["max_angle_limit"] * 1000
            current_position[servo_id - 1] = np.clip(
                current_position[servo_id - 1], min_limit, max_limit
            )

        # Move robot
        self._ensure_joint_control_mode()
        self.joint_position = current_position.tolist()
        self.motors_bus.JointCtrl(*[int(q) for q in current_position])

    def set_motors_positions(
        self, q_target_rad: np.ndarray, enable_gripper: bool = False
    ) -> None:
        """
        Write the positions to the motors.

        If the robot is connected, the position is written to the motors.
        We always move the robot in the simulation.

        This does not control the gripper.

        q_target_rad is in radians.

        Args:
            q_target_rad: The position to move the motors to. This is in radians.
            enable_gripper: If True, the gripper will be moved to the position specified in q_target_rad.
        """
        logger.debug(
            f"Piper: Setting motors to {q_target_rad} rad, gripper enabled: {enable_gripper}"
        )
        joint_indices = (
            self.actuated_joints
        )  # size 6, the gripper is excluded from actuated_joints in the Piper class
        target_positions = [q_target_rad[i] for i in joint_indices]

        # Validate against an immediate joint sync. The background stepping
        # loop is asynchronous, so reading joint states right after
        # setJointMotorControlArray can return the previous pose and flatten
        # real CAN commands back to the current robot pose.
        self.sim.sync_joints_immediate(
            robot_id=self.p_robot_id,
            joint_indices=joint_indices,
            target_positions=target_positions,
        )
        # Keep motor targets aligned with the synced pose for subsequent
        # background steps and visualization.
        self.sim.set_joints_states(
            robot_id=self.p_robot_id,
            joint_indices=joint_indices,
            target_positions=target_positions,
        )
        if enable_gripper and len(q_target_rad) >= self.GRIPPER_SERVO_ID:
            gripper_open = self._rad_to_open_command(
                q_target_rad[self.GRIPPER_SERVO_ID - 1]
            )
            self.move_gripper_in_sim(open=gripper_open)
        self.sim.step()

        if self.is_connected:
            validated_q_target_array = np.array(
                self.sim.get_joints_states(
                    robot_id=self.p_robot_id, joint_indices=joint_indices
                )
            )
            q_target = self._radians_vec_to_motor_units(validated_q_target_array)
            logger.debug(
                "Piper: validated joint target "
                f"{np.round(validated_q_target_array, 6).tolist()} rad -> "
                f"{q_target.tolist()} motor units"
            )
            if enable_gripper and len(q_target_rad) < self.GRIPPER_SERVO_ID:
                q_target = np.append(
                    q_target,
                    q_target_rad[-1]
                    * self.GRIPPER_MAX_ANGLE
                    * self.RESOLUTION
                    / (np.pi / 2),
                )
            self.write_group_motor_position(
                q_target=q_target, enable_gripper=enable_gripper
            )

    def write_group_motor_position(
        self,
        q_target: np.ndarray,  # in motor units
        enable_gripper: bool,
    ) -> None:
        # First 6 values of q_target are the joints position.
        # Clamp joints in the allowed range for the motors using self.piper_limits_degrees * 1000
        # Simply clamping the values is not strong enough, as in certain positions the angle limits are exceeded.
        # To avoid this, we first set the joints in pybullet, then read the validated position and use it to clamp the values.

        clamped_joints = []
        for i, joint in enumerate(q_target):
            # in self.piper_limits_degrees, there are only indexes 1 to 6, we forgo the gripper
            servo_id = i + 1
            if servo_id in self.piper_limits_degrees:
                min_limit = self.piper_limits_degrees[i + 1]["min_angle_limit"] * 1000
                max_limit = self.piper_limits_degrees[i + 1]["max_angle_limit"] * 1000
                # q_target[i] = np.clip(joint, min_limit, max_limit)  # noqa: F821
                clamped_joint = int(np.clip(joint, min_limit, max_limit))
                clamped_joints.append(clamped_joint)

        self._ensure_joint_control_mode()
        self.motors_bus.JointCtrl(*clamped_joints)

        # Move the gripper if it is enabled
        if enable_gripper and len(q_target) >= self.GRIPPER_SERVO_ID:
            # The last value q_target[-1] is the gripper position in motor units. Rescale it between (0, 1) to write the gripper command
            gripper_position = q_target[-1]
            gripper_command = gripper_position / (
                self.GRIPPER_MAX_ANGLE * self.RESOLUTION
            )
            self.write_gripper_command(gripper_command)

    def read_motor_position(self, servo_id: int, **kwargs: Any) -> Optional[int]:
        """
        Read the position of the motor. This should return the position in motor units.
        """
        return self.read_group_motor_position()[servo_id - 1]

    def read_joints_position(
        self,
        unit: Literal["rad", "motor_units", "degrees", "other"] = "rad",
        source: Literal["sim", "robot"] = "robot",
        joints_ids: Optional[List[int]] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> np.ndarray:
        """
        Read the position of the joints. This should return the position in motor units.
        """
        # The parent method reads the joints, but not the gripper.
        joints = super().read_joints_position(
            unit=unit,
            source=source,
            joints_ids=joints_ids,
            min_value=min_value,
            max_value=max_value,
        )

        # Add the gripper position if it is not already present
        if len(joints) < self.GRIPPER_SERVO_ID and (
            joints_ids is None or self.GRIPPER_SERVO_ID in joints_ids
        ):
            gripper_position = self.read_gripper_command(
                source=source, unit=unit, min_value=min_value, max_value=max_value
            )

            joints = np.array(joints.tolist() + [gripper_position]).astype(np.float32)
        return joints

    def read_group_motor_position(self) -> np.ndarray:
        """
        Read the position of all the motors. This should return the position in motor units.
        """
        joint_state = self.motors_bus.GetArmJointMsgs().joint_state
        # in 0.001 deg
        position_unit = np.array(
            [
                joint_state.joint_1,
                joint_state.joint_2,
                joint_state.joint_3,
                joint_state.joint_4,
                joint_state.joint_5,
                joint_state.joint_6,
            ]
        )

        return position_unit

    def calibrate_motors(self, **kwargs: Any) -> None:
        """
        This is called during the calibration phase of the robot.
        It sets the offset of all motors to self.RESOLUTION/2.
        """
        # Set zero positions for motors and gripper
        self.motors_bus.JointConfig(set_zero=0xAE)  # Set zero position of motors
        # Set zero position of gripper
        self.motors_bus.GripperCtrl(0, self.GRIPPER_EFFORT, 0x00, 0xAE)

    def _units_vec_to_radians(self, units: np.ndarray) -> np.ndarray:
        """
        Route Piper through BaseManipulator's offset/sign-aware conversion path.
        """
        return super()._units_vec_to_radians(units)

    def _radians_vec_to_motor_units(self, radians: np.ndarray) -> np.ndarray:
        """
        Route Piper through BaseManipulator's offset/sign-aware conversion path.
        """
        return super()._radians_vec_to_motor_units(radians)

    async def calibrate(self) -> tuple[Literal["success", "in_progress", "error"], str]:
        """
        This is called during the calibration phase of the robot.
        CAUTION : Set the robot in sleep mode where falling wont be an issue and close the gripper.
        """
        if not self.is_connected:
            logger.warning("Robot is not connected. Cannot calibrate.")
            return "error", "Robot is not connected. Cannot calibrate."

        if self.calibration_current_step == 0:
            self.calibration_current_step = 1
            return (
                "in_progress",
                "STEP 1/3: NEXT STEP, THE ROBOT WILL FALL. HOLD THE ROBOT to prevent it from falling.",
            )
        elif self.calibration_current_step == 1:
            self.disable_torque()
            self.calibration_current_step = 2
            return (
                "in_progress",
                "STEP 2/3: Move the robot to its sleep position. Close the gripper fully.",
            )
        elif self.calibration_current_step == 2:
            self.calibration_current_step = 0
            self.calibrate_motors()
            self.enable_torque()
            return (
                "success",
                "STEP 3/3: Calibration completed successfully. Offsets and signs saved to the robot.",
            )

        return (
            "error",
            "Calibration failed. Please try again.",
        )

    def read_gripper_command(
        self,
        source: Literal["sim", "robot"] = "robot",
        unit: Literal["motor_units", "rad", "degrees", "other"] = "motor_units",
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> float:
        """
        Read if gripper is open or closed.
        """

        if not self.is_connected:
            logger.warning("Robot not connected")
            return 0

        if source == "robot":
            gripper_ctrl = self.motors_bus.GetArmGripperMsgs().gripper_state
            gripper_position = gripper_ctrl.grippers_angle
            # Calculate normalized gripper position [0, 1]
            normalized = (gripper_position - self.GRIPPER_ZERO_POSITION) / (
                self.GRIPPER_MAX_ANGLE * 1000
            )
        elif source == "sim":
            if not self._gripper_joint_indices:
                return 0.0

            fractions = []
            for jidx, closed_pos, open_pos in zip(
                self._gripper_joint_indices,
                self._gripper_closed_positions,
                self._gripper_open_positions,
            ):
                pos = float(
                    self.sim.get_joint_state(
                        robot_id=self.p_robot_id, joint_index=jidx
                    )[0]
                )
                denom = open_pos - closed_pos
                f = (pos - closed_pos) / denom if abs(denom) > 1e-9 else 0.0
                fractions.append(float(np.clip(f, 0.0, 1.0)))
            normalized = float(np.mean(fractions)) if fractions else 0.0
        else:
            raise ValueError(f"Unknown source: {source}")

        if unit == "motor_units":
            # Don't do anything
            gripper_units = normalized
        elif unit == "rad":
            # Convert the gripper from (0, GRIPPER_MAX_ANGLE) to (0, pi / 2)
            gripper_units = normalized * (np.pi / 2)
        elif unit == "degrees":
            # Convert the gripper from (0, GRIPPER_MAX_ANGLE) to (0, 90)
            gripper_units = normalized * 90
        elif unit == "other":
            # Convert the gripper from (0, GRIPPER_MAX_ANGLE) to (min_value, max_value)
            if min_value is None or max_value is None:
                raise ValueError(
                    "min_value and max_value must be provided for 'other' unit."
                )
            gripper_units = normalized * (max_value - min_value) + min_value
        else:
            raise ValueError(f"Unknown unit: {unit}")

        return gripper_units

    def _rad_to_open_command(self, radians: float) -> float:
        """
        Convert radians to an open command for the gripper.
        The open command is in the range [0, 1], where 0 is fully closed and 1 is fully open.
        """
        # Clip to valid range and normalize to [0, 1]
        clipped_radians = np.clip(radians, 0, np.pi / 2)  # Max 90 degrees (π/2 rad)
        open_command = clipped_radians / (np.pi / 2)  # Normalize to [0, 1]
        return open_command

    def write_gripper_command(self, command: float) -> None:
        """
        Open or close the gripper.

        command: 0 to close, 1 to open
        """
        if not self.is_connected:
            logger.debug("Robot not connected, cannot write gripper command")
            return
        # Gripper -> Convert from 0->RESOLUTION to 0->GRIPPER_MAX_ANGLE
        unit_degree = command * self.GRIPPER_MAX_ANGLE
        unit_command = self.GRIPPER_ZERO_POSITION + int(unit_degree * 1000)
        self.motors_bus.GripperCtrl(
            gripper_angle=unit_command,
            gripper_effort=self.GRIPPER_EFFORT,
            gripper_code=self.ENABLE_GRIPPER,
            set_zero=0,
        )
        self.update_object_gripping_status()

    def is_powered_on(self) -> bool:
        """
        Check if the robot is powered on.
        """
        return self.is_connected

    def status(self) -> RobotConfigStatus:
        """
        Get the status of the robot.

        Returns:
            RobotConfigStatus object
        """
        return RobotConfigStatus(
            name=self.name, device_name=self.can_name, robot_type="manipulator"
        )

    def _read_sdk_end_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Read the current end-effector pose from the SDK.

        Returns (position_meters, orientation_rad) as numpy arrays.
        SDK units: 0.001mm for position, 0.001deg for orientation.
        """
        msgs = self.motors_bus.GetArmEndPoseMsgs()
        pose = msgs.end_pose
        position = np.array([
            pose.X_axis * 1e-6,  # 0.001mm -> m
            pose.Y_axis * 1e-6,
            pose.Z_axis * 1e-6,
        ])
        orientation_rad = np.array([
            np.deg2rad(pose.RX_axis * 1e-3),  # 0.001deg -> deg -> rad
            np.deg2rad(pose.RY_axis * 1e-3),
            np.deg2rad(pose.RZ_axis * 1e-3),
        ])
        return position, orientation_rad

    def _ensure_joint_control_mode(self) -> None:
        """Force joint control mode (MOVE J) before joint-space commands.

        The previous working implementation sent this ModeCtrl command for
        every joint write. That is intentional: the Piper controller can be
        reset or left in a different mode even when our cached state says
        MOVE J, and `home`/`ready` depend on raw JointCtrl commands.
        """
        self.motors_bus.ModeCtrl(
            ctrl_mode=0x01,
            move_mode=0x01,  # MOVE J
            move_spd_rate_ctrl=100,
            is_mit_mode=0x00,
        )
        self._current_move_mode = 0x01

    def _send_sdk_end_pose(
        self,
        position_m: np.ndarray,
        orientation_rad: np.ndarray,
    ) -> None:
        """Send an absolute end-effector pose via the SDK's EndPoseCtrl.

        Converts from meters/radians to SDK units (0.001mm / 0.001deg).
        Switches to Cartesian control mode (MOVE P) before sending.
        """
        # Switch to Cartesian point-to-point mode (ctrl_mode=0x01, move_mode=0x00)
        self.motors_bus.MotionCtrl_2(
            ctrl_mode=0x01,
            move_mode=0x00,  # MOVE P (Cartesian point-to-point)
            move_spd_rate_ctrl=100,
            is_mit_mode=0x00,
        )
        self._current_move_mode = 0x00

        x_cmd = int(round(position_m[0] * 1e6))  # m -> 0.001mm
        y_cmd = int(round(position_m[1] * 1e6))
        z_cmd = int(round(position_m[2] * 1e6))
        rx_cmd = int(round(np.rad2deg(orientation_rad[0]) * 1e3))  # rad -> deg -> 0.001deg
        ry_cmd = int(round(np.rad2deg(orientation_rad[1]) * 1e3))
        rz_cmd = int(round(np.rad2deg(orientation_rad[2]) * 1e3))

        logger.debug(
            f"Piper EndPoseCtrl: X={x_cmd} Y={y_cmd} Z={z_cmd} "
            f"RX={rx_cmd} RY={ry_cmd} RZ={rz_cmd}"
        )
        self.motors_bus.EndPoseCtrl(
            X=x_cmd, Y=y_cmd, Z=z_cmd,
            RX=rx_cmd, RY=ry_cmd, RZ=rz_cmd,
        )

    async def move_robot_relative(
        self,
        target_position: np.ndarray,
        target_orientation_rad: np.ndarray,
    ) -> None:
        """Move the Piper end-effector by a relative delta using SDK-native Cartesian control.

        When connected to real hardware, this bypasses PyBullet IK entirely and
        uses the SDK's EndPoseCtrl for accurate Cartesian motion. Falls back to
        the base class IK path for simulation-only mode.

        Args:
            target_position: Delta [dx, dy, dz] in meters (already converted from cm).
            target_orientation_rad: Delta [drx, dry, drz] in radians. The
                control endpoint already provides relative orientation deltas.
        """
        if not self.is_connected:
            # Simulation-only: fall back to PyBullet IK path
            current_pos, current_orient = self.forward_kinematics(sync_robot_pos=False)
            abs_pos = current_pos + np.array([
                0 if v is None else v for v in target_position
            ])
            abs_orient = current_orient + np.array([
                0 if v is None else v for v in target_orientation_rad
            ])
            await self.move_robot_absolute(
                target_position=abs_pos,
                target_orientation_rad=abs_orient,
            )
            return

        # Read the current pose from the SDK (not from PyBullet FK)
        current_pos, current_orient = self._read_sdk_end_pose()

        # Apply delta
        delta_pos = np.array([0 if v is None else v for v in target_position])
        delta_orient = np.array([0 if v is None else v for v in target_orientation_rad])

        new_pos = current_pos + delta_pos
        new_orient = current_orient + delta_orient

        logger.debug(
            f"Piper move_robot_relative: SDK current_pos={np.round(current_pos, 4).tolist()} "
            f"delta={np.round(delta_pos, 4).tolist()} -> target={np.round(new_pos, 4).tolist()}"
        )

        # Send via SDK-native Cartesian control
        self._send_sdk_end_pose(new_pos, new_orient)

        # Also update sim for visualization (non-critical path)
        try:
            # Read back the joint positions from the real robot after a short delay
            await asyncio.sleep(0.05)
            current_joints = self.read_joints_position(unit="rad", source="robot")
            target_positions = current_joints[:len(self.actuated_joints)].tolist()
            self.sim.sync_joints_immediate(
                robot_id=self.p_robot_id,
                joint_indices=self.actuated_joints,
                target_positions=target_positions,
            )
        except Exception as exc:
            logger.debug(f"Piper: sim sync after SDK move failed (non-critical): {exc}")

    async def move_to_ready_position(self) -> None:
        await super().move_to_ready_position()
        effector_position, effector_orientation_rad = self.forward_kinematics(
            sync_robot_pos=self.is_connected
        )
        logger.info(
            "Piper ready FK: "
            f"eef_link='{self._resolved_end_effector_link_name}' "
            f"position={np.round(effector_position, 4).tolist()} "
            f"orientation_rad={np.round(effector_orientation_rad, 4).tolist()}"
        )

    def move_gripper_in_sim(self, open: float) -> None:
        """
        Move the AgileX Piper gripper in the simulation.
        `open` is normalized: 0.0 = fully closed, 1.0 = fully open.
        This updates both prismatic fingers (URDF joints `joint7` and `joint8`).
        """

        # Early returns for edge cases
        if not self._gripper_joint_indices:
            logger.debug("No gripper joints available")
            return

        if self.is_object_gripped:
            return

        # Clamp input to [0,1]
        open = float(np.clip(open, 0.0, 1.0))

        # Calculate target positions using cached data
        target_positions = []
        for i, (close, open_pos, (lower, upper)) in enumerate(
            zip(
                self._gripper_closed_positions,
                self._gripper_open_positions,
                self._gripper_joint_limits,
            )
        ):
            # Interpolate target position
            target = close + (open_pos - close) * open

            # Apply cached joint limits
            if lower <= upper:
                target = float(np.clip(target, lower, upper))

            target_positions.append(target)

        # Apply the joint positions
        self.sim.set_joints_states(
            robot_id=self.p_robot_id,
            joint_indices=self._gripper_joint_indices,
            target_positions=target_positions,
        )
