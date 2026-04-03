import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from phosphobot.configs import config
from phosphobot.robot import RobotConnectionManager


@patch("phosphobot.robot.list_ports.comports", return_value=[])
@patch(
    "phosphobot.robot.list_can_interfaces",
    return_value=["can_piper", "can7", "can_extra"],
)
def test_scan_ports_uses_detected_can_interface_names(
    mock_list_can_interfaces, _mock_comports
):
    manager = RobotConnectionManager()
    previous_enable_can = config.ENABLE_CAN
    previous_max_can_interfaces = config.MAX_CAN_INTERFACES
    previous_preferred_can_interfaces = config.PREFERRED_CAN_INTERFACES

    try:
        config.ENABLE_CAN = True
        config.MAX_CAN_INTERFACES = 2
        config.PREFERRED_CAN_INTERFACES = None

        ports, can_ports = manager._scan_ports()

        assert ports == []
        assert can_ports == ["can_piper", "can7", "can_extra"]
        mock_list_can_interfaces.assert_called_once_with(
            max_interfaces=2, preferred_interfaces=None
        )
    finally:
        config.ENABLE_CAN = previous_enable_can
        config.MAX_CAN_INTERFACES = previous_max_can_interfaces
        config.PREFERRED_CAN_INTERFACES = previous_preferred_can_interfaces


@patch("phosphobot.robot.list_ports.comports", return_value=[])
@patch("phosphobot.robot.list_can_interfaces", return_value=["can2"])
def test_scan_ports_passes_preferred_can_interfaces(
    mock_list_can_interfaces, _mock_comports
):
    manager = RobotConnectionManager()
    previous_enable_can = config.ENABLE_CAN
    previous_max_can_interfaces = config.MAX_CAN_INTERFACES
    previous_preferred_can_interfaces = config.PREFERRED_CAN_INTERFACES

    try:
        config.ENABLE_CAN = True
        config.MAX_CAN_INTERFACES = 4
        config.PREFERRED_CAN_INTERFACES = ["can_piper", "can2"]

        ports, can_ports = manager._scan_ports()

        assert ports == []
        assert can_ports == ["can2"]
        mock_list_can_interfaces.assert_called_once_with(
            max_interfaces=4,
            preferred_interfaces=["can_piper", "can2"],
        )
    finally:
        config.ENABLE_CAN = previous_enable_can
        config.MAX_CAN_INTERFACES = previous_max_can_interfaces
        config.PREFERRED_CAN_INTERFACES = previous_preferred_can_interfaces
