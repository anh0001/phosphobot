import json
from unittest.mock import Mock, patch

from phosphobot.utils import list_can_interfaces


@patch("phosphobot.utils.subprocess.run")
def test_list_can_interfaces_resolves_altname_to_primary_interface(
    mock_subprocess_run,
):
    mock_subprocess_run.return_value = Mock(
        stdout=json.dumps(
            [
                {"ifname": "can0", "altnames": ["can_piper"]},
                {"ifname": "can1", "altnames": ["can_gripper"]},
            ]
        )
    )

    can_interfaces = list_can_interfaces(preferred_interfaces=["can_piper"])

    assert can_interfaces == ["can0"]
    mock_subprocess_run.assert_called_once_with(
        ["ip", "-j", "-d", "link", "show", "type", "can"],
        capture_output=True,
        text=True,
        check=True,
        timeout=3,
    )
