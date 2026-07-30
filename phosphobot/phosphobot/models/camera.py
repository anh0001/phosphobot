from typing import List

from pydantic import BaseModel, Field

from phosphobot.types import CameraTypes


class SingleCameraStatus(BaseModel):
    camera_id: int
    is_active: bool
    is_disabled: bool = Field(
        default=False,
        description="Whether the camera was turned off on purpose, releasing the "
        "underlying device so other processes can use it. A camera that is neither "
        "active nor disabled failed to open.",
    )
    camera_type: CameraTypes = Field(
        description="Type of camera."
        + "\n`classic`: Standard camera detected by OpenCV."
        + "\n`stereo`: Stereoscopic camera. It has two lenses: left eye and right eye to give a 3D effect. The left half of the image is the left eye, and the right half is the right eye."
        + "\n`realsense`: Intel RealSense camera. It use infrared sensors to provide depth information. It requires a special driver."
        + "\n`dummy`: Dummy camera. Used for testing."
        + "\n`dummy_stereo`: Dummy stereoscopic camera. Used for testing."
        + "\n`unknown`: Unknown camera type."
    )
    width: int
    height: int
    fps: int


class AllCamerasStatus(BaseModel):
    """
    Description of the status of all cameras. Use this to know which camera to stream.
    """

    cameras_status: List[SingleCameraStatus] = Field(default_factory=list)
    is_stereo_camera_available: bool = Field(
        default=False, description="Whether a stereoscopic camera is available."
    )
    realsense_available: bool = Field(
        default=False, description="Whether a RealSense camera is available."
    )
    video_cameras_ids: List[int] = Field(
        default_factory=list,
        description="List of camera ids that are video cameras.",
    )


class CameraToggleResult(BaseModel):
    """Outcome of enabling or disabling a single camera."""

    camera_id: int
    is_active: bool = Field(
        description="Whether the camera is streaming after the operation."
    )
    is_disabled: bool = Field(
        description="Whether the camera is intentionally off after the operation."
    )


class CameraToggleResponse(BaseModel):
    """Outcome of an enable/disable request on one or several cameras."""

    message: str
    cameras: List[CameraToggleResult] = Field(default_factory=list)
