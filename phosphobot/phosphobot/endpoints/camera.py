import asyncio
import base64
from typing import Dict, Optional

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
)
from fastapi.responses import StreamingResponse
from loguru import logger

from phosphobot.camera import AllCameras, ZMQCamera, get_all_cameras
from phosphobot.models import (
    AddZMQCameraRequest,
    CameraToggleResponse,
    CameraToggleResult,
)

router = APIRouter(tags=["camera"])


def _toggle_result(camera_id: int, cameras: AllCameras) -> CameraToggleResult:
    """Read back the state of a camera after an enable/disable operation."""
    camera = cameras.get_camera_by_id(camera_id)
    return CameraToggleResult(
        camera_id=camera_id,
        is_active=camera.is_active if camera is not None else False,
        is_disabled=camera.is_disabled if camera is not None else True,
    )


@router.get(
    "/video/{camera_id}",
    description="Stream video feed of the specified camera. "
    + "If no camera id is provided, the default camera is used. "
    + "Specify a target size and quality using query parameters.",
    responses={
        200: {"description": "Streaming video feed of the specified camera."},
        404: {"description": "Camera not available"},
    },
    response_model=None,
)
def video_feed_for_camera(
    request: Request,
    camera_id: Optional[int],
    height: Optional[int] = None,
    width: Optional[int] = None,
    quality: Optional[int] = None,
    cameras: AllCameras = Depends(get_all_cameras),
) -> StreamingResponse | HTTPException:
    """
    Stream video feed of the specified camera.
    """

    if width is None or height is None:
        target_size = None
    else:
        target_size = (width, height)
    logger.debug(
        f"Received request for camera {camera_id} with target size {target_size} and quality {quality}"
    )
    if camera_id is None:
        camera_id = 0

    # Convert to integer the parameter if read as a string
    if isinstance(camera_id, str) and camera_id.isdigit():
        camera_id = int(camera_id)

    if not (isinstance(camera_id, int) or isinstance(camera_id, str)):
        raise HTTPException(
            status_code=400,
            detail=f"Unprocessable type for camera id. Received {type(camera_id)}",
        )

    if quality and (quality < 0 or quality > 100):
        raise HTTPException(
            status_code=400,
            detail=f"Quality must be between 0 and 100. Received {quality}",
        )

    stream_params = {
        "target_size": target_size,
        "quality": quality,
    }

    camera = cameras.get_camera_by_id(camera_id)
    if camera is not None and camera.is_disabled:
        raise HTTPException(
            status_code=404,
            detail=f"Camera {camera_id} is disabled. "
            f"Enable it with POST /cameras/{camera_id}/enable to stream it.",
        )
    if camera is None or not camera.is_active:
        raise HTTPException(status_code=404, detail="Camera not available")
    logger.info(f"Starting video feed with params {stream_params}")
    return StreamingResponse(
        camera.generate_rgb_frames(
            target_size=target_size, quality=quality, request=request
        ),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@router.get(
    "/frames",
    response_model=Dict[str, Optional[str]],
    description="Capture frames from all available cameras. "
    + "Returns a dictionary with camera IDs as keys and base64 encoded JPG images as values. "
    + "If a camera is not available or fails to capture, its value will be None.",
    responses={
        200: {
            "description": "Successfully captured frames from available cameras",
            "content": {
                "application/json": {
                    "example": {
                        "0": "base64_encoded_image_string",
                        "1": None,
                        "realsense": "base64_encoded_image_string",
                    }
                }
            },
        },
        500: {"description": "Server error while capturing frames"},
    },
)
async def get_all_camera_frames(
    resize_x: Optional[int] = None,
    resize_y: Optional[int] = None,
    cameras: AllCameras = Depends(get_all_cameras),
) -> Dict[str, Optional[str]]:
    """
    Capture and return frames from all available cameras.
    Returns:
        Dict[str, Optional[str]]: Dictionary mapping camera IDs to base64 encoded JPG images
        or None if camera is unavailable/failed to capture
    """
    logger.debug("Received request for all camera frames")

    # We can add a resize here if needed
    if resize_x is not None and resize_y is not None:
        resize = (resize_x, resize_y)
    else:
        resize = None

    frames = cameras.get_rgb_frames_for_all_cameras(resize=resize)

    # Initialize response dictionary
    response: Dict[str, Optional[str]] = {}

    import cv2

    # Process each frame
    for camera_id, frame in frames.items():
        try:
            if frame is None:
                response[camera_id] = None
                continue

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Encode frame as JPG
            _, buffer = cv2.imencode(".jpg", rgb_frame)

            # Convert to base64 string
            base64_frame = base64.b64encode(buffer.tobytes()).decode("utf-8")

            response[camera_id] = base64_frame

        except Exception as e:
            logger.error(f"Error processing frame for camera {camera_id}: {str(e)}")
            response[camera_id] = None

    if not response:
        raise HTTPException(
            status_code=503,
            detail=f"No frames captured from any camera: frames={frames} and cameras={cameras}",
        )

    return response


@router.post(
    "/cameras/refresh",
    response_model=dict,
    description="Refresh the list of available cameras. "
    + "This operation can take a few seconds as it disconnects and reconnects to all cameras. "
    + "It is useful when cameras are added or removed while the application is running.",
)
async def refresh_camera_list(
    cameras: AllCameras = Depends(get_all_cameras),
) -> dict:
    """
    Refresh the list of available cameras.
    This operation can take a few seconds as it disconnects and reconnects to all cameras.
    It is useful when cameras are added or removed while the application is running.
    """
    cameras.refresh()
    return {"message": "Camera list refreshed successfully"}


@router.post(
    "/cameras/enable-all",
    response_model=CameraToggleResponse,
    description="Re-acquire every camera that was previously disabled.",
)
async def enable_all_cameras(
    cameras: AllCameras = Depends(get_all_cameras),
) -> CameraToggleResponse:
    """
    Enable all cameras, re-opening the devices phosphobot released.
    """
    results = cameras.set_all_cameras_enabled(enabled=True)
    failed = [camera_id for camera_id, is_active in results.items() if not is_active]
    if failed:
        message = (
            f"Enabled {len(results) - len(failed)}/{len(results)} cameras. "
            f"Could not re-open: {failed}. They may still be used by another process."
        )
    else:
        message = f"Enabled {len(results)} cameras"

    return CameraToggleResponse(
        message=message,
        cameras=[_toggle_result(camera_id, cameras) for camera_id in results],
    )


@router.post(
    "/cameras/disable-all",
    response_model=CameraToggleResponse,
    description="Disable every camera and release the underlying devices, "
    + "so that another process can use them. Streaming and recording will not "
    + "capture frames until the cameras are enabled again.",
)
async def disable_all_cameras(
    cameras: AllCameras = Depends(get_all_cameras),
) -> CameraToggleResponse:
    """
    Disable all cameras and release the underlying devices.
    """
    results = cameras.set_all_cameras_enabled(enabled=False)
    return CameraToggleResponse(
        message=f"Disabled {len(results)} cameras. Devices are released.",
        cameras=[_toggle_result(camera_id, cameras) for camera_id in results],
    )


@router.post(
    "/cameras/{camera_id}/enable",
    response_model=CameraToggleResponse,
    description="Re-acquire a camera device that was previously disabled.",
    responses={
        404: {"description": "Camera not found"},
        409: {"description": "Camera could not be re-opened"},
    },
)
async def enable_camera(
    camera_id: int,
    cameras: AllCameras = Depends(get_all_cameras),
) -> CameraToggleResponse:
    """
    Enable a single camera, re-opening the device phosphobot released.
    """
    try:
        is_active = cameras.set_camera_enabled(camera_id, enabled=True)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if not is_active:
        raise HTTPException(
            status_code=409,
            detail=f"Camera {camera_id} could not be re-opened. "
            "It may still be used by another process.",
        )

    return CameraToggleResponse(
        message=f"Camera {camera_id} enabled",
        cameras=[_toggle_result(camera_id, cameras)],
    )


@router.post(
    "/cameras/{camera_id}/disable",
    response_model=CameraToggleResponse,
    description="Disable a camera and release the underlying device, so that "
    + "another process can use it. Streaming and recording will not capture "
    + "frames from this camera until it is enabled again.",
    responses={404: {"description": "Camera not found"}},
)
async def disable_camera(
    camera_id: int,
    cameras: AllCameras = Depends(get_all_cameras),
) -> CameraToggleResponse:
    """
    Disable a single camera and release the underlying device.
    """
    try:
        cameras.set_camera_enabled(camera_id, enabled=False)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return CameraToggleResponse(
        message=f"Camera {camera_id} disabled. Device is released.",
        cameras=[_toggle_result(camera_id, cameras)],
    )


@router.post(
    "/cameras/add-zmq",
    response_model=dict,
    description="Add a camera feed from a ZMQ publisher. ",
)
async def add_zmq_camera_feed(
    query: AddZMQCameraRequest,
    cameras: AllCameras = Depends(get_all_cameras),
) -> dict:
    """
    Add a camera feed from a ZMQ publisher.
    This allows the application to receive camera frames from a ZMQ publisher.
    """
    try:
        zmq_camera = ZMQCamera(connect_to=query.tcp_address, topic=query.topic)
        await asyncio.sleep(0.1)  # Allow some time for the camera to initialize
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to connect to ZMQ publisher at {query.tcp_address}: {str(e)}",
        )

    # Add to cameras
    cameras.add_custom_camera(zmq_camera)
    return {"message": "ZMQ camera added successfully"}
