"""Tests for enabling/disabling cameras so other processes can use the devices."""

import threading
from typing import Optional, Tuple

import numpy as np
import pytest
from phosphobot.camera import DummyCamera, VideoCamera

# ---------------------------------------------------------------------------
# A fake OpenCV capture that emulates the exclusive access of /dev/videoN
# ---------------------------------------------------------------------------

CAP_PROP_FRAME_WIDTH = 3
CAP_PROP_FRAME_HEIGHT = 4
CAP_PROP_FPS = 5


class FakeDevice:
    """Stands in for a V4L2 node: only one capture may hold it at a time."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def acquire(self) -> bool:
        return self._lock.acquire(blocking=False)

    def release(self) -> None:
        try:
            self._lock.release()
        except RuntimeError:
            pass

    @property
    def is_free(self) -> bool:
        if self._lock.acquire(blocking=False):
            self._lock.release()
            return True
        return False


class FakeCapture:
    def __init__(self, device: FakeDevice) -> None:
        self.device = device
        self.opened = device.acquire()

    def isOpened(self) -> bool:  # noqa: N802 - mirrors the cv2 API
        return self.opened

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.opened:
            return False, None
        return True, np.zeros((480, 640, 3), dtype=np.uint8)

    def get(self, prop: int) -> float:
        return {
            CAP_PROP_FRAME_WIDTH: 640.0,
            CAP_PROP_FRAME_HEIGHT: 480.0,
            CAP_PROP_FPS: 30.0,
        }.get(prop, 0.0)

    def set(self, prop: int, value: float) -> bool:
        return True

    def release(self) -> None:
        if self.opened:
            self.opened = False
            self.device.release()


class FakeVideoCamera(VideoCamera):
    """A VideoCamera backed by FakeDevice instead of a real /dev/videoN."""

    def __init__(self, device: FakeDevice) -> None:
        self.device = device
        super().__init__(video=FakeCapture(device), camera_id=0)  # type: ignore[arg-type]

    def _open_capture(self) -> Optional[FakeCapture]:  # type: ignore[override]
        return FakeCapture(self.device)


@pytest.fixture
def device() -> FakeDevice:
    return FakeDevice()


@pytest.fixture
def camera(device: FakeDevice):
    cam = FakeVideoCamera(device)
    yield cam
    cam.disable()


class TestVideoCameraToggle:
    def test_camera_holds_the_device_while_enabled(self, camera, device) -> None:
        assert camera.is_active
        assert not camera.is_disabled
        assert not device.is_free

    def test_disable_releases_the_device(self, camera, device) -> None:
        camera.disable()

        assert not camera.is_active
        assert camera.is_disabled
        assert camera.video is None
        assert device.is_free, "another process must be able to open the device"

    def test_disable_stops_the_capture_thread(self, camera) -> None:
        camera.disable()

        assert not camera.is_alive()

    def test_enable_reacquires_the_device(self, camera, device) -> None:
        camera.disable()

        assert camera.enable() is True
        assert camera.is_active
        assert not camera.is_disabled
        assert not device.is_free

    def test_frames_flow_again_after_enable(self, camera) -> None:
        camera.disable()
        camera.enable()
        camera.join(timeout=0)  # let the fresh capture thread produce a frame
        _wait_for_frame(camera)

        assert camera.get_rgb_frame() is not None

    def test_repeated_toggles_keep_working(self, camera, device) -> None:
        for _ in range(3):
            camera.disable()
            assert device.is_free
            assert camera.enable() is True
            assert not device.is_free

    def test_enable_is_idempotent(self, camera, device) -> None:
        assert camera.enable() is True
        assert camera.is_active
        assert not device.is_free

    def test_disable_is_idempotent(self, camera, device) -> None:
        camera.disable()
        camera.disable()

        assert camera.is_disabled
        assert device.is_free

    def test_enable_fails_cleanly_when_device_is_taken(self, camera, device) -> None:
        camera.disable()
        blocker = FakeCapture(device)
        assert blocker.isOpened()

        assert camera.enable() is False
        assert not camera.is_active
        assert camera.video is None, "a failed enable must not leak a capture handle"

        blocker.release()
        assert camera.enable() is True, "must recover once the device is free"

    def test_disabled_camera_reports_no_frames(self, camera) -> None:
        camera.disable()

        assert camera.get_rgb_frame() is None


class TestDummyCameraToggle:
    def test_dummy_camera_round_trip(self) -> None:
        cam = DummyCamera(camera_type="dummy")
        assert cam.is_active

        cam.disable()
        assert not cam.is_active
        assert cam.is_disabled

        assert cam.enable() is True
        assert cam.is_active
        assert not cam.is_disabled
        assert cam.get_rgb_frame() is not None

        cam.disable()

    def test_dummy_camera_never_opens_a_real_device(self) -> None:
        """A dummy camera must not grab /dev/video0 when it is re-enabled."""
        cam = DummyCamera(camera_type="dummy")
        cam.disable()
        cam.enable()

        capture = cam._open_capture()
        assert capture is not None
        assert not capture.isOpened(), "dummy capture must not be bound to a device"

        cam.disable()


def _wait_for_frame(camera: VideoCamera, timeout: float = 2.0) -> None:
    import time

    deadline = time.perf_counter() + timeout
    while camera.last_frame is None and time.perf_counter() < deadline:
        time.sleep(0.01)
