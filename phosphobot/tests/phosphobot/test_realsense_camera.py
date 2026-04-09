"""Tests for RealSense background capture loop and virtual camera caching."""

import threading
import time
from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers to build a fake pyrealsense2 module
# ---------------------------------------------------------------------------


def _make_fake_rs_module(
    num_devices: int = 1,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
):
    """Return a SimpleNamespace that quacks like ``pyrealsense2``."""

    class FakeFrame:
        def __init__(self, data: np.ndarray):
            self._data = data

        def get_data(self) -> np.ndarray:
            return self._data

    class FakeFrameset:
        """Returned by pipeline.wait_for_frames()."""

        def __init__(self) -> None:
            # BGR colour frame
            self._color = FakeFrame(
                np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            )
            # Depth frame (single-channel uint16-like, but stored as 3-ch for cvtColor compat)
            self._depth = FakeFrame(
                np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            )

        def get_color_frame(self):  # noqa: ANN201
            return self._color

        def get_depth_frame(self):  # noqa: ANN201
            return self._depth

    class FakeVideoProfile:
        def width(self) -> int:
            return width

        def height(self) -> int:
            return height

        def fps(self) -> int:
            return fps

    class FakeStream:
        def as_video_stream_profile(self):  # noqa: ANN201
            return FakeVideoProfile()

    class FakeProfile:
        def get_stream(self, _stream_type):  # noqa: ANN001, ANN201
            return FakeStream()

    class FakePipeline:
        def __init__(self) -> None:
            self._started = False
            self._call_count = 0

        def start(self, _config) -> None:  # noqa: ANN001
            self._started = True

        def stop(self) -> None:
            self._started = False

        def get_active_profile(self):  # noqa: ANN201
            return FakeProfile()

        def wait_for_frames(self, timeout_ms: int = 5000):  # noqa: ANN201
            self._call_count += 1
            return FakeFrameset()

    class FakeDevice:
        def __init__(self, serial: str, name: str = "FakeRS"):
            self._serial = serial
            self._name = name

        def get_info(self, info_type):  # noqa: ANN001, ANN201
            if info_type == "serial_number":
                return self._serial
            return self._name

    class FakeDeviceList:
        def __init__(self, devices):  # noqa: ANN001
            self._devices = devices

        def size(self) -> int:
            return len(self._devices)

        def __getitem__(self, idx: int):  # noqa: ANN204
            return self._devices[idx]

    class FakeContext:
        def query_devices(self):  # noqa: ANN201
            serials = [f"SN{i:04d}" for i in range(num_devices)]
            return FakeDeviceList(
                [FakeDevice(s, f"FakeRS-{i}") for i, s in enumerate(serials)]
            )

    class FakeConfig:
        def enable_device(self, _serial: str) -> None:
            pass

        def enable_stream(self, **_kwargs) -> None:  # noqa: ANN003
            pass

    # Build the namespace that mimics ``import pyrealsense2 as rs``
    rs = SimpleNamespace(
        pipeline=FakePipeline,
        config=FakeConfig,
        context=FakeContext,
        camera_info=SimpleNamespace(serial_number="serial_number", name="name"),
        stream=SimpleNamespace(color="color", depth="depth"),
        format=SimpleNamespace(bgr8="bgr8", z16="z16"),
    )
    return rs, FakePipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_rs():
    """Patch ``pyrealsense2`` and import ``RealSenseCamera`` under the fake."""
    rs_mod, FakePipeline = _make_fake_rs_module(num_devices=1)

    with patch.dict("sys.modules", {"pyrealsense2": rs_mod}):
        # We need to re-evaluate the camera module's try/except block.
        # The simplest way is to manually construct RealSenseCamera using the
        # already-imported class but with the patched rs.
        from phosphobot.camera import RealSenseCamera  # type: ignore[attr-defined]

        # Monkey-patch the module-level ``rs`` that the class captured at import
        import phosphobot.camera as cam_mod

        original_rs = getattr(cam_mod, "rs", None)
        cam_mod.rs = rs_mod  # type: ignore[attr-defined]

        yield rs_mod, RealSenseCamera, FakePipeline

        # Restore
        if original_rs is not None:
            cam_mod.rs = original_rs  # type: ignore[attr-defined]


@pytest.fixture()
def fake_rs_two_devices():
    rs_mod, FakePipeline = _make_fake_rs_module(num_devices=2)

    with patch.dict("sys.modules", {"pyrealsense2": rs_mod}):
        from phosphobot.camera import RealSenseCamera  # type: ignore[attr-defined]

        import phosphobot.camera as cam_mod

        original_rs = getattr(cam_mod, "rs", None)
        cam_mod.rs = rs_mod  # type: ignore[attr-defined]

        yield rs_mod, RealSenseCamera, FakePipeline

        if original_rs is not None:
            cam_mod.rs = original_rs  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBackgroundCaptureLoop:
    """Verify that the background reader caches frames correctly."""

    def test_single_reader_services_rgb_and_depth(self, fake_rs):
        """One background reader should service repeated get_rgb/get_depth calls."""
        rs_mod, RealSenseCamera, _ = fake_rs

        cam = RealSenseCamera(device_serial="SN0000", device_index=0)
        assert cam.is_active
        assert cam._capture_thread is not None
        assert cam._capture_thread.is_alive()

        # Give the background thread a moment to cache at least one frame
        time.sleep(0.3)

        rgb = cam.get_rgb_frame()
        assert rgb is not None
        assert rgb.shape[2] == 3  # 3-channel RGB

        depth = cam.get_depth_frame()
        assert depth is not None

        # Repeated calls should also succeed (reading from cache)
        for _ in range(5):
            assert cam.get_rgb_frame() is not None
            assert cam.get_depth_frame() is not None

        cam.stop()
        assert not cam.is_active

    def test_no_extra_wait_for_frames_per_consumer(self, fake_rs):
        """Multiple get_rgb_frame() calls should NOT trigger extra
        wait_for_frames() — only the background thread calls it."""
        rs_mod, RealSenseCamera, _ = fake_rs

        cam = RealSenseCamera(device_serial="SN0000", device_index=0)
        time.sleep(0.3)

        # Record the current call count
        count_before = cam.pipeline._call_count

        # Call get_rgb_frame and get_depth_frame many times
        for _ in range(20):
            cam.get_rgb_frame()
            cam.get_depth_frame()

        # The call count should NOT have jumped by 40 (20+20).
        # Only the background thread increments it, so the delta should be
        # small (a few iterations of the capture loop at most).
        count_after = cam.pipeline._call_count
        consumer_calls = 40
        bg_delta = count_after - count_before
        assert bg_delta < consumer_calls, (
            f"wait_for_frames was called {bg_delta} extra times — "
            "consumers should read from cache, not the pipeline"
        )

        cam.stop()

    def test_stop_joins_capture_thread(self, fake_rs):
        """stop() should terminate the background capture thread."""
        _, RealSenseCamera, _ = fake_rs

        cam = RealSenseCamera(device_serial="SN0000", device_index=0)
        thread = cam._capture_thread
        assert thread is not None

        cam.stop()
        assert not thread.is_alive()

    def test_get_rgb_frame_with_resize(self, fake_rs):
        """get_rgb_frame should support an optional resize parameter."""
        _, RealSenseCamera, _ = fake_rs

        cam = RealSenseCamera(device_serial="SN0000", device_index=0)
        time.sleep(0.3)

        frame = cam.get_rgb_frame(resize=(320, 240))
        assert frame is not None
        assert frame.shape == (240, 320, 3)

        cam.stop()


class TestVirtualCamerasCaching:
    """RealSenseVirtualCamera should read from the parent's cache."""

    def test_virtual_cameras_use_parent_cache(self, fake_rs):
        _, RealSenseCamera, _ = fake_rs
        from phosphobot.camera import RealSenseVirtualCamera

        parent = RealSenseCamera(device_serial="SN0000", device_index=0)
        time.sleep(0.3)

        rgb_virtual = RealSenseVirtualCamera(parent, "rgb", camera_id=10)
        depth_virtual = RealSenseVirtualCamera(parent, "depth", camera_id=11)

        assert rgb_virtual.get_rgb_frame() is not None
        assert depth_virtual.get_rgb_frame() is not None  # depth virtual returns colorized depth

        parent.stop()

    def test_multiple_virtual_cameras_no_extra_pipeline_calls(self, fake_rs):
        _, RealSenseCamera, _ = fake_rs
        from phosphobot.camera import RealSenseVirtualCamera

        parent = RealSenseCamera(device_serial="SN0000", device_index=0)
        time.sleep(0.3)

        count_before = parent.pipeline._call_count

        v1 = RealSenseVirtualCamera(parent, "rgb", camera_id=10)
        v2 = RealSenseVirtualCamera(parent, "depth", camera_id=11)

        for _ in range(10):
            v1.get_rgb_frame()
            v2.get_rgb_frame()

        count_after = parent.pipeline._call_count
        bg_delta = count_after - count_before
        # Only the background thread should have called wait_for_frames
        assert bg_delta < 20

        parent.stop()


class TestTwoDevicesNonBlocking:
    """Two RealSense devices should each have their own capture loop."""

    def test_two_devices_independent_readers(self, fake_rs_two_devices):
        rs_mod, RealSenseCamera, _ = fake_rs_two_devices

        cam0 = RealSenseCamera(device_serial="SN0000", device_index=0)
        cam1 = RealSenseCamera(device_serial="SN0001", device_index=1)

        time.sleep(0.3)

        assert cam0.is_active
        assert cam1.is_active

        # Both should return frames independently
        assert cam0.get_rgb_frame() is not None
        assert cam1.get_rgb_frame() is not None

        cam0.stop()
        cam1.stop()
