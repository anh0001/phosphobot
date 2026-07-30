import importlib.util
from pathlib import Path
from typing import AsyncGenerator

from starlette.requests import Request


def _load_camera_endpoint_module():
    module_path = (
        Path(__file__).resolve().parents[2] / "phosphobot" / "endpoints" / "camera.py"
    )
    spec = importlib.util.spec_from_file_location("camera_endpoint_module", module_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


camera_endpoint_module = _load_camera_endpoint_module()


class _DummyCamera:
    is_active = True
    is_disabled = False

    async def generate_rgb_frames(
        self,
        target_size: tuple[int, int] | None,
        quality: int | None,
        request: Request | None,
    ) -> AsyncGenerator[bytes, None]:
        del target_size, quality, request
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\nframe\r\n"


class _DummyCameras:
    def get_camera_by_id(self, camera_id: int) -> _DummyCamera | None:
        if camera_id == 0:
            return _DummyCamera()
        return None


def test_video_feed_sets_no_cache_headers() -> None:
    request = Request({"type": "http", "method": "GET", "path": "/video/0"})
    response = camera_endpoint_module.video_feed_for_camera(
        request=request,
        camera_id=0,
        cameras=_DummyCameras(),
    )

    assert response.headers["cache-control"] == (
        "no-store, no-cache, must-revalidate, max-age=0"
    )
    assert response.headers["pragma"] == "no-cache"
    assert response.headers["expires"] == "0"
