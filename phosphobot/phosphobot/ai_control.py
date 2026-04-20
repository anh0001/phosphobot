import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple
from urllib.parse import urlparse

import httpx
import numpy as np
from fastapi import HTTPException
from loguru import logger
from supabase import AsyncClient

from phosphobot.am.act import ACT
from phosphobot.am.gr00t import Gr00tN1, Gr00tSpawnConfig
from phosphobot.am.lerobot import LeRobotSpawnConfig
from phosphobot.am.pi05 import Pi05, Pi05SpawnConfig
from phosphobot.am.smolvla import SmolVLA
from phosphobot.camera import AllCameras
from phosphobot.configs import config
from phosphobot.control_signal import AIControlSignal
from phosphobot.hardware.base import BaseManipulator
from phosphobot.models import ServerInfoResponse
from phosphobot.supabase import get_client
from phosphobot.utils import get_tokens


def validate_remote_inference_url(url: str) -> str:
    """Validate and normalize a remote inference base URL.

    Accepts http:// or https:// URLs with an explicit host.
    Allows private IPs, Tailscale CGNAT-range IPs, and normal hostnames.
    Returns the normalized URL (without trailing slash).

    Raises:
        ValueError: If the URL is malformed or missing required parts.
    """
    if not url:
        raise ValueError("Remote inference base URL must not be empty.")

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"Remote inference URL must use http:// or https://, got '{parsed.scheme}://'."
        )
    if not parsed.hostname:
        raise ValueError("Remote inference URL must include an explicit host.")

    return url.rstrip("/")


class CustomAIControlSignal(AIControlSignal):
    _status: Literal["stopped", "running", "paused", "waiting"]
    _last_status_update: Optional[
        Literal["stopped", "running", "paused", "waiting"]
    ] = None
    _supabase_client: Optional[AsyncClient] = None

    def __init__(self) -> None:
        super().__init__()

    def _update_supabase(
        self, started_at: Optional[datetime] = None, ended_at: Optional[datetime] = None
    ) -> None:
        # schedule the real work in the running loop
        if self._status == self._last_status_update:
            return
        loop = asyncio.get_event_loop()
        loop.create_task(self._update_supabase_async(started_at, ended_at))

    async def _update_supabase_async(
        self, started_at: Optional[datetime] = None, ended_at: Optional[datetime] = None
    ) -> None:
        if self._supabase_client is None:
            self._supabase_client = await get_client()

        payload: Dict[str, Any] = {"status": self._status}
        if started_at is not None:
            payload["started_at"] = started_at.isoformat()
        if ended_at is not None:
            payload["ended_at"] = ended_at.isoformat()

        try:
            await (
                self._supabase_client.table("ai_control_sessions")
                .update(payload)
                .eq("id", self.id)
                .execute()
            )
            self._last_status_update = self._status
        except Exception as e:
            logger.warning(f"Error updating Supabase: {e}")
        return None

    def new_id(self) -> None:
        super().new_id()

    def start(self) -> None:
        with self._lock:
            self._is_in_loop = True
            self._status = "waiting"
            self._update_supabase()

    def set_running(self) -> None:
        with self._lock:
            self._is_in_loop = True
            self._status = "running"
            self._update_supabase(started_at=datetime.now(timezone.utc))

    def stop(self) -> None:
        with self._lock:
            self._is_in_loop = False
            self._status = "stopped"
            self._update_supabase(ended_at=datetime.now(timezone.utc))

    def is_in_loop(self) -> bool:
        with self._lock:
            return self._is_in_loop

    @property
    def status(self) -> Literal["stopped", "running", "paused", "waiting"]:
        return self._status

    @status.setter
    def status(self, value: Literal["stopped", "running", "paused", "waiting"]) -> None:
        if value == "stopped":
            self.stop()
        elif value == "running":
            self._status = value
            with self._lock:
                self._is_in_loop = True
        elif value == "paused":
            self._status = value
        elif value == "waiting":
            self._status = value
            with self._lock:
                self._is_in_loop = True

        self._update_supabase()


def _resolve_inference_mode(
    request_mode: Optional[Literal["modal", "remote_url"]],
    request_base_url: Optional[str],
) -> Tuple[Literal["modal", "remote_url"], str]:
    """Resolve the inference mode and base URL from request overrides or saved settings.

    Resolution order:
    1. Explicit request override
    2. Saved admin setting
    3. Fallback to "modal"

    Returns:
        Tuple of (mode, base_url). base_url is empty when mode is "modal".
    """
    mode: Literal["modal", "remote_url"] = (
        request_mode
        or config.DEFAULT_AI_INFERENCE_MODE  # type: ignore[assignment]
    )
    base_url = ""
    if mode == "remote_url":
        base_url = request_base_url or config.DEFAULT_AI_REMOTE_INFERENCE_BASE_URL
    return mode, base_url


async def setup_ai_control(
    robots: List[BaseManipulator],
    all_cameras: AllCameras,
    ai_control_signal_id: str,
    model_type: Literal["gr00t", "ACT", "ACT_BBOX", "pi0.5", "smolvla"],
    model_id: str = "PLB/GR00T-N1-lego-pickup-mono-2",
    cameras_keys_mapping: Optional[dict[str, int]] = None,
    init_connected_robots: bool = True,
    verify_cameras: bool = True,
    checkpoint: Optional[int] = None,
    inference_mode: Optional[Literal["modal", "remote_url"]] = None,
    inference_base_url: Optional[str] = None,
) -> Tuple[
    Gr00tN1 | ACT | Pi05 | SmolVLA,
    Gr00tSpawnConfig | Pi05SpawnConfig | LeRobotSpawnConfig,
    ServerInfoResponse,
]:
    """
    Setup the AI control loop by spawning the inference server and returning the model.
    This function is called when the user clicks on the "Start AI Control" button in the UI.
    """

    resolved_mode, resolved_base_url = _resolve_inference_mode(
        inference_mode, inference_base_url
    )

    model_types: Dict[str, type[ACT | Gr00tN1 | Pi05 | SmolVLA]] = {
        "gr00t": Gr00tN1,
        "ACT": ACT,
        "ACT_BBOX": ACT,
        "pi0.5": Pi05,
        "smolvla": SmolVLA,
    }

    try:
        model_used = model_types[model_type]
        model_spawn_config = model_used.fetch_and_verify_config(
            model_id=model_id,
            all_cameras=all_cameras,
            robots=robots,  # type: ignore
            cameras_keys_mapping=cameras_keys_mapping,
            verify_cameras=verify_cameras,
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Model verification failed for {model_type}: {e}",
        )

    # ── Remote URL mode ──────────────────────────────────────────────
    if resolved_mode == "remote_url":
        if model_type != "smolvla":
            raise HTTPException(
                status_code=400,
                detail=f"Remote URL inference is only supported for smolvla models, got '{model_type}'.",
            )

        try:
            validated_url = validate_remote_inference_url(resolved_base_url)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Verify the remote server is reachable and, when possible, that it
        # serves the model selected in the UI.
        timeout = config.DEFAULT_AI_REMOTE_INFERENCE_TIMEOUT_SECONDS
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                health = await client.get(f"{validated_url}/health")
                health.raise_for_status()
                health_payload = health.json()
        except Exception as e:
            raise HTTPException(
                status_code=502,
                detail=f"Remote inference server at {validated_url} is not reachable: {e}",
            )

        remote_model_id = None
        if isinstance(health_payload, dict):
            raw_remote_model = health_payload.get("model_id")
            if isinstance(raw_remote_model, str) and raw_remote_model.strip():
                remote_model_id = raw_remote_model.strip()

        if remote_model_id is None:
            logger.warning(
                "Remote inference server at '{}' did not report a model_id in /health. "
                "Unable to verify the requested model '{}' before starting control.",
                validated_url,
                model_id,
            )
        elif remote_model_id != model_id:
            logger.error(
                "Remote inference model mismatch at '{}': requested='{}', remote='{}'",
                validated_url,
                model_id,
                remote_model_id,
            )
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Remote inference server at {validated_url} is serving "
                    f"'{remote_model_id}', but phosphobot is configured to use "
                    f"'{model_id}'. Start the remote server with '--model-id {model_id}' "
                    "or select the matching model in the dashboard."
                ),
            )

        # Build a synthetic ServerInfoResponse (no Modal spawn needed)
        parsed = urlparse(validated_url)
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        server_info = ServerInfoResponse(
            url=validated_url,
            port=port,
            tcp_socket=(parsed.hostname or "", port),
            model_id=model_id,
            timeout=timeout,
        )

        model = model_types[model_type](
            server_url=validated_url,
            server_port=port,
            **model_spawn_config.model_dump(),
        )

        if init_connected_robots:
            logger.debug("Resetting robot to initial position")
            if len(robots) == 0:
                raise HTTPException(
                    status_code=400,
                    detail="No robot connected. Exiting AI control loop.",
                )
            for robot in robots:
                await robot.move_to_initial_position(open_gripper=True)

        return model, model_spawn_config, server_info

    # ── Modal mode (existing behavior) ───────────────────────────────
    tokens = get_tokens()
    if tokens.MODAL_API_URL is None:
        raise HTTPException(
            status_code=400,
            detail="Modal API key not found. Please check your configuration.",
        )

    supabase_client = await get_client()
    session = await supabase_client.auth.get_session()
    if session is None:
        raise HTTPException(
            status_code=401,
            detail="Session expired. Please log in again.",
        )

    def sanitize(o: Any) -> object:
        if isinstance(o, float):
            return 0.0 if (np.isnan(o) or np.isinf(o)) else o
        if isinstance(o, dict):
            return {k: sanitize(v) for k, v in o.items()}
        if isinstance(o, list):
            return [sanitize(v) for v in o]
        return o

    raw = model_spawn_config.model_dump()
    clean = sanitize(raw)

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            url=f"{tokens.MODAL_API_URL}/spawn",
            json={
                "model_id": model_id,
                "checkpoint": checkpoint,
                "model_type": model_type,
                "model_specifics": clean,
            },
            headers={
                "Authorization": f"Bearer {session.access_token}",
                "Content-Type": "application/json",
            },
        )

    if response.status_code != 200:
        logger.error(f"Failed to start inference server: {response.text}")
        await (
            supabase_client.table("ai_control_sessions")
            .update(
                {
                    "status": "stopped",
                }
            )
            .eq("id", ai_control_signal_id)
            .execute()
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start inference server: {response.text}",
        )

    server_info = ServerInfoResponse.model_validate(response.json())

    connects_through_tcp = ["gr00t", "pi0.5"]

    if model_type in connects_through_tcp:
        server_url = server_info.tcp_socket[0]
        server_port = server_info.tcp_socket[1]
    else:
        server_url = server_info.url
        server_port = server_info.port

    model = model_types[model_type](
        server_url=server_url,
        server_port=server_port,
        **model_spawn_config.model_dump(),
    )

    if init_connected_robots:
        logger.debug("Resetting robot to initial position")
        if len(robots) == 0:
            raise HTTPException(
                status_code=400,
                detail="No robot connected. Exiting AI control loop.",
            )
        for robot in robots:
            await robot.move_to_initial_position(open_gripper=True)

    return model, model_spawn_config, server_info
