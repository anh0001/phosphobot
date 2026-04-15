"""Standalone SmolVLA inference server.

Run on a GPU machine reachable over Tailscale (or any network).
Usage:
    phosphobot serve-smolvla --model-id <hf-or-local-path> --host 0.0.0.0 --port 8080

Exposes:
    GET  /health  – readiness probe
    POST /act     – inference (same payload shape as the Modal LeRobot server)
"""

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import uvicorn
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel


class InferenceRequest(BaseModel):
    encoded: str  # json_numpy encoded dict


def _load_smolvla_policy(model_path: str, device: str) -> nn.Module:
    """Load a SmolVLA policy from a HuggingFace repo ID or local path."""
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # type: ignore

    logger.info(f"Loading SmolVLA policy from {model_path} on device={device}")
    start = time.time()
    policy = SmolVLAPolicy.from_pretrained(model_path)
    policy = policy.to(device=device)
    policy.eval()
    logger.success(f"SmolVLA policy loaded in {time.time() - start:.1f}s")
    return policy


def _resolve_device() -> str:
    """Select the best available device with a warning for CPU."""
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        logger.warning(
            "CUDA is not available — running on CPU. "
            "Inference will be significantly slower."
        )
    return device


def create_smolvla_app(model_path: str, device: str | None = None) -> FastAPI:
    """Create a FastAPI app serving a SmolVLA policy.

    This is the reusable core used by both the CLI command and tests.
    """
    import json_numpy  # type: ignore

    if device is None:
        device = _resolve_device()

    policy = _load_smolvla_policy(model_path, device)

    app = FastAPI(title="SmolVLA Inference Server")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/act")
    async def act(request: InferenceRequest) -> Any:
        try:
            payload: dict = json_numpy.loads(request.encoded)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid encoded payload: {e}")

        # Extract state key (the first key matching observation.state pattern)
        state_key = None
        for key in payload:
            if "state" in key:
                state_key = key
                break
        if state_key is None:
            raise HTTPException(
                status_code=400,
                detail="No state key found in payload. Expected a key containing 'state'.",
            )

        current_qpos = payload[state_key]

        # Extract image keys
        image_keys = [k for k in payload if "image" in k or "camera" in k]
        images = [payload[k] for k in image_keys]

        # Extract prompt
        prompt = payload.get("prompt", payload.get("task", ""))
        if not prompt:
            raise HTTPException(
                status_code=400,
                detail="Prompt is required for SmolVLA inference. "
                "Include a 'prompt' or 'task' key in the payload.",
            )

        # Build batch
        batch: dict[str, Any] = {}
        batch[state_key] = torch.tensor(
            current_qpos, dtype=torch.float32, device=device
        ).unsqueeze(0)

        for i, img in enumerate(images):
            if not isinstance(img, np.ndarray):
                img = np.array(img)
            img_tensor = torch.from_numpy(img).float() / 255.0
            if img_tensor.dim() == 3 and img_tensor.shape[2] == 3:
                img_tensor = img_tensor.permute(2, 0, 1)
            batch[image_keys[i]] = img_tensor.unsqueeze(0).to(device)

        batch["task"] = prompt

        try:
            with torch.no_grad():
                if device == "cuda":
                    with torch.autocast(device_type="cuda"):
                        actions = policy.predict_action_chunk(batch)
                else:
                    actions = policy.predict_action_chunk(batch)
            result = actions.cpu().numpy()
        except Exception as e:
            logger.error(f"Inference error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

        return json_numpy.dumps(result)

    return app


def run_server(
    model_id: str,
    host: str = "0.0.0.0",
    port: int = 8080,
) -> None:
    """Entry point for the CLI command."""
    app = create_smolvla_app(model_id)
    logger.info(f"Starting SmolVLA server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
