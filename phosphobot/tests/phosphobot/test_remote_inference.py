"""Tests for remote SmolVLA inference over Tailscale.

Covers:
- Admin settings round-trip with new inference fields
- URL validation for Tailscale-style URLs
- _resolve_inference_mode resolution order
- setup_ai_control mode routing (remote_url vs modal)
"""

import pytest

from phosphobot.ai_control import (
    _resolve_inference_mode,
    validate_remote_inference_url,
)
from phosphobot.models import (
    AdminSettingsRequest,
    AdminSettingsResponse,
    StartAIControlRequest,
)


# ── URL Validation ──────────────────────────────────────────────────


class TestValidateRemoteInferenceUrl:
    def test_valid_tailscale_ip(self) -> None:
        url = validate_remote_inference_url("http://100.64.0.10:8080")
        assert url == "http://100.64.0.10:8080"

    def test_valid_tailscale_magicdns(self) -> None:
        url = validate_remote_inference_url("http://gpu-box.tailnet.ts.net:8080")
        assert url == "http://gpu-box.tailnet.ts.net:8080"

    def test_valid_https(self) -> None:
        url = validate_remote_inference_url("https://gpu.example.com:443")
        assert url == "https://gpu.example.com:443"

    def test_valid_private_ip(self) -> None:
        url = validate_remote_inference_url("http://192.168.1.100:8080")
        assert url == "http://192.168.1.100:8080"

    def test_trailing_slash_stripped(self) -> None:
        url = validate_remote_inference_url("http://100.64.0.10:8080/")
        assert url == "http://100.64.0.10:8080"

    def test_empty_url_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            validate_remote_inference_url("")

    def test_missing_scheme_raises(self) -> None:
        with pytest.raises(ValueError, match="http:// or https://"):
            validate_remote_inference_url("100.64.0.10:8080")

    def test_ftp_scheme_raises(self) -> None:
        with pytest.raises(ValueError, match="http:// or https://"):
            validate_remote_inference_url("ftp://100.64.0.10:8080")

    def test_missing_host_raises(self) -> None:
        with pytest.raises(ValueError, match="explicit host"):
            validate_remote_inference_url("http://")

    def test_no_port_ok(self) -> None:
        url = validate_remote_inference_url("http://gpu-box.tailnet.ts.net")
        assert url == "http://gpu-box.tailnet.ts.net"


# ── Inference Mode Resolution ───────────────────────────────────────


class TestResolveInferenceMode:
    def test_explicit_request_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from phosphobot import configs

        monkeypatch.setattr(configs.config, "DEFAULT_AI_INFERENCE_MODE", "modal")
        mode, url = _resolve_inference_mode("remote_url", "http://1.2.3.4:8080")
        assert mode == "remote_url"
        assert url == "http://1.2.3.4:8080"

    def test_falls_back_to_saved_setting(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from phosphobot import configs

        monkeypatch.setattr(configs.config, "DEFAULT_AI_INFERENCE_MODE", "remote_url")
        monkeypatch.setattr(
            configs.config, "DEFAULT_AI_REMOTE_INFERENCE_BASE_URL", "http://saved:8080"
        )
        mode, url = _resolve_inference_mode(None, None)
        assert mode == "remote_url"
        assert url == "http://saved:8080"

    def test_defaults_to_modal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from phosphobot import configs

        monkeypatch.setattr(configs.config, "DEFAULT_AI_INFERENCE_MODE", "modal")
        mode, url = _resolve_inference_mode(None, None)
        assert mode == "modal"
        assert url == ""


# ── Admin Settings Models ───────────────────────────────────────────


class TestAdminSettingsModels:
    def test_request_defaults(self) -> None:
        req = AdminSettingsRequest(
            dataset_name="test",
            episode_format="lerobot_v2.1",
            freq=30,
            video_codec="avc1",
            video_size=[320, 240],
            task_instruction="None",
        )
        assert req.ai_inference_mode == "modal"
        assert req.ai_remote_inference_base_url == ""
        assert req.ai_remote_inference_timeout_seconds == 30

    def test_request_with_remote_url(self) -> None:
        req = AdminSettingsRequest(
            dataset_name="test",
            episode_format="lerobot_v2.1",
            freq=30,
            video_codec="avc1",
            video_size=[320, 240],
            task_instruction="None",
            ai_inference_mode="remote_url",
            ai_remote_inference_base_url="http://100.64.0.10:8080",
            ai_remote_inference_timeout_seconds=60,
        )
        assert req.ai_inference_mode == "remote_url"
        assert req.ai_remote_inference_base_url == "http://100.64.0.10:8080"
        assert req.ai_remote_inference_timeout_seconds == 60

    def test_response_defaults(self) -> None:
        resp = AdminSettingsResponse(
            dataset_name="test",
            freq=30,
            episode_format="lerobot_v2.1",
            video_codec="avc1",
            video_size=[320, 240],
            task_instruction="None",
            cameras_to_record=None,
            hf_private_mode=False,
        )
        assert resp.ai_inference_mode == "modal"
        assert resp.ai_remote_inference_base_url == ""
        assert resp.ai_remote_inference_timeout_seconds == 30

    def test_round_trip(self) -> None:
        req = AdminSettingsRequest(
            dataset_name="ds",
            episode_format="lerobot_v2.1",
            freq=30,
            video_codec="avc1",
            video_size=[320, 240],
            task_instruction="pick up",
            ai_inference_mode="remote_url",
            ai_remote_inference_base_url="http://gpu-box.tailnet.ts.net:8080",
            ai_remote_inference_timeout_seconds=45,
        )
        data = req.model_dump()
        resp = AdminSettingsResponse(cameras_to_record=None, hf_private_mode=False, **{
            k: v for k, v in data.items()
            if k not in ("cameras_to_record", "hf_private_mode")
        })
        assert resp.ai_inference_mode == "remote_url"
        assert resp.ai_remote_inference_base_url == "http://gpu-box.tailnet.ts.net:8080"
        assert resp.ai_remote_inference_timeout_seconds == 45


# ── StartAIControlRequest ───────────────────────────────────────────


class TestStartAIControlRequest:
    def test_inference_fields_optional(self) -> None:
        req = StartAIControlRequest(
            model_id="test/model",
            model_type="smolvla",
            angle_format="rad",
        )
        assert req.inference_mode is None
        assert req.inference_base_url is None

    def test_inference_fields_set(self) -> None:
        req = StartAIControlRequest(
            model_id="test/model",
            model_type="smolvla",
            angle_format="rad",
            inference_mode="remote_url",
            inference_base_url="http://100.64.0.10:8080",
        )
        assert req.inference_mode == "remote_url"
        assert req.inference_base_url == "http://100.64.0.10:8080"

    def test_non_smolvla_model_types_still_valid(self) -> None:
        """Non-smolvla types can still set inference_mode — the backend
        validates this at runtime, not at model parse time."""
        req = StartAIControlRequest(
            model_id="test/model",
            model_type="ACT",
            angle_format="rad",
            inference_mode="remote_url",
        )
        assert req.inference_mode == "remote_url"
