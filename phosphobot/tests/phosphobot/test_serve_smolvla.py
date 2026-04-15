"""Tests for the standalone SmolVLA inference server.

Tests the FastAPI app created by create_smolvla_app() with a mocked policy.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not installed")
pytest.importorskip("json_numpy", reason="json_numpy not installed")

from fastapi.testclient import TestClient  # noqa: E402


def _make_mock_policy() -> MagicMock:
    """Create a mock policy that returns a plausible action chunk."""
    policy = MagicMock()
    # predict_action_chunk returns a tensor of shape (1, action_dim)
    policy.predict_action_chunk.return_value = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]
    )
    return policy


@pytest.fixture()
def client() -> TestClient:
    """Create a test client with a mocked SmolVLA policy."""
    mock_policy = _make_mock_policy()

    with patch(
        "phosphobot.serve_smolvla._load_smolvla_policy", return_value=mock_policy
    ):
        from phosphobot.serve_smolvla import create_smolvla_app

        app = create_smolvla_app(model_path="mock/model", device="cpu")
        return TestClient(app)


class TestHealthEndpoint:
    def test_health_ok(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


class TestActEndpoint:
    def _make_payload(self) -> dict:
        """Create a valid /act payload."""
        import json_numpy  # type: ignore

        inputs = {
            "observation.state": np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
            "observation.images.wrist_camera": np.random.randint(
                0, 255, (224, 224, 3), dtype=np.uint8
            ),
            "task": "pick up the block",
        }
        return {"encoded": json_numpy.dumps(inputs)}

    def test_act_returns_actions(self, client: TestClient) -> None:
        payload = self._make_payload()
        response = client.post("/act", json=payload)
        assert response.status_code == 200

    def test_act_missing_state_key(self, client: TestClient) -> None:
        import json_numpy  # type: ignore

        inputs = {
            "observation.images.wrist_camera": np.random.randint(
                0, 255, (224, 224, 3), dtype=np.uint8
            ),
            "task": "pick up",
        }
        payload = {"encoded": json_numpy.dumps(inputs)}
        response = client.post("/act", json=payload)
        assert response.status_code == 400
        assert "state" in response.json()["detail"].lower()

    def test_act_missing_prompt(self, client: TestClient) -> None:
        import json_numpy  # type: ignore

        inputs = {
            "observation.state": np.array([0.0, 0.1, 0.2]),
        }
        payload = {"encoded": json_numpy.dumps(inputs)}
        response = client.post("/act", json=payload)
        assert response.status_code == 400
        assert "prompt" in response.json()["detail"].lower()

    def test_act_invalid_encoded(self, client: TestClient) -> None:
        payload = {"encoded": "not-valid-json"}
        response = client.post("/act", json=payload)
        assert response.status_code == 400
