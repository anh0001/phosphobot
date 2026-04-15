import asyncio
import importlib.util
from pathlib import Path

from phosphobot.models import WandBTokenRequest

PAGES_MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "phosphobot" / "endpoints" / "pages.py"
)
PAGES_SPEC = importlib.util.spec_from_file_location(
    "phosphobot_pages_for_tests", PAGES_MODULE_PATH
)
assert PAGES_SPEC is not None and PAGES_SPEC.loader is not None
pages = importlib.util.module_from_spec(PAGES_SPEC)
PAGES_SPEC.loader.exec_module(pages)


def test_submit_wandb_token_accepts_modern_wandb_v1_token(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setattr(pages, "get_home_app_path", lambda: tmp_path)
    token = (
        "wandb_v1_B5rTnuVKdh6Rx7v50ZKuIB7HD22_eERHrxtgzOANtSCr9fd6FM94bp2Sa4rya3q"
        "CsMsoC5M10OxTZ"
    )

    response = asyncio.run(pages.submit_wandb_token(WandBTokenRequest(token=token)))

    assert response.status == "ok"
    assert (tmp_path / "wandb.token").read_text() == token


def test_submit_wandb_token_accepts_legacy_40_character_token(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setattr(pages, "get_home_app_path", lambda: tmp_path)
    token = "a" * 40

    response = asyncio.run(pages.submit_wandb_token(WandBTokenRequest(token=token)))

    assert response.status == "ok"
    assert (tmp_path / "wandb.token").read_text() == token


def test_submit_wandb_token_removes_existing_token_file_for_blank_input(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setattr(pages, "get_home_app_path", lambda: tmp_path)
    token_path = tmp_path / "wandb.token"
    token_path.write_text("existing-token")

    response = asyncio.run(pages.submit_wandb_token(WandBTokenRequest(token="")))

    assert response.status == "ok"
    assert not token_path.exists()


def test_submit_wandb_token_strips_surrounding_whitespace_before_saving(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setattr(pages, "get_home_app_path", lambda: tmp_path)
    token = "wandb_v1_example_token"

    response = asyncio.run(
        pages.submit_wandb_token(WandBTokenRequest(token=f"  {token}\n"))
    )

    assert response.status == "ok"
    assert (tmp_path / "wandb.token").read_text() == token
