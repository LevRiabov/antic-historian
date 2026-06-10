from pathlib import Path

import pytest

from ahx.config import Settings

ENV_VARS = ("AHX_ENV", "AHX_DATABASE_URL")


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Isolate from any local .env file (cwd-relative) and AHX_* env vars."""
    monkeypatch.chdir(tmp_path)
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.mark.usefixtures("clean_env")
def test_defaults() -> None:
    settings = Settings()
    assert settings.env == "dev"
    assert settings.database_url.startswith("postgresql+psycopg://")


@pytest.mark.usefixtures("clean_env")
def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AHX_ENV", "prod")
    monkeypatch.setenv("AHX_DATABASE_URL", "postgresql+psycopg://u:p@db:5432/x")
    settings = Settings()
    assert settings.env == "prod"
    assert settings.database_url == "postgresql+psycopg://u:p@db:5432/x"
