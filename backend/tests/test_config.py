from pathlib import Path

import pytest
from pydantic import SecretStr

from ahx.config import Settings, validate_serving_config

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


@pytest.mark.usefixtures("clean_env")
def test_validate_serving_config_requires_key_for_hosted_endpoint() -> None:
    # Hosted embed/chat endpoints with no API key would 401 on the first real query
    # while booting green — fail loud at serve time instead.
    hosted_no_key = Settings(
        embed_base_url="https://openrouter.ai/api/v1",
        chat_base_url="https://openrouter.ai/api/v1",
        embed_api_key=None,
        chat_api_key=None,
    )
    with pytest.raises(RuntimeError, match="API key"):
        validate_serving_config(hosted_no_key)


@pytest.mark.usefixtures("clean_env")
def test_validate_serving_config_ok_for_local_or_keyed() -> None:
    local = Settings(
        embed_base_url="http://127.0.0.1:8080/v1",
        chat_base_url="http://localhost:8080/v1",
    )
    assert validate_serving_config(local) == []  # local endpoints need no key

    hosted_keyed = Settings(
        embed_base_url="https://openrouter.ai/api/v1",
        chat_base_url="https://openrouter.ai/api/v1",
        embed_api_key=SecretStr("sk-e"),
        chat_api_key=SecretStr("sk-c"),
    )
    assert validate_serving_config(hosted_keyed) == []


@pytest.mark.usefixtures("clean_env")
def test_validate_serving_config_warns_on_proxy_ip_trust_off_in_prod() -> None:
    prod = Settings(
        env="prod",
        embed_base_url="http://127.0.0.1:8080/v1",
        chat_base_url="http://127.0.0.1:8080/v1",
        rate_limit_per_window=20,
        trust_forwarded_for=False,
    )
    warnings = validate_serving_config(prod)
    assert any("trust_forwarded_for" in w for w in warnings)
