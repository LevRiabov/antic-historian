"""Application settings — single source of configuration truth.

All values can be overridden via environment variables with the AHX_ prefix
(e.g. AHX_DATABASE_URL), or a local .env file. Validated at startup.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AHX_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: str = "dev"
    database_url: str = "postgresql+psycopg://ahx:ahx@localhost:5432/ahx"

    # Observability (Phase 6) — optional until wired.
    langfuse_host: str | None = None
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
