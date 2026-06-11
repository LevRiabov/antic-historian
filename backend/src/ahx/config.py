"""Application settings — single source of configuration truth.

All values can be overridden via environment variables with the AHX_ prefix
(e.g. AHX_DATABASE_URL), or a local .env file. Validated at startup.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/src/ahx/config.py -> repo root is 3 levels above the package dir.
_REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AHX_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: str = "dev"
    database_url: str = "postgresql+psycopg://ahx:ahx@localhost:5433/ahx"

    # Embedding (provisional model pending gate D2; served by local llama-swap).
    embed_base_url: str = "http://127.0.0.1:8080/v1"
    embed_model: str = "qwen3-embedding-0.6b"
    embed_dim: int = 1024
    embed_batch_size: int = 16

    # Corpus locations (downloaded texts are gitignored; manifest is committed).
    corpus_dir: Path = _REPO_ROOT / "corpus"

    @property
    def manifest_path(self) -> Path:
        return self.corpus_dir / "ai_historian_corpus_eu_pd.txt"

    @property
    def corpus_raw_dir(self) -> Path:
        return self.corpus_dir / "raw"

    @property
    def corpus_normalized_dir(self) -> Path:
        return self.corpus_dir / "normalized"

    # Observability (Phase 6) — optional until wired.
    langfuse_host: str | None = None
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
