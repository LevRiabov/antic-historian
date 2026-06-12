"""Application settings — single source of configuration truth.

All values can be overridden via environment variables with the AHX_ prefix
(e.g. AHX_DATABASE_URL), or a local .env file. Validated at startup.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/src/ahx/config.py -> repo root is 3 levels above the package dir.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_BACKEND_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AHX_",
        # Absolute, not ".": a relative env_file resolves against the CWD,
        # so the CLI would silently skip .env when run outside backend/.
        env_file=_BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: str = "dev"
    database_url: str = "postgresql+psycopg://ahx:ahx@localhost:5433/ahx"

    # Embedding — gate D2 DECIDED 2026-06-12 (ADR-002): qwen3-embedding-8b
    # hosted via OpenRouter pinned to Nebius, MRL-truncated to 1024 dims.
    # Defaults match the decision so a missing API key fails LOUDLY — never
    # silently embed queries with a different model than the corpus (0.6b at
    # 1024d would produce garbage similarities with no error). Local fallback
    # (qwen3-embedding-0.6b on llama-swap, -18 recall@5): override via env.
    # Changing model/dim/provider requires `ahx db reset-chunks` +
    # `ahx ingest load` + `ahx ingest parity --update`.
    embed_base_url: str = "https://openrouter.ai/api/v1"
    embed_model: str = "qwen/qwen3-embedding-8b"
    embed_dim: int = 1024
    embed_batch_size: int = 32
    embed_api_key: str | None = None  # required for the hosted default
    # OpenRouter provider pinning: reproducible vectors (one runtime, one
    # quantization) + no slow-provider roulette. None = provider's default.
    embed_provider: str | None = "nebius"
    # MRL truncation (docs/embeddings.md footgun #4): vectors longer than
    # embed_dim are truncated + L2-renormalized. Only for MRL-trained models.
    embed_mrl_truncate: bool = True

    # Chat LLM (gate D5 open — any OpenAI-compatible endpoint; provisional
    # default is local gemma via llama-swap, same server as embeddings).
    chat_base_url: str = "http://127.0.0.1:8080/v1"
    chat_model: str = "gemma-12b-16k"
    chat_api_key: str | None = None

    # Judge LLM (eval generation tier, phase boundaries). Unset = judge
    # layer unavailable; use a strong model — weak judges miscalibrate.
    judge_base_url: str | None = None
    judge_model: str | None = None
    judge_api_key: str | None = None

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
