"""Application settings — single source of configuration truth.

All values can be overridden via environment variables with the AHX_ prefix
(e.g. AHX_DATABASE_URL), or a local .env file. Validated at startup.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChatEndpoint(BaseModel):
    """One OpenAI-compatible chat endpoint in the fallback lineup (6.4 / D5). Distinct
    provider+model from the primary so a single outage isn't total. Parsed from JSON in
    AHX_CHAT_FALLBACKS, e.g. '[{"base_url":"https://...","model":"...","api_key":"..."}]'."""

    base_url: str
    model: str
    api_key: str | None = None


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

    # Reranker (Phase 4.2 cross-encoder arm). Provider-agnostic over the
    # Cohere-compatible POST /rerank shape — BOTH local llama.cpp and OpenRouter's
    # /api/v1/rerank speak it, so swapping the local qwen3/bge reranker for a
    # hosted Cohere ceiling reference (cohere/rerank-v3.5, cohere/rerank-4-pro,
    # which ride the SAME OpenRouter base_url + key as embeddings) is three values,
    # zero code. Default = local llama-swap reranker: rerank is a query-time cost
    # forever (the expensive ledger), so $0-local is the default; hosted = ceiling.
    rerank_base_url: str = "http://127.0.0.1:8080/v1"
    rerank_model: str = "qwen3-reranker-0.6b"
    rerank_api_key: str | None = None
    rerank_provider: str | None = None
    # Dense candidate-pool depth fed to the reranker. The rerank-bait categories'
    # answers sit deep (cross-book recall@20 60.5%); N=50 default, with a one-off
    # N=100 sensitivity check before locking (phase-4-plan.md §4.2).
    rerank_pool_n: int = 50

    # Hybrid BM25+dense (Phase 4.3): depth pulled from EACH of the dense and sparse
    # lists before RRF fusion; rrf_k damps the long tail (1/(k+rank)). Standard k=60.
    hybrid_pool_n: int = 50
    rrf_k: int = 60

    # Chat LLM (gate D5 open — any OpenAI-compatible endpoint; provisional
    # default is local gemma via llama-swap, same server as embeddings).
    chat_base_url: str = "http://127.0.0.1:8080/v1"
    chat_model: str = "gemma-12b-16k"
    chat_api_key: str | None = None
    # Decoding temperature — pinned low for run-to-run stability (the agent's
    # answer/refuse decision is high-variance; eval-log 2026-06-16). 0.0 asks for
    # greedy decoding where the provider honours it; recorded as an explicit knob
    # rather than an invisible default so it is part of every run's config.
    chat_temperature: float = 0.0
    # Fallback lineup (6.4 / D5): ordered alternates wrapped with the primary in a
    # CompositeChatModel. Empty = no fallover (the bare primary serves). Distinct
    # providers so one outage != total outage; the served model rides `served_by` to
    # the SSE indicator + trace. JSON-encoded list of ChatEndpoint via AHX_CHAT_FALLBACKS.
    chat_fallbacks: list[ChatEndpoint] = []

    # Rate limiting + per-session cap (6.4) — load-bearing for a public, abusable demo
    # where every query is hosted spend (phase-6-plan §cost-of-a-query). In-memory,
    # single-instance (api/limits.py); Redis is the documented scale path. Both limits
    # disabled when set to 0.
    #   - IP sliding window: abuse protection, returns a structured 429 + Retry-After.
    #   - Per-session cap: free-tier protection keyed on a client X-Session-Id header;
    #     every allowed answer carries "N of M left" on the SSE sources event.
    rate_limit_per_window: int = 20  # max requests per IP per window (0 = off)
    rate_limit_window_seconds: int = 60
    session_query_cap: int = 30  # lifetime queries per session id (0 = off)
    # Behind a reverse proxy (prod), the real client IP is in X-Forwarded-For, not
    # request.client. OFF by default (locally the header is spoofable); enable in prod.
    trust_forwarded_for: bool = False

    # Judge LLM (eval generation tier, phase boundaries). Unset = judge
    # layer unavailable; use a strong model — weak judges miscalibrate.
    judge_base_url: str | None = None
    judge_model: str | None = None
    judge_api_key: str | None = None

    # Attribution judge — the attribution rubric is judge-noise-limited on a
    # flash-tier judge (0.66/question swing, full 1<->5 flips on multi-source;
    # eval-log 2026-06-15), itself a hard reasoning task. Route ONLY attribution
    # to a stronger model than the faith/compl judge. Unset = the main judge
    # scores attribution too (back-compat). A judge must be >= the GENERATED
    # model's tier — a weaker judge can't reliably grade a stronger model.
    attrib_judge_base_url: str | None = None
    attrib_judge_model: str | None = None
    attrib_judge_api_key: str | None = None

    # Enrichment LLM (Phase 4.1 contextual-note + metadata pass). Offline,
    # one-time, cached to corpus/enriched/ — so a local model is the cheap
    # default (gemma-12b-enrich = the parallel-slot llama-swap profile). To run
    # the pass hosted instead (deepseek-v4-flash ≈ $5/46k), override these three.
    enrich_base_url: str = "http://127.0.0.1:8080/v1"
    enrich_model: str = "gemma-12b-enrich"
    enrich_api_key: str | None = None
    # Concurrent in-flight calls. Kept BELOW the llama-swap profile's -np slot
    # count: gemma-12B on the 5070 Ti is compute-bound (~0.8 chunk/s, batching
    # peaks at ~3-4 concurrent), and going at/above the slot count trips
    # llama-swap 429 backpressure. 6 saturates the GPU with 2 slots of headroom.
    enrich_concurrency: int = 6
    # Output ceiling. The grammar stops at the natural JSON close, so a typical
    # reply is ~150-200 tokens regardless; this cap only bites a runaway note on
    # the corpus's densest chunks (geography catalogs, name indices). 256→512→1024
    # as those edge chunks kept hitting finish=length; 1024 clears them. (maxLength
    # on the note is ignored by this llama.cpp build, so the cap is the only guard.)
    enrich_max_tokens: int = 1024

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

    # Observability (Phase 6.1) — tracing is OPT-IN: all three must be set or
    # init_langfuse returns None and the API runs untraced (obs.py). For a local
    # self-hosted instance the host is http://localhost:3000; keys come from the
    # project's settings page (or LANGFUSE_INIT_* if the stack was bootstrapped).
    langfuse_host: str | None = None
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None

    # Security lab (Phase 6.3-lab) — a tripwire token seeded into the system prompt
    # so prompt-extraction is a deterministic test (did it appear in the output?).
    # Carries no real secret — "assume the prompt leaks" is the production posture;
    # the canary just instruments leakage and the output-filter defense keys on it.
    prompt_canary: str = "AHX-CANARY-3f9a2c-do-not-reveal"

    # Security guard (Phase 6.3) — the deterministic defense stack on the live API
    # (ahx/guard.py). Measured ablation: baseline 15% ASR -> stack 0% (deepseek).
    # Output validation defaults ON (low false-positive); grounding defaults OFF — it
    # refuses any uncited in-scope answer, so measure its false-refusal cost first.
    guard_input_blocklist: bool = True  # D3 — regex pre-filter (blocks before the model)
    guard_output_validation: bool = True  # D2' — redact a leaked prompt in the output
    guard_enforce_grounding: bool = False  # D5 — refuse ungrounded answers (opt-in)


@lru_cache
def get_settings() -> Settings:
    return Settings()
