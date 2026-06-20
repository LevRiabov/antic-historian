"""Application settings — single source of configuration truth.

All values can be overridden via environment variables with the AHX_ prefix
(e.g. AHX_DATABASE_URL), or a local .env file. Validated at startup.
"""

from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from pydantic import BaseModel, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChatEndpoint(BaseModel):
    """One OpenAI-compatible chat endpoint in the fallback lineup (6.4 / D5). Distinct
    provider+model from the primary so a single outage isn't total. Parsed from JSON in
    AHX_CHAT_FALLBACKS, e.g. '[{"base_url":"https://...","model":"...","api_key":"..."}]'."""

    base_url: str
    model: str
    api_key: SecretStr | None = None  # SecretStr: masked in reprs/logs (unwrap at the wire)


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
    embed_api_key: SecretStr | None = None  # required for the hosted default
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
    rerank_api_key: SecretStr | None = None
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
    chat_api_key: SecretStr | None = None
    # Decoding temperature — pinned low for run-to-run stability (the agent's
    # answer/refuse decision is high-variance; eval-log 2026-06-16). 0.0 asks for
    # greedy decoding where the provider honours it; recorded as an explicit knob
    # rather than an invisible default so it is part of every run's config.
    chat_temperature: float = 0.0
    # Output ceiling for every served chat call (single-shot answer + each of the
    # agent's think steps). Unbounded generation is unbounded query-time spend
    # (the expensive ledger) and the root cause of the runaway/JSON-truncation
    # path the agent grammar defends against; cap it. ~2k tokens comfortably fits
    # a full cited answer. None = no cap (not recommended on a public demo).
    chat_max_tokens: int | None = 2048
    # Fallback lineup (6.4 / D5): ordered alternates wrapped with the primary in a
    # CompositeChatModel. Empty = no fallover (the bare primary serves). Distinct
    # providers so one outage != total outage; the served model rides `served_by` to
    # the SSE indicator + trace. JSON-encoded list of ChatEndpoint via AHX_CHAT_FALLBACKS.
    chat_fallbacks: list[ChatEndpoint] = []

    # Deep mode (6.7) — the agent served as opt-in "deep mode" over /ask (streams its
    # search->read->cite loop). Retrieval = the D5 KEEP (dense-ctx-v1, NO paid reranker
    # on the served path); max_steps matches the eval default so served == measured.
    agent_retriever: str = "dense-ctx-v1"
    agent_max_steps: int = 8

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
    # GLOBAL daily spend ceiling — a hard kill-switch on total /ask volume, not a
    # per-client limit. The per-IP window + session cap are both client-rotatable
    # (X-Session-Id is client-sent; IPs rotate), so neither bounds total cost on a
    # public demo. This does: a single rolling-24h counter across ALL callers, after
    # which every /ask 429s until the oldest request ages out. At ~$0.02-0.05/query
    # this caps daily spend at a few dollars regardless of who is calling. 0 = off.
    daily_request_cap: int = 100
    # Hard wall-clock ceiling for a single /ask SSE stream. A stalled provider
    # otherwise pins the connection (and its DB/pool slot) for the LLM read
    # timeout x N agent steps; this bounds it and emits a terminal error frame.
    # Generous enough for a full deep-mode loop; tighten for a stricter demo.
    request_timeout_seconds: int = 180
    # Per-step bound on ONE deep-mode model call (a think/synthesis turn). The overall
    # request_timeout_seconds caps the whole run, but a single hung call could eat all
    # 180s before failing — too long for one step. This kills a stalled call far sooner
    # (it propagates as the terminal error frame). Set above the slowest LEGIT call: a
    # think reply is small, the synthesis answer is the long one, so leave headroom.
    # Applies to the LIVE deep path only — the eval harness runs untimed so a measured
    # run is never corrupted by a cut. 0 disables the per-step bound.
    agent_step_timeout_seconds: int = 90
    # Behind a reverse proxy (prod), the real client IP is in X-Forwarded-For, not
    # request.client. OFF by default (locally the header is spoofable). MUST be ON in
    # the nginx-proxied prod deploy (ADR-004) — otherwise every request shares the
    # proxy's IP and the per-IP limiter collapses into one global bucket. Safe ONLY if
    # the proxy OVERWRITES X-Forwarded-For (not appends); an appending proxy lets a
    # client spoof the leading IP. validate_serving_config() warns when this is off in
    # a non-dev env with rate limiting on.
    trust_forwarded_for: bool = False

    # Judge LLM (eval generation tier, phase boundaries). Unset = judge
    # layer unavailable; use a strong model — weak judges miscalibrate.
    judge_base_url: str | None = None
    judge_model: str | None = None
    judge_api_key: SecretStr | None = None

    # Attribution judge — the attribution rubric is judge-noise-limited on a
    # flash-tier judge (0.66/question swing, full 1<->5 flips on multi-source;
    # eval-log 2026-06-15), itself a hard reasoning task. Route ONLY attribution
    # to a stronger model than the faith/compl judge. Unset = the main judge
    # scores attribution too (back-compat). A judge must be >= the GENERATED
    # model's tier — a weaker judge can't reliably grade a stronger model.
    attrib_judge_base_url: str | None = None
    attrib_judge_model: str | None = None
    attrib_judge_api_key: SecretStr | None = None

    # Enrichment LLM (Phase 4.1 contextual-note + metadata pass). Offline,
    # one-time, cached to corpus/enriched/ — so a local model is the cheap
    # default (gemma-12b-enrich = the parallel-slot llama-swap profile). To run
    # the pass hosted instead (deepseek-v4-flash ≈ $5/46k), override these three.
    enrich_base_url: str = "http://127.0.0.1:8080/v1"
    enrich_model: str = "gemma-12b-enrich"
    enrich_api_key: SecretStr | None = None
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
    def eval_runs_dir(self) -> Path:
        # Saved eval run records (backend/evals/runs). The API publishes the latest
        # -rag / -agent run from here (api/evals.py); the eval CLI writes them.
        return _BACKEND_DIR / "evals" / "runs"

    @property
    def security_runs_dir(self) -> Path:
        # Saved security-audit records (backend/evals/security_runs). The API publishes
        # the latest -baseline / -defended run from here for the security page.
        return _BACKEND_DIR / "evals" / "security_runs"

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
    langfuse_secret_key: SecretStr | None = None

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


_LOCAL_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}


def _is_local_url(url: str) -> bool:
    """True for a loopback endpoint (local llama-swap/vLLM), where an API key is
    legitimately absent. A hosted provider is anything else."""
    return (urlparse(url).hostname or "") in _LOCAL_HOSTS


def validate_serving_config(settings: Settings) -> list[str]:
    """Fail-loud preflight for the API serving path, called at lifespan startup.

    Construction-time (model_validator) is the wrong layer: offline CLI commands
    and tests build Settings whose hosted defaults they never call, and would all
    break. This runs only when we actually serve.

    Raises on hard errors (a hosted served endpoint with no key would 401 on the
    first real query while booting green). Returns soft warnings for the caller to
    log (deployment posture that can't be proven wrong from config alone)."""
    missing: list[str] = []
    if not _is_local_url(settings.embed_base_url) and not settings.embed_api_key:
        missing.append("AHX_EMBED_API_KEY (embed_base_url is a hosted endpoint)")
    if not _is_local_url(settings.chat_base_url) and not settings.chat_api_key:
        missing.append("AHX_CHAT_API_KEY (chat_base_url is a hosted endpoint)")
    if missing:
        raise RuntimeError(
            "Missing required API key(s) for serving — set them or point the base_url "
            "at a local endpoint: " + "; ".join(missing)
        )

    warnings: list[str] = []
    if (
        settings.env != "dev"
        and settings.rate_limit_per_window
        and not settings.trust_forwarded_for
    ):
        warnings.append(
            "trust_forwarded_for is off in a non-dev env with rate limiting on: behind a "
            "proxy every client shares one IP, so the per-IP limit becomes a single global "
            "bucket. Set AHX_TRUST_FORWARDED_FOR=1 (and ensure the proxy overwrites XFF)."
        )
    return warnings
