"""Model price table + per-request cost (Phase 6.2).

Prices are FETCHED from OpenRouter, never hand-typed (rule #6 — verified, dated,
sourced). `ahx pricing refresh` regenerates data/pricing.json from the live
/models endpoint; this module loads that snapshot and does the arithmetic. The
snapshot carries `fetched_at` + `source` so every cost number is traceable to a
dated fetch.

Scope of the number: GENERATION tokens only. Query embedding (~10-20 tokens at
~$0.01/M ≈ $2e-7) is rounding noise, and EmbeddingClient does not surface usage —
excluded by design, documented here so the cost's meaning is explicit. Local
models (no "/" in the id — served on llama-swap) cost $0: the cost ledger says
query-time HOSTED spend is the expensive line; local GPU is one-time and free.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import httpx
from pydantic import BaseModel

from ahx.llm import Usage

# Next to the module (NOT under a data/ dir — that path is gitignored for corpus
# data). This dated snapshot is a committed, version-controlled artifact (rule #6).
_PRICING_PATH = Path(__file__).parent / "pricing_snapshot.json"

# The production GENERATION lineup (ADR-003) priced by default, regardless of which
# model a run's env points at. `refresh` also picks up any hosted model in Settings.
# Embeddings are intentionally absent — embed cost is out of scope (see module docstring),
# and the embedding model isn't in OpenRouter's /models LLM listing anyway.
LINEUP_MODELS: frozenset[str] = frozenset(
    {
        "deepseek/deepseek-v4-pro",  # agent reasoner
        "deepseek/deepseek-v4-flash",  # cheaper judge / candidate cheap tier
        "moonshotai/kimi-k2.6",  # judge (refusal/faith/compl)
        "qwen/qwen3.7-max",  # attribution judge
    }
)


class ModelPrice(BaseModel):
    input_per_m: float  # USD per 1M prompt tokens
    output_per_m: float  # USD per 1M completion tokens
    # Prompt-cache read price (6.6 readiness); None when the provider has none.
    cache_read_per_m: float | None = None


class PriceTable(BaseModel):
    fetched_at: str  # ISO date of the OpenRouter fetch
    source: str
    models: dict[str, ModelPrice]


class Cost(BaseModel):
    """Per-request generation cost. `usd` is None when the model is hosted but
    absent from the table — surfaced as unpriced, NEVER silently $0 (rule #6)."""

    usd: float | None
    input_tokens: int
    output_tokens: int
    model: str
    priced: bool


def _is_local(model: str) -> bool:
    # llama-swap ids are bare ("gemma-12b-16k"); hosted ids are "provider/model".
    return "/" not in model


@lru_cache
def load_price_table() -> PriceTable | None:
    """The committed snapshot, or None if it was never generated. Cached — the
    file is read once per process (it changes only via `ahx pricing refresh`)."""
    if not _PRICING_PATH.exists():
        return None
    return PriceTable.model_validate_json(_PRICING_PATH.read_text(encoding="utf-8"))


def cost_for(model: str, usage: Usage | None, table: PriceTable | None) -> Cost:
    inp = usage.prompt_tokens if usage else 0
    out = usage.completion_tokens if usage else 0
    price = table.models.get(model) if table else None
    if price is not None:
        usd = inp / 1_000_000 * price.input_per_m + out / 1_000_000 * price.output_per_m
        return Cost(
            usd=round(usd, 6), input_tokens=inp, output_tokens=out, model=model, priced=True
        )
    if _is_local(model):
        return Cost(usd=0.0, input_tokens=inp, output_tokens=out, model=model, priced=True)
    return Cost(usd=None, input_tokens=inp, output_tokens=out, model=model, priced=False)


def fetch_prices(base_url: str, api_key: str | None, wanted: set[str]) -> dict[str, ModelPrice]:
    """Pull `wanted` model prices from OpenRouter's /models. Values there are USD
    PER TOKEN strings (claude-opus-4.8 prompt "0.000005" = $5/M) -> x1e6 for /M."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    response = httpx.get(f"{base_url}/models", headers=headers, timeout=30.0)
    response.raise_for_status()
    data = cast(list[dict[str, Any]], response.json()["data"])
    found: dict[str, ModelPrice] = {}
    for model in data:
        mid = cast(str, model["id"])
        if mid not in wanted:
            continue
        pricing = cast(dict[str, Any], model.get("pricing") or {})
        cache_read = pricing.get("input_cache_read")
        found[mid] = ModelPrice(
            input_per_m=float(pricing["prompt"]) * 1_000_000,
            output_per_m=float(pricing["completion"]) * 1_000_000,
            cache_read_per_m=float(cache_read) * 1_000_000 if cache_read else None,
        )
    return found


def write_price_table(table: PriceTable) -> Path:
    _PRICING_PATH.parent.mkdir(parents=True, exist_ok=True)
    _PRICING_PATH.write_text(table.model_dump_json(indent=2) + "\n", encoding="utf-8")
    load_price_table.cache_clear()
    return _PRICING_PATH
