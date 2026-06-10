# Embedding Models — Landscape & Selection (Gate D2)

> **Status:** research input for decision gate D2 (see [project-plan.md](../project-plan.md)).
> **Written:** 2026-06-10. Pricing/free tiers in this space change quarterly — re-verify before committing money. Items marked ⚠ are from training knowledge or secondary sources, not verified against official pages.
> **Final decision is made by ablation on our golden set, not by this doc.** This doc picks the *shortlist*.

---

## 1. Our constraints (what actually matters here)

| Constraint | Implication |
|---|---|
| Queries embedded **at request time in production** | Model must run on a cheap CPU container *or* be a hosted API. The 5070 Ti only exists at ingest time. |
| **Same representation** for corpus and queries | One model (and one runtime + prefix policy) for both sides. Mixed stacks are a known silent-degradation footgun (§6). |
| ~3–6M token one-time ingest, low query volume (~hundreds/day) | Ingest cost is trivial almost everywhere ($0.06–$0.90 even on paid APIs). **Cost is a non-issue; quality, ops, and lock-in are the real criteria.** |
| Target ~$0/month running cost | Free tiers or self-hosted CPU inference. |
| English, **archaic/Victorian translations** (Gutenberg-era prose) | Out-of-distribution for every model. Leaderboard deltas of 1–3 points won't predict the winner on our corpus → **ablation is mandatory**. |
| Re-embedding ~15k chunks is cheap (minutes on GPU) | The provisional pick is low-risk; we can switch after measuring. |
| Portfolio project | The model-selection ablation itself becomes case-study content. |

---

## 2. Cloud / API models

| Model | Price /M tok | Free tier | Dims | Ctx | Notes |
|---|---|---|---|---|---|
| **Voyage 4 family** (`voyage-4-large` / `-4` / `-4-lite`), Jan 2026 | $0.12 / $0.06 / $0.02 | **200M tokens per model** | 2048 MRL (256–2048) | 32k | Shared embedding space across the whole family incl. open-weight nano (§3). Output: float/int8/binary. `input_type` query/document. MongoDB-owned. |
| **Gemini `gemini-embedding-001`** | $0.15 ($0.075 batch) | Yes — generous TPM (≈10M ⚠), token-based limits | 3072 MRL (128–3072) | 2,048 ⚠ (short!) | #1 proprietary on MTEB English (68.32). Task types (`RETRIEVAL_QUERY`/`_DOCUMENT`). Free-tier data may be used for training ⚠. **Gemini Embedding 2** (multimodal) now exists — text-only pricing TBD. |
| **Cohere `embed-v4.0`** | $0.12 | Trial keys: ~1k calls/mo ⚠, non-production license | 256–1536 MRL | 128k | Multimodal, longest context. Trial too tight for our ingest. |
| **OpenAI `text-embedding-3-small` / `-large`** | $0.02 / $0.13 | None | 1536 / 3072 (truncatable) | 8k | Mid-pack quality by 2026 (62.3 / 64.6). Boring-reliable ops. No query/doc types, float only. |
| **Jina v4 / v5** | ~$0.02 ⚠ (packs) | ~10M tokens on new key ⚠ | 2048 MRL | 32k | v4 weights are **CC-BY-NC** → no commercial self-hosting; v5 ⚠ license unverified. API-only for us. |
| **Mistral `mistral-embed`** | $0.10 | Small free plan ⚠ | 1024 | 8k | Dated, no MRL, no task types. **Skip.** |
| **Open weights via API** — DeepInfra / OpenRouter / Nebius serving **Qwen3-Embedding** etc. | ~$0.01 (8B on OpenRouter) | Pay-as-you-go, no minimum | model-native | model-native | The "same weights local + hosted" play. ⚠ Hosted APIs do **not** auto-apply instruction prefixes — you must send them yourself (§6). |

**When cloud-only is the right call:** you want the absolute quality ceiling (gemini-embedding / voyage-4-large) and accept vendor lock-in — switching later means re-embedding *and* re-validating everything.
**When to avoid:** hard $0 requirements with rate-limit-sensitive UX (free tiers throttle), or when you want the embedding step inside your own latency budget.

---

## 3. Open-weight models (self-hostable)

Ranked roughly by English-retrieval quality within their size class. All scores approximate; MTEB v2 (2025+) and legacy MTEB scores are **not comparable** — treat ranks, not numbers.

### Tier A — quality leaders (GPU-class, ingest or beefy serving only)
| Model | Params | Dims | Ctx | License | Notes |
|---|---|---|---|---|---|
| **Qwen3-Embedding-8B / -4B** | 7.6B / 4B | 4096 / 2560 MRL | 32k | Apache 2.0 | #1 open on MTEB multilingual (70.58). 8B needs quantization to fit the 5070 Ti (16GB). Not CPU-servable. Hosted on DeepInfra/OpenRouter ~$0.01/M. |
| NV-Embed-v2, bge-en-icl, gte-Qwen2-7B | 7B-class | 4096 | 32k | varies | Strong on legacy MTEB; same "too big to serve cheap" problem. Skip unless API-hosted. |

### Tier B — the sweet spot: CPU-feasible, modern, strong retrieval
| Model | Params | Dims | Ctx | License | CPU latency (1 query, 2–4 vCPU) ⚠est. | Notes |
|---|---|---|---|---|---|---|
| **voyage-4-nano** (Jan 2026) | ~340M | 2048 MRL (256–2048) | 32k | **Apache 2.0** | ~50–150ms int8 | **Shared embedding space with hosted voyage-4/lite/large** — unique escape hatch (§5). Beats voyage-3.5-lite. ONNX + GGUF community ports exist. Prompts handled by `encode_query()`/`encode_document()`. |
| **Qwen3-Embedding-0.6B** | 595M | 1024 MRL (32–1024) | 32k | Apache 2.0 | ~100–300ms Q8 | Best raw quality in CPU-reachable class. Needs ~1–2GB RAM quantized. Instruction-prefix + last-token-pooling footguns (§6). Official GGUF; also hosted cheap. |
| **snowflake-arctic-embed-l-v2.0** | 568M | 1024 MRL | 8k | Apache 2.0 | ~80–200ms int8 | Strong English despite multilingual. `query: ` prefix. |
| **gte-modernbert-base** | **149M** | 768 | 8k | Apache 2.0 | ~15–50ms int8 | **No prefixes needed.** ModernBERT = genuinely fast on CPU. bge-large quality at half the size, 16× the context. Best tiny-container option. |
| **granite-embedding-english-r2** (IBM) / **-small-r2** | 149M / 47M | 768 / 384 | 8k | Apache 2.0 | ~15–50ms / ~5–20ms | ModernBERT-based, surprisingly competitive. The 47M one fits anywhere. |
| **EmbeddingGemma** (Google) | 308M | 768 MRL | 2k | Gemma ToU (not OSI) | ~20–80ms int4 | Built for on-device (<200MB RAM claimed). Short context; license needs reading for commercial use. |

### Tier C — older defaults, only if a reason exists
| Model | Why it's here |
|---|---|
| **bge-m3** (568M, 1024d, 8k) | What rag-historian used. Multilingual + built-in sparse/ColBERT vectors (interesting for hybrid experiments), but mid-pack English dense retrieval by 2026 — consistent with the "performs not great" experience. |
| bge-large-en-v1.5, e5-large-instruct, nomic-embed-v1.5, mxbai-embed-large | Mature, ubiquitous, well-tooled — and all outclassed in Tier B at similar or smaller sizes. 512-ctx limits on several. |

**When local is the right call:** $0 forever, no rate limits, no vendor risk, embedding inside your own latency budget, and (for us) the same weights at ingest and query by construction.
**When to avoid:** if your host gives <512MB RAM (then only granite-small/gte-modernbert int8 fit, slowly), or if you need the 7B-class quality ceiling without a GPU.

---

## 4. Hosting reality check (where the query-time model lives)

| Host | Free/cheap tier | Fits |
|---|---|---|
| **HF Spaces** (free CPU Basic) | 2 vCPU, **16GB RAM**, free, sleeps when idle | Anything in Tier B, even Qwen3-0.6B fp16. Cold-start wake is the UX cost. |
| **Fly.io** | No free tier anymore; shared-cpu 1–2GB ≈ $2–5/mo | Tier B comfortably. |
| **Render** (free) | 512MB RAM, ~0.1 vCPU, 15-min spin-down | Only 47–150M int8 models, and slowly. |
| **Railway** | $5/mo Hobby | Tier B comfortably. |

⚠ Tier specs drift — re-verify at Phase 7 (deploy).

---

## 5. Recommendation for this project

**Working hypothesis: `voyage-4-nano`, validated by ablation against `Qwen3-Embedding-0.6B`, `gte-modernbert-base`, and one hosted ceiling reference (`voyage-4-lite` or `gemini-embedding-001`).**

Why nano is the front-runner on paper:

1. **It dissolves our hardest constraint.** Apache 2.0 open weights at ~340M params → runs on CPU in the API container (same weights at ingest on the 5070 Ti and at query time in prod — parity by construction, no API dependency, $0/month, no rate limits).
2. **The shared embedding space is a unique escape hatch.** All Voyage 4 models embed into one space. If nano's quality disappoints *after* launch, we can upgrade the *corpus* side to `voyage-4-large` via API (200M free tokens ≫ our 6M ingest) **without re-embedding queries or changing the serving stack** — or upgrade queries and keep the local corpus. No other vendor offers asymmetric upgrades without a full re-index.
3. **Modern feature set:** 32k context (contextual-retrieval notes won't hit a ceiling), MRL dims 256–2048 (storage/quality ablation for free), int8/binary output types, clean query/document prompt handling in sentence-transformers.
4. **Risk:** it's new (Jan 2026) — less battle-tested tooling than BGE/Qwen; community ONNX/GGUF ports need a parity check against the reference implementation before trusting them.

Why the challengers stay in:

- **Qwen3-Embedding-0.6B** — likely the strongest raw retrieval quality we can serve on CPU; if it beats nano by a clear margin on our golden set, the extra RAM (~1–2GB, fine on HF Spaces) and prefix discipline are worth it. Also hosted at ~$0.01/M (DeepInfra/OpenRouter) as an ops fallback.
- **gte-modernbert-base** — the insurance policy: 149M, no prefixes, fastest CPU latency; if the final host turns out RAM-starved, this is the fallback. Also a useful "how much does size buy" data point in the ablation.
- **Hosted ceiling reference** — tells us how much quality we're leaving on the table by going local. If the gap on *our* corpus is huge, the architecture conversation reopens.

**The ablation (Phase 2, Gate D2):** embed the corpus with each shortlisted model → run golden-set retrieval metrics (recall@k, MRR, per-category) → compare quality, query latency on target hardware, RAM, ops complexity. Publish the table. That's both our decision and a case-study section.

### What we explicitly skip, and why
- **bge-m3** — already measured in rag-historian; mid-pack English by 2026. Its sparse/ColBERT outputs may return as a *hybrid-retrieval* experiment, not as the dense embedder.
- **Jina v3/v4** — NC-licensed weights kill the local play; API-only adds lock-in without a quality argument over Voyage/Gemini.
- **Cohere, Mistral, OpenAI** — no free path / dated / mid-pack; nothing they uniquely offer fits our constraints.
- **7B-class open models self-hosted** — quality ceiling is real, but serving cost breaks the $0 target; revisit only via cheap hosted APIs if the ablation shows small models failing badly.

---

## 6. Footguns (each of these has burned real projects, some burned rag-historian)

1. **Prefix/instruction parity.** Most modern embedders want different prompts for queries vs documents (`encode_query` vs `encode_document`, `input_type`, instruction prefixes). Hosted APIs serving open models generally **don't inject these automatically**. Corpus embedded with prefixes + queries without (or vice versa) = silent 1–5%+ quality loss. → Wrap embedding in ONE module that owns the prefix policy; never call the model directly from two places.
2. **Same model ≠ same vectors.** Runtime differences (sentence-transformers vs llama.cpp vs vLLM vs hosted), fp16/fp32, and GGUF quantization all shift vectors. Early Qwen3-on-Ollama had outright pooling/padding bugs. → Parity test at ingest time: embed 20 fixed sentences on both stacks, assert cosine ≥ 0.999; re-run the golden-set retrieval eval after *any* runtime switch.
3. **Pooling and normalization.** Qwen3 uses last-token pooling; BGE/GTE use CLS/mean. Frameworks usually get this right via model config — hand-rolled ONNX pipelines usually don't. Always L2-normalize unless measured otherwise.
4. **MRL truncation isn't free.** 2048→512 dims is tempting for storage, but loss is model-specific (Voyage/Jina trained for it; others retrofit). Measure at the dims you'll ship, not the dims the leaderboard used.
5. **Leaderboard ≠ our corpus.** MTEB v2 vs legacy scores aren't comparable, several public "2026 leaderboards" mix them, and none contain Victorian-translation prose. The golden set is the only leaderboard that counts here.
6. **OpenAI SDK base64 default** (from rag-historian): pin `encoding_format="float"` when using OpenAI-compatible endpoints for embeddings, or you may parse garbage.

---

## 7. Sources

- [Voyage AI pricing](https://docs.voyageai.com/docs/pricing) · [Voyage 4 family announcement](https://blog.voyageai.com/2026/01/15/voyage-4/) · [voyage-4-nano model card](https://huggingface.co/voyageai/voyage-4-nano) (+ [community ONNX](https://huggingface.co/onnx-community/voyage-4-nano-ONNX))
- [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) (live) · [Awesome Agents MTEB snapshot, Apr 2026](https://awesomeagents.ai/leaderboards/embedding-model-leaderboard-mteb-april-2026/) (⚠ mixes legacy/v2 scores) · [Milvus: choosing embedding models for RAG 2026](https://milvus.io/blog/choose-embedding-model-rag-2026.md)
- [Gemini API rate limits](https://ai.google.dev/gemini-api/docs/rate-limits) · [Gemini embeddings docs](https://ai.google.dev/gemini-api/docs/embeddings)
- [Qwen3 Embedding 8B on OpenRouter](https://openrouter.ai/qwen/qwen3-embedding-8b) · [DeepInfra pricing](https://deepinfra.com/pricing) · Qwen3-Embedding model cards: [0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- Model cards for Tier B/C models on Hugging Face (gte-modernbert-base, granite-embedding-english-r2, snowflake-arctic-embed-l-v2.0, EmbeddingGemma, bge-m3)
- rag-historian Module 5/6 eval results (internal) — bge-m3 baseline experience
