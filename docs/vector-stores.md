# Vector Store — Landscape & Selection (Gate D3)

> **Status:** research input for decision gate D3 (see [project-plan.md](../project-plan.md)).
> **Written:** 2026-06-10. Free-tier terms change often — re-verify before committing. ⚠ = unverified / from secondary sources.
> Strategy per plan: **experiment locally first, move to a free cloud tier for production.** This doc adds the key principle that makes that strategy safe: **engine parity** — pick something that runs identically in Docker locally and on a free cloud tier, so ablation results transfer.

---

## 1. Our requirements

| Requirement | Detail |
|---|---|
| **Scale (measured 2026-06-10)** | First batch (`ai_historian_corpus_eu_pd.txt`, 16 PG texts, HEAD-verified): **35.7 MB raw** ≈ ~8M tokens ≈ **~18k chunks**. Full list (appendix: Tacitus, Caesar, Josephus, remaining Grote vols…) plausibly ~2×: **~70 MB / ~15M tokens / ~35k chunks**. Cloud budget at 35k chunks, 1024d `halfvec`: vectors ~70 MB + HNSW ~85 MB + chunk text ~75 MB + source text ~60 MB + FTS index ~50 MB ≈ **~350 MB** — fits 500 MB free tiers, but only with the knobs: halfvec or ≤1024 MRL dims, ONE embedding config in cloud, raw source text externalized if it gets tight. |
| **Ablation multiplies storage — locally** | Multiple embedding models × chunking versions coexist during Phases 2–4 (the `chunking_version` pattern). That's a *local Docker* concern (disk is free at home); the cloud only ever hosts the winning config. |
| **Relational data exists anyway** | Sources table, golden set, eval runs, locator metadata, rate-limit state. Either the vector store's neighbor handles this… or the vector store *is* a relational DB and one store does everything. |
| **Hybrid search** | BM25/sparse + dense is a planned Phase 4 ablation arm (likely valuable at this corpus size) — built-in support is a real plus. |
| **Metadata filtering** | Source-isolated search (`source_slug = X`) powered the contradiction feature; filtered vector search must be first-class. |
| **Demo-liveness** | A client may open the demo **months** after launch, unattended. Free-tier *pause/delete-on-inactivity* behavior matters as much as storage size. |
| **Framework + market signal** | Must integrate cleanly with the chosen framework (D1); bonus if the choice reads as "what real production teams use." |

---

## 2. Comparison table

| Store | Local (Docker)? | Free cloud tier | Inactivity behavior | Hybrid | Verdict for us |
|---|---|---|---|---|---|
| **Postgres + pgvector** | ✅ trivial (`pgvector/pgvector` image) | **Supabase**: 500 MB DB, 2 projects · **Neon**: 0.5 GB/project, 100 CU-hrs/mo | Supabase: **pauses after 1 wk idle, manual restore** ⚠ · Neon: scale-to-zero but **auto-wakes on connection** (~secs) | Via Postgres FTS (`tsvector`) — DIY but real | **Front-runner** — one DB for vectors *and* all relational data |
| **Qdrant** | ✅ trivial (`qdrant/qdrant`) | **Cloud free forever: 1 GB RAM / 4 GB disk** (~1M 768d vectors), no card | **Suspends after 1 wk unused; deleted after 4 wks** ⚠ — risky for an unattended demo | ✅ built-in (named + sparse vectors, fusion) | **Challenger** — best pure-vector free tier + features |
| **LanceDB** | ✅ embedded — no server at all (a directory of files) | n/a — the "cloud" is your API container; ships inside it | None — lives and dies with the app | ✅ built-in FTS (tantivy) + vector | **Wildcard** — zero ops, perfect local/prod parity by construction |
| **Pinecone** | ❌ **no local engine** | Starter: 2 GB, 5 serverless indexes, us-east-1 only | Indexes **paused after 3 wks idle** ⚠ | Sparse-dense supported | ❌ fails the parity principle; experiments wouldn't transfer |
| **Weaviate** | ✅ Docker | Cloud sandbox **expires in 14 days**; cheapest managed = $25/mo | n/a | ✅ built-in | ❌ no durable free tier |
| **Milvus / Zilliz** | ⚠ heavy compose (etcd+minio+milvus) | Zilliz free: limited (1 collection ⚠) | ⚠ | ✅ | ❌ ops burden disproportionate to 20k vectors |
| **Chroma** | ✅ easy, embedded or server | Cloud is young; free credits ⚠ | ⚠ | partial | Prototype-grade signal; nothing it does better for us |
| **MongoDB Atlas** | DB yes; **vector search is Atlas-only** ⚠ (parity broken) | M0: 512 MB, **1 vector index** | M0 doesn't pause ⚠ | ✅ Atlas Search | Interesting only via the Voyage tie-in; 1-index cap kills ablations |
| **sqlite-vec** | ✅ embedded | n/a (ships with app) | None | ❌ (pair with FTS5, DIY) | Strictly weaker LanceDB for our needs |
| Redis / Elasticsearch / OpenSearch | ✅ | 30 MB / trial-only | — | — | ❌ free tiers unusable or absent |

---

## 3. The three real candidates

### Postgres + pgvector — front-runner
**Pros:**
- **One database for everything.** Chunks+vectors, sources, canonical locators, golden set, eval runs, app state — joined with SQL. Every alternative means running *two* stores (vector DB + something relational) or abusing a vector DB's payload as a database.
- Perfect parity: identical engine in local Docker and on Supabase/Neon.
- You already know Postgres (ParadeDB last project) — depth, not novelty, where it doesn't pay.
- Strongest market signal: "we did RAG on the Postgres you already run" is a sentence freelance clients love; pgvector is the default at most shops.
- Filtered vector search, transactions, backups — boring and proven.

**Cons / footguns:**
- **`vector` type HNSW indexing caps at 2,000 dims** — voyage-4-nano's default 2048d won't index. Fix: `halfvec` (fp16, indexes up to 4,000 dims, negligible quality loss) or MRL-truncate to 1024d. Decide at ingest, verify in the D2 ablation.
- Hybrid search is assemble-it-yourself: `tsvector` BM25-ish + vector + RRF in SQL. Doable (and educational), not turnkey. (ParadeDB's `pg_search` gives real BM25 but isn't on Supabase/Neon ⚠ — hybrid quality on managed Postgres is part of the ablation.)
- Supabase free pauses after 1 week idle with **manual** restore — unacceptable for an unattended demo. Mitigations: weekly keep-alive ping (GitHub Actions cron hitting a health endpoint), or prefer **Neon** (auto-wake on connection, cold start measured in seconds).
- ANN performance tuning (HNSW params) is on us — irrelevant at 20k vectors, worth a sentence in the case study.

### Qdrant — challenger
**Pros:** best-in-class free cluster (4 GB disk ≫ our needs); **named vectors** (multiple embedding models per point — convenient for the D2 ablation even cloud-side); built-in sparse vectors + fusion = turnkey hybrid; first-class filtered search; Docker parity; "AI-native stack" résumé signal; LlamaIndex/LangChain integrations are first-tier.
**Cons:** second store still needed for relational data (sources, eval runs) — likely SQLite/Postgres anyway, so the "one store" simplicity is gone; free cluster **suspends after 1 week unused and is deleted after 4 weeks** ⚠ — a portfolio app that dies a month after you stop touching it is a landmine (keep-alive cron is mandatory, and deletion risk remains).

### LanceDB (embedded) — wildcard
**Pros:** zero ops, zero cost, zero pause risk — the store is files inside the API container, rebuilt from the ingestion pipeline at deploy time; local/prod parity is *by construction*; built-in FTS gives easy hybrid; columnar format is genuinely modern (Lance/Arrow).
**Cons:** corpus updates require redeploy (fine for a static corpus, wrong the moment the story becomes "clients can add documents"); single-process embedded model; weaker "I can run your production DB" signal; relational data still needs a home (SQLite alongside).

---

## 4. Recommendation

**Default: Postgres + pgvector.** Local Docker for all experimentation (every chunking/embedding variant side by side), **Neon** free tier for production (auto-wake beats Supabase's manual unpause for an unattended demo; re-verify both at deploy time). One engine end-to-end, one store for vectors *and* the relational backbone the project needs regardless, and the strongest client-facing story.

**Run the D2 embedding ablation with Qdrant's named vectors in mind but on pgvector anyway** — at 20k vectors, an extra embedding column per model in Postgres is trivial; we don't need named vectors badly enough to take on a second store.

**Decision check at D3 (after Phase 1 ingestion + Phase 4 hybrid arm):** switch to Qdrant only if (a) the hybrid ablation shows DIY Postgres FTS+RRF meaningfully underperforming Qdrant's native fusion, or (b) managed-Postgres limits bite in practice. Adopt LanceDB only if deployment friction with a managed DB becomes real — it's the escape hatch, not the plan.

**What this rules out and why:** Pinecone (no local engine — breaks experiment→prod parity; vendor-locked), Weaviate (no durable free tier), Milvus (ops weight absurd at our scale), MongoDB M0 (1 vector index kills ablations), Chroma/sqlite-vec (no advantage over the three above for us).

---

## 5. Footguns

1. **pgvector 2,000-dim index ceiling** — 2048d embeddings (voyage-4-nano default) need `halfvec` or MRL truncation to 1024. Catch this at schema design, not after ingesting.
2. **Free-tier necrosis** — Supabase pauses (manual restore), Qdrant free *deletes* after 4 idle weeks, Pinecone pauses after 3. For a portfolio demo, wire a weekly keep-alive cron (GitHub Actions) against a health endpoint that touches the DB, whatever the store.
3. **Don't benchmark ANN recall at 20k vectors and generalize** — at this scale even exact scan is fast; HNSW-vs-flat differences are invisible. Frame any perf numbers honestly in the case study.
4. **Distance-metric mismatch** — embeddings are L2-normalized → cosine and dot product agree; pick one (`vector_cosine_ops`) and assert normalization at ingest. A mixed-metric index silently returns garbage rankings.
5. **Egress caps** — Supabase free includes 5 GB egress/mo ⚠; chunky `SELECT *` over full text in eval loops can eat it. Run heavy eval loops against local Docker, not the cloud instance.

---

## 6. Sources

- [Supabase pricing](https://supabase.com/pricing) · [Supabase free-tier limits overview](https://uibakery.io/blog/supabase-pricing) ⚠ secondary
- [Neon plans](https://neon.com/docs/introduction/plans) · [Neon free plan guide](https://neon.com/blog/how-to-make-the-most-of-neons-free-plan)
- [Qdrant pricing](https://qdrant.tech/pricing/) · [Qdrant Cloud cluster docs](https://qdrant.tech/documentation/cloud/create-cluster/) (free-tier suspend/delete terms) ⚠ partially secondary
- [Pinecone limits](https://docs.pinecone.io/reference/quotas-and-limits) · [Pinecone pricing](https://www.pinecone.io/pricing/)
- [MongoDB Atlas free-cluster limits](https://www.mongodb.com/docs/atlas/reference/free-shared-limitations/)
- [pgvector README](https://github.com/pgvector/pgvector) (dim limits, halfvec) — stable, from training knowledge
- Comparison articles (⚠ secondary): [Firecrawl best vector DBs 2026](https://www.firecrawl.dev/blog/best-vector-databases), [vector DB pricing comparison](https://www.buildmvpfast.com/api-costs/vector-db)
