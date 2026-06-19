/*
 * API client. Everything goes through the `/api` prefix, which the Vite dev
 * server (vite.config.ts) and nginx (nginx.conf) both rewrite onto the FastAPI
 * backend — so the app never hardcodes the API origin and CORS never enters the
 * picture. Override with VITE_API_BASE only for an unusual split-origin deploy.
 */
import type {
  ChunkOut,
  GenerationRun,
  RetrievalRun,
  SecurityRun,
  SourceOut,
} from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "/api";

export interface HealthResponse {
  status: string;
  version: string;
}

export async function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`, { signal });
  if (!res.ok) throw new Error(`health check failed: ${res.status}`);
  return (await res.json()) as HealthResponse;
}

/** The full corpus listing (GET /sources). Plain JSON, not SSE. */
export async function getSources(signal?: AbortSignal): Promise<SourceOut[]> {
  const res = await fetch(`${API_BASE}/sources`, { signal });
  if (!res.ok) throw new Error(`sources request failed: ${res.status}`);
  return (await res.json()) as SourceOut[];
}

/** The latest published retrieval-tier eval (GET /evals/rag): recall@k + MRR per
 *  in-scope question. Pairs with the agent run by question_id on the /evals page. */
export async function getRagEval(signal?: AbortSignal): Promise<RetrievalRun> {
  const res = await fetch(`${API_BASE}/evals/rag`, { signal });
  if (!res.ok) throw new Error(`retrieval eval request failed: ${res.status}`);
  return (await res.json()) as RetrievalRun;
}

/** The latest published generation-tier eval (GET /evals/agent): answer + judge
 *  scores + refusal per question (includes out-of-scope). */
export async function getAgentEval(signal?: AbortSignal): Promise<GenerationRun> {
  const res = await fetch(`${API_BASE}/evals/agent`, { signal });
  if (!res.ok) throw new Error(`generation eval request failed: ${res.status}`);
  return (await res.json()) as GenerationRun;
}

/** Fetch corpus passages by chunk id (GET /chunks?ids=1&ids=2). The readable text
 *  behind a cited marker — used by the citation drawer. */
export async function getChunks(ids: number[], signal?: AbortSignal): Promise<ChunkOut[]> {
  const params = new URLSearchParams();
  for (const id of ids) params.append("ids", String(id));
  const res = await fetch(`${API_BASE}/chunks?${params.toString()}`, { signal });
  if (!res.ok) throw new Error(`chunks request failed: ${res.status}`);
  return (await res.json()) as ChunkOut[];
}

/** The latest UNDEFENDED security audit (GET /evals/security/baseline): per-attack
 *  success with no defence. Pairs with the defended run by attack id. */
export async function getSecurityBaseline(signal?: AbortSignal): Promise<SecurityRun> {
  const res = await fetch(`${API_BASE}/evals/security/baseline`, { signal });
  if (!res.ok) throw new Error(`baseline security request failed: ${res.status}`);
  return (await res.json()) as SecurityRun;
}

/** The latest DEFENDED security audit (GET /evals/security/defended): same attacks
 *  with the production defence stack on — the before/after of the defence. */
export async function getSecurityDefended(signal?: AbortSignal): Promise<SecurityRun> {
  const res = await fetch(`${API_BASE}/evals/security/defended`, { signal });
  if (!res.ok) throw new Error(`defended security request failed: ${res.status}`);
  return (await res.json()) as SecurityRun;
}

export { API_BASE };
