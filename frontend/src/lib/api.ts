/*
 * API client. Everything goes through the `/api` prefix, which the Vite dev
 * server (vite.config.ts) and nginx (nginx.conf) both rewrite onto the FastAPI
 * backend — so the app never hardcodes the API origin and CORS never enters the
 * picture. Override with VITE_API_BASE only for an unusual split-origin deploy.
 */
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

export { API_BASE };
