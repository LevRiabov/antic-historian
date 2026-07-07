/*
 * Per-client session id for the demo's free-tier query cap.
 *
 * The backend caps queries per `X-Session-Id` header (api/limits.py). When the
 * header is absent the cap falls back to the client IP — which, behind the prod
 * nginx/load-balancer edge, collapses every visitor onto ONE shared budget. The
 * in-memory counter has no TTL, so once that shared bucket hits the cap a brand-new
 * visitor is rejected before any LLM call ("cap reached" with no OpenRouter
 * traffic). We give each browser a stable random id so the cap is genuinely
 * per-visitor and independent of network topology.
 *
 * Persisted (not per-tab) so a page reload doesn't hand out a fresh budget — the
 * point of the cap is to bound one visitor's spend across reloads. A genuinely new
 * browser / cleared storage / incognito gets a fresh id, which is the intended
 * "new session" semantic.
 */

const STORAGE_KEY = "ahx-session-id";

// Falls back to an in-memory id when localStorage throws (private mode, storage
// disabled). Per-tab rather than persistent, but still distinct per client — far
// better than omitting the header and collapsing onto the shared IP bucket.
let memoryId: string | null = null;

function newId(): string {
  // randomUUID needs a secure context (prod HTTPS + localhost both qualify); the
  // catch keeps a non-secure origin from throwing on page load.
  try {
    return crypto.randomUUID();
  } catch {
    return `sess-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
  }
}

export function getSessionId(): string {
  try {
    const existing = localStorage.getItem(STORAGE_KEY);
    if (existing) return existing;
    const id = newId();
    localStorage.setItem(STORAGE_KEY, id);
    return id;
  } catch {
    if (!memoryId) memoryId = newId();
    return memoryId;
  }
}
