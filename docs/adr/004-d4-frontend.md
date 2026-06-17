# ADR-004 — Gate D4: Frontend Shape & Hosting

**Date:** 2026-06-17 · **Status:** accepted (scaffold landed; pages built in Phase 7)
**Decision:** **Vite + React SPA** (TypeScript, strict), served as a **static bundle by nginx in a
multi-stage Docker image**. The browser talks to one origin: nginx (and the Vite dev server) proxy a
`/api` prefix to the FastAPI backend, so **no CORS** and no API origin baked into the client. Next.js
is rejected.

## Context

Gate D4 (project-plan.md): Next.js vs Vite SPA, taken when UI work begins (now — Phase 7 start).
Unlike D2/D3/D5 this is **not a golden-set ablation** — no retrieval/generation quality rides on it,
so rule #1 (the Phase-4 measurement door) doesn't apply. It's an architecture-fit judgment, recorded
here because rule #2 ("gates, not vibes") wants every gate's rationale written down.

Framing constraints:
- The backend is **FastAPI either way** (the SSE `/ask` stream + `/health`), so Next.js API routes buy
  nothing — there's no second backend to host.
- Deploy target is a **no-GPU CPU tier** (Fly/Render/HF Spaces for the API). The frontend is static
  files; whoever serves them is interchangeable.
- The product is a **portfolio demo** that must survive a 90-second walkthrough and a hard refresh on
  any route. Docker is the deploy unit the rest of the stack already uses (compose for Postgres).

## Why Vite SPA over Next.js

| Consideration | Next.js | **Vite SPA (chosen)** |
|---|---|---|
| Second backend (API routes / SSR server) | yes — a Node server to host + keep async-safe | **none — static files** |
| First paint / SSR | free SSR landing | CSR; fine for an interactive demo behind a loader |
| Hosting | Vercel-shaped (a node runtime) | **any static host or nginx in our existing Docker** |
| Streaming `/ask` (SSE over POST) | same hand-rolled fetch client either way | same |
| Fit with "API is the product" | blurs front/back boundary | **keeps the thin FastAPI seam clean** |

Next's headline win (SSR first paint) is weak for an app whose whole value is an interactive,
streaming query box — there's no content-heavy landing to server-render. Against that, Next adds a
Node server to run, secure, and keep off the event loop — a second moving part for zero capability we
need. SPA + nginx collapses the deploy to static files we already know how to containerize.

## Decision

1. **Vite 6 + React 19 + TypeScript (strict)**, `pnpm`. Strict TS mirrors the backend's pyright-strict
   posture (CLAUDE.md conventions).
2. **One origin, `/api` proxy.** The Vite dev server and the prod nginx both rewrite `/api/*` → the
   backend (stripping the prefix), so the client never knows the API origin and the backend needs **no
   CORS middleware** (keeps rule #7's lean async API untouched).
3. **Docker: multi-stage** (Node build → nginx:alpine serving `dist/`). nginx config is an envsubst
   template (`API_UPSTREAM` configurable per-env) with **SSE buffering disabled** so `/ask` streams
   token-by-token (the "deep mode" watch-it-search UX, 6.7).
4. **Stack:** react-router v7 (routing), TanStack Query (non-streaming endpoints — health now,
   sources/evals later), a hand-rolled fetch SSE client (`src/lib/sse.ts`) because the native
   `EventSource` is GET-only and `/ask` is a POST. Styling = **Tailwind v4**, tokens lifted from the
   `design/` mockups into a CSS-first `@theme`.

## Open / deferred

- **Static host vs nginx-in-Docker for prod.** The image serves static files, so the frontend can also
  go on Vercel/CF Pages if that's cheaper than a container — a deploy-time call (Phase 7 exit / Phase 8),
  not an architecture one. The `/api` rewrite is the only thing that has to move with it.
- **Page implementations** (chat, sources, evals, security, how-it-works) — built against the typed API
  client over the rest of Phase 7; the scaffold ships route stubs.

## Consequences

- **No CORS surface** and no API URL in the bundle — the proxy is the single integration seam, identical
  in dev and prod.
- **Frontend is hostable anywhere** static files go; the container is the default, not a lock-in.
- **Reversible-ish, but the cheapest of the gates to revisit** — it's the presentation layer; nothing in
  `backend/` depends on it. A later move to Next would be a frontend-only rewrite.
- The SSE-over-POST client is **bespoke** (no `EventSource`); it's small and typed against the pydantic
  event models (`src/lib/types.ts`), and is the one piece to keep in lockstep with the backend contract.
