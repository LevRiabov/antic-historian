# Frontend

React SPA for Antic Historian — chat with inline citations, sources, evals, and
security pages. Talks to the FastAPI backend over a streaming SSE API.

**Gate D4 decided:** Vite SPA (not Next.js), served as a static bundle by nginx
in Docker. The backend is FastAPI either way, so Next API routes bought us
nothing; the SPA is simpler and containerizes cleanly.

## Stack

| Concern | Choice | Notes |
|---|---|---|
| Build / dev server | **Vite 6** + React 19 + TS (strict) | `pnpm` (≈ uv/`uv sync`) |
| Styling | **Tailwind v4** | CSS-first `@theme` in `src/index.css`; tokens lifted from `design/chat-05-hybrid.html` |
| Routing | **react-router v7** | `src/router.tsx` |
| Server state | **TanStack Query** | non-streaming endpoints (health now; sources/evals later) |
| Streaming | hand-rolled fetch SSE | `src/lib/sse.ts` — `POST /ask` can't use native `EventSource` |
| Container | multi-stage Node build → nginx | `Dockerfile` + `nginx.conf.template` |

## Develop

```sh
pnpm install
pnpm dev          # http://localhost:5173  (proxies /api -> http://127.0.0.1:8000)
pnpm typecheck    # tsc strict
pnpm lint
pnpm build        # -> dist/
```

The dev server proxies `/api/*` to the backend (override with `VITE_PROXY_TARGET`),
so run `uv run ahx serve` in `backend/` alongside it. The header's health badge
goes green when the proxy reaches the API.

## Docker

```sh
docker build -t ahx-frontend .
docker run -p 8080:80 -e API_UPSTREAM=host.docker.internal:8000 ahx-frontend
```

nginx serves the static bundle and proxies `/api` to `$API_UPSTREAM` (default
`backend:8000` for compose), with SSE buffering disabled so `/ask` streams live.

## Layout

```
src/
  lib/        api.ts (health + base), sse.ts (typed /ask stream), types.ts (mirrors backend models)
  components/ Layout, HealthBadge, PageStub
  routes/     Chat, Sources, Evals, Security, HowItWorks  (stubs — Phase 7)
  router.tsx  route table
  index.css   Tailwind import + design tokens
design/       static HTML mockups (the source of truth for the look)
```
