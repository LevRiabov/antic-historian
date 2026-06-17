# Multi-stage: build the static SPA with Node, serve it with nginx.
# Stage 1 — build. Pin to the major Node we develop on (24).
FROM node:24-alpine AS build
WORKDIR /app

# corepack ships with Node; activate the pnpm version from package.json so the
# lockfile resolves identically to local dev.
RUN corepack enable

# Install deps first (cached unless the manifests change).
COPY package.json pnpm-lock.yaml* ./
RUN pnpm install --frozen-lockfile

# Build. VITE_API_BASE is baked at build time; default "/api" matches nginx.conf.
COPY . .
ARG VITE_API_BASE=/api
ENV VITE_API_BASE=${VITE_API_BASE}
RUN pnpm build

# Stage 2 — serve. nginx serves the static bundle and proxies /api to the backend.
FROM nginx:1.27-alpine AS runtime
# The nginx image runs envsubst over /etc/nginx/templates/*.template at startup,
# so API_UPSTREAM is configurable per-environment (docker compose / Fly / Render)
# without rebaking the image. Default = the compose service name.
ENV API_UPSTREAM=backend:8000
COPY nginx.conf.template /etc/nginx/templates/default.conf.template
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
# `daemon off` keeps nginx in the foreground as PID 1.
CMD ["nginx", "-g", "daemon off;"]
