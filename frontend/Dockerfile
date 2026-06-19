# Multi-stage: build the static SPA with Node, serve it with nginx.
# Stage 1 — build. Pin to the major Node we develop on (24).
FROM node:24-alpine AS build
WORKDIR /app

# corepack ships with Node; activate the pnpm version from package.json so the
# lockfile resolves identically to local dev.
RUN corepack enable

# Install deps first (cached unless the manifests change). pnpm-workspace.yaml
# carries the build-script allowlist (allowBuilds: esbuild), so it must be present
# at install time or pnpm 11 aborts on esbuild's ignored postinstall.
COPY package.json pnpm-lock.yaml* pnpm-workspace.yaml ./
RUN pnpm install --frozen-lockfile

# Build. VITE_API_BASE is baked at build time; default "/api" matches nginx.conf.
COPY . .
ARG VITE_API_BASE=/api
ENV VITE_API_BASE=${VITE_API_BASE}
RUN pnpm build

# Stage 2 — serve. The UNPRIVILEGED nginx image runs as the non-root `nginx` user
# (uid 101) and listens on 8080 (binding :80 needs root). Same envsubst-on-templates
# entrypoint as the stock image, so the template/API_UPSTREAM contract is unchanged.
FROM nginxinc/nginx-unprivileged:1.27-alpine AS runtime
# The nginx image runs envsubst over /etc/nginx/templates/*.template at startup,
# so API_UPSTREAM is configurable per-environment (docker compose / Fly / Render)
# without rebaking the image. Default = the compose service name.
ENV API_UPSTREAM=backend:8000
COPY nginx.conf.template /etc/nginx/templates/default.conf.template
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 8080
# Liveness: the SPA shell must render. Runs as the container user; busybox wget ships
# in the alpine base. Use 127.0.0.1 (not localhost — which can resolve to ::1 first,
# where nginx isn't listening) on the same 8080 the server binds.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget -q -O /dev/null http://127.0.0.1:8080/ || exit 1
# `daemon off` keeps nginx in the foreground as PID 1.
CMD ["nginx", "-g", "daemon off;"]
