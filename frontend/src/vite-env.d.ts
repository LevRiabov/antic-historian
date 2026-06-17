/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Override the API base; defaults to "/api" (proxied to FastAPI). */
  readonly VITE_API_BASE?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
