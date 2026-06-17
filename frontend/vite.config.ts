import { fileURLToPath, URL } from "node:url";

import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// In dev, the FastAPI backend has no CORS middleware (by design — see CLAUDE.md
// rule #7 keeps the API lean). So we proxy a single `/api` prefix to it and strip
// the prefix on the way out. nginx does the identical rewrite in prod (nginx.conf),
// which means the app always calls `/api/...` and never has to know the API origin.
const API_TARGET = process.env.VITE_PROXY_TARGET ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: API_TARGET,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
