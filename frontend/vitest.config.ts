import { fileURLToPath, URL } from "node:url";

import { defineConfig } from "vitest/config";

// The lib/ layer under test is framework-free (pure transforms + a fetch-driven
// SSE generator), so the suite runs in a plain `node` environment — no jsdom, no
// React. globals stay off; tests import { describe, it, expect } from "vitest"
// explicitly, matching the repo's verbatim-module-syntax posture.
export default defineConfig({
  resolve: {
    alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) },
  },
  test: {
    environment: "node",
    globals: false,
    include: ["src/**/*.test.ts"],
  },
});
