import { afterEach, describe, expect, it, vi } from "vitest";

import { getSessionId } from "@/lib/session";

/* The suite runs in a `node` env (vitest.config.ts) where `localStorage` is
 * undefined, so these tests stub it explicitly to exercise both the persistent
 * path and the storage-disabled fallback. */

afterEach(() => {
  vi.unstubAllGlobals();
  vi.resetModules();
});

function fakeStorage(): Storage {
  const map = new Map<string, string>();
  return {
    getItem: (k) => map.get(k) ?? null,
    setItem: (k, v) => void map.set(k, v),
    removeItem: (k) => void map.delete(k),
    clear: () => map.clear(),
    key: () => null,
    get length() {
      return map.size;
    },
  } as Storage;
}

describe("getSessionId", () => {
  it("returns a non-empty id and persists it across calls", async () => {
    vi.stubGlobal("localStorage", fakeStorage());
    const { getSessionId: fresh } = await import("@/lib/session");
    const first = fresh();
    expect(first).toMatch(/.+/);
    expect(fresh()).toBe(first); // stable, not regenerated
  });

  it("reuses an id already in storage rather than minting a new one", async () => {
    const store = fakeStorage();
    store.setItem("ahx-session-id", "preexisting-id");
    vi.stubGlobal("localStorage", store);
    const { getSessionId: fresh } = await import("@/lib/session");
    expect(fresh()).toBe("preexisting-id");
  });

  it("falls back to a stable in-memory id when storage throws", () => {
    vi.stubGlobal("localStorage", {
      getItem: () => {
        throw new Error("storage disabled");
      },
      setItem: () => {
        throw new Error("storage disabled");
      },
    } as unknown as Storage);
    const first = getSessionId();
    expect(first).toMatch(/.+/);
    expect(getSessionId()).toBe(first); // memoized, still stable per session
  });
});
