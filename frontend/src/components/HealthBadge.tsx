import { useQuery } from "@tanstack/react-query";

import { getHealth } from "@/lib/api";

// Tiny live indicator that proves the whole stack is wired: React Query ->
// lib/api -> the `/api` proxy -> FastAPI /health. Also the smoke test that the
// dev proxy / nginx rewrite is working.
export function HealthBadge() {
  const { data, isError, isLoading } = useQuery({
    queryKey: ["health"],
    queryFn: ({ signal }) => getHealth(signal),
    refetchInterval: 30_000,
  });

  const ok = !isError && !isLoading && data?.status === "ok";
  const color = isLoading ? "bg-ink-faint" : ok ? "bg-good" : "bg-refuse";
  const label = isLoading ? "connecting" : ok ? `API v${data?.version}` : "API offline";

  return (
    <span className="inline-flex items-center gap-2 text-xs text-ink-soft">
      <span className={`h-2 w-2 rounded-full ${color}`} aria-hidden />
      <span className="font-mono">{label}</span>
    </span>
  );
}
