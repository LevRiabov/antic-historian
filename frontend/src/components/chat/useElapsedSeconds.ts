import { useEffect, useState } from "react";

/*
 * A live whole-seconds counter that runs while `running` is true and resets when it
 * flips back on. Used by the streaming indicators so a long time-to-first-token reads
 * as "working (14s)" rather than a frozen spinner — the single biggest defence
 * against the "is this stuck?" fear on a slow provider. It's a reassurance display,
 * not a measurement (the authoritative latency is the client-measured elapsedMs on
 * the done event), so counting from mount is close enough.
 */
export function useElapsedSeconds(running: boolean): number {
  const [seconds, setSeconds] = useState(0);
  useEffect(() => {
    if (!running) return;
    const start = Date.now();
    setSeconds(0);
    const id = setInterval(() => setSeconds(Math.floor((Date.now() - start) / 1000)), 1000);
    return () => clearInterval(id);
  }, [running]);
  return seconds;
}
