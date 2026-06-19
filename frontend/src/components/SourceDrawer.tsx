import { useEffect } from "react";

/*
 * The citation drawer — slides in from the right with the verbatim passage behind
 * a cited marker. Reused across pages: the evals page fetches the passage by id
 * and feeds it here; the chat page passes the Citation it already has from the SSE
 * `sources` event. Purely presentational — the parent owns open/close + data.
 *
 * The structural prop type below is the common ground between ChunkOut (/chunks)
 * and Citation (the SSE event) — both carry author/work/locator/text; pd_basis is
 * present only on ChunkOut, so it's optional here.
 */
export interface DrawerPassage {
  chunk_id: number;
  author: string;
  work_title: string;
  locator: string[];
  text: string;
  pd_basis?: string;
}
export function SourceDrawer({
  open,
  onClose,
  marker,
  passage,
  loading,
  error,
}: {
  open: boolean;
  onClose: () => void;
  marker?: number; // the [n] that was clicked; shown in the header / corner badge
  passage: DrawerPassage | null;
  loading: boolean;
  error: boolean;
}) {
  // Close on Escape while open (matches the chat mock's keyboard behaviour).
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  return (
    <>
      <div
        onClick={onClose}
        aria-hidden
        className={`fixed inset-0 z-40 bg-ink/25 transition-opacity duration-200 ${
          open ? "opacity-100" : "pointer-events-none opacity-0"
        }`}
      />
      <aside
        role="dialog"
        aria-modal="true"
        aria-hidden={!open}
        className={`fixed top-0 right-0 bottom-0 z-50 flex w-[430px] max-w-[92vw] flex-col border-l border-line bg-surface shadow-[-12px_0_40px_rgba(31,35,40,0.14)] transition-transform duration-300 ${
          open ? "translate-x-0" : "translate-x-full"
        }`}
      >
        <header className="flex items-center gap-3 border-b border-line px-[18px] py-4">
          <div className="flex-1">
            <div className="font-serif text-base font-bold text-ink">
              {marker !== undefined ? `Citation [${marker}]` : "Source"}
            </div>
            <div className="text-xs text-ink-faint">Verified public-domain passage</div>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="grid h-8 w-8 place-items-center rounded-lg border border-line-strong bg-surface text-ink-soft hover:bg-accent-soft hover:text-accent-ink"
          >
            ✕
          </button>
        </header>

        <div className="overflow-y-auto p-[18px]">
          {loading ? (
            <DrawerSkeleton />
          ) : error ? (
            <p className="text-sm text-refuse">Couldn’t load this passage.</p>
          ) : passage ? (
            <SourceCard passage={passage} marker={marker} />
          ) : (
            <p className="text-sm text-ink-faint">
              This passage is no longer in the corpus (the index was rebuilt since this run).
            </p>
          )}
        </div>

        <footer className="mt-auto flex items-center gap-2 border-t border-line px-[18px] py-3 text-[11.5px] text-ink-faint">
          <span className="text-good" aria-hidden>
            ●
          </span>
          Quoted verbatim from the EU-hosted public-domain corpus
        </footer>
      </aside>
    </>
  );
}

function SourceCard({ passage, marker }: { passage: DrawerPassage; marker?: number }) {
  const locator = passage.locator.join(" · ");
  return (
    <div className="rounded-2xl border border-line bg-[#fcfbf9] p-4">
      {marker !== undefined && (
        <span className="mb-3 inline-flex h-6 w-6 items-center justify-center rounded-md bg-accent text-xs font-bold text-white">
          {marker}
        </span>
      )}
      <div className="font-serif text-[17px] leading-tight font-bold text-ink">
        {passage.work_title}
      </div>
      <div className="mt-1 inline-flex items-center gap-1.5 text-[12.5px] text-accent-ink">
        <strong>{passage.author}</strong>
        {locator && (
          <>
            <span aria-hidden>·</span>
            <code className="font-mono">{locator}</code>
          </>
        )}
      </div>

      <blockquote className="my-4 rounded-r-[10px] border-l-[3px] border-accent bg-surface px-4 py-3.5 font-serif text-[15px] leading-relaxed text-[#2b2620]">
        {passage.text}
      </blockquote>

      <div className="mt-4 flex flex-col gap-2 border-t border-line pt-3.5 text-xs text-ink-faint">
        <div className="flex flex-wrap items-center gap-2">
          <span className="inline-flex items-center gap-1.5 rounded-full border border-good/25 bg-good-soft px-2 py-0.5 text-good">
            <span aria-hidden>✓</span> Public domain
          </span>
          {passage.pd_basis && <span>{passage.pd_basis}</span>}
        </div>
        <div>Retrieved by dense semantic search · contextualized passage</div>
      </div>
    </div>
  );
}

function DrawerSkeleton() {
  return (
    <div className="animate-pulse rounded-2xl border border-line bg-[#fcfbf9] p-4">
      <div className="h-6 w-6 rounded-md bg-line/60" />
      <div className="mt-3 h-5 w-2/3 rounded bg-line/60" />
      <div className="mt-2 h-3 w-1/3 rounded bg-line/40" />
      <div className="mt-4 h-24 rounded bg-line/40" />
    </div>
  );
}
