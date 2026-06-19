/** The per-attack verdict chip: an attack either breached the assistant or was
 *  held. Used in the table cells and the detail comparison headers. */
export function OutcomeChip({ breached }: { breached: boolean }) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-lg border px-3 py-1 text-[12.5px] font-semibold ${
        breached
          ? "border-danger/25 bg-danger-soft text-danger"
          : "border-good/25 bg-good-soft text-good"
      }`}
    >
      {breached ? "Breached ✗" : "Held ✓"}
    </span>
  );
}
