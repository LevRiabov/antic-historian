/** A small colored pill with a leading dot. The hue is data-driven (a per-category
 *  color), so it's an inline style — a dynamic value can't be a static Tailwind
 *  class. Shared by the evals + security tables. */
export function ColorBadge({ label, color }: { label: string; color: string }) {
  return (
    <span
      className="inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-semibold whitespace-nowrap"
      style={{ background: `${color}1a`, color, borderColor: `${color}40` }}
    >
      <span className="h-2 w-2 flex-none rounded-full" style={{ background: color }} aria-hidden />
      {label}
    </span>
  );
}
