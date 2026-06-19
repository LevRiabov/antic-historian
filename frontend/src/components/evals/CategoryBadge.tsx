import { ColorBadge } from "@/components/ui/ColorBadge";
import { CAT_META } from "@/lib/evals";
import type { Category } from "@/lib/types";

/** Colored category pill — the hue comes from CAT_META so the badge, the filter
 *  tabs, and the detail sidecard all read the same palette. */
export function CategoryBadge({ category }: { category: Category }) {
  return <ColorBadge label={category} color={CAT_META[category].color} />;
}
