/**
 * One accent colour per section, spelled out as literal Tailwind classes.
 *
 * Tailwind finds classes by scanning source text for them; a name built at
 * runtime ("border-" + color + "-200") is invisible to that scan and ships
 * unstyled. So every accent is written out in full exactly once, here, and
 * every section borrows from this object instead of constructing its own
 * strings.
 *
 * Citations are deliberately NOT part of this — they stay teal everywhere
 * (see markdown.jsx) because a citation is the same kind of thing regardless
 * of which section it renders in.
 */
export const ACCENTS = {
  // Ask a question — the main line of work on the page.
  indigo: {
    card: "border-indigo-200 bg-indigo-50/40",
    heading: "text-indigo-700",
    solid: "border border-indigo-700 bg-indigo-700 text-white hover:bg-indigo-800 disabled:bg-slate-300 disabled:border-slate-300",
    chipBorder: "border-indigo-200 text-indigo-800 hover:bg-indigo-50",
    mark: "text-indigo-600",
  },
  // Weekly Summary — a distinct, occasional action, not the everyday flow.
  violet: {
    card: "border-violet-200 bg-violet-50/40",
    heading: "text-violet-700",
    solid: "border border-violet-700 bg-violet-700 text-white hover:bg-violet-800 disabled:bg-slate-300 disabled:border-slate-300",
    mark: "text-violet-600",
  },
  // Evidence — provenance, the same teal citations already use.
  teal: {
    card: "border-teal-200 bg-teal-50/30",
    heading: "text-teal-800",
    mark: "text-teal-700",
  },
  // Waiting on you — the one place amber has always meant "a person must
  // decide"; now it also happens to be this section's colour.
  amber: {
    card: "border-amber-300 bg-amber-50/50",
    heading: "text-amber-800",
    solid: "border border-amber-700 bg-amber-700 text-white hover:bg-amber-800 disabled:bg-slate-300 disabled:border-slate-300",
    mark: "text-amber-700",
  },
  // Decisions — a settled record. Deliberately the quietest section: it is
  // describing what already happened, not asking for anything.
  slate: {
    card: "border-slate-200 bg-slate-50/50",
    heading: "text-slate-700",
    mark: "text-slate-500",
  },
};

/** The card every section is wrapped in, parameterised only by accent. */
export const cardClass = (accent) =>
  `rounded-xl border p-5 md:p-6 ${ACCENTS[accent].card}`;

export const headingClass = (accent) =>
  `mb-3 border-b border-rule pb-2 text-[13px] font-semibold ${ACCENTS[accent].heading}`;
