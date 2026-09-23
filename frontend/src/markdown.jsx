import React from "react";

/**
 * Just enough markdown, rendered as React elements.
 *
 * The answer arrived as markdown from the first day and was printed raw, so a
 * grounded answer about INS-101 opened with a literal `## INS-101` and said
 * `**State: Impeded**`. Cosmetic, and the first thing anyone sees.
 *
 * ## Why not a library
 *
 * `react-markdown` pulls in a dozen transitive packages to render six
 * constructs. The larger objection is that every markdown library's escape
 * hatch is raw HTML, and this text comes from a model reading Confluence pages
 * that anyone at the client can edit. The safe configuration exists; relying on
 * remembering to keep it set does not.
 *
 * So: a parser that emits React elements and has no way to produce HTML. There
 * is no `dangerouslySetInnerHTML` here and nowhere for one to be added — the
 * output type is wrong for it. A `<script>` in a Confluence page renders as the
 * text `<script>`, which is what it is.
 *
 * ## What it covers, and what happens to the rest
 *
 * Headings, emphasis, inline code, bullet and numbered lists, tables, rules,
 * and paragraphs — measured against what the answering prompt actually produces
 * rather than against the CommonMark spec. Anything unrecognised falls through
 * as its own text, which is the failure mode to want: an unsupported construct
 * looks slightly untidy instead of vanishing.
 *
 * Citations are the reason this cannot simply be `white-space: pre-wrap`. Every
 * `[n]` stays a button wired to the evidence rail, inside a heading or a table
 * cell as readily as in a paragraph — the provenance has to survive formatting,
 * because a citation you cannot click is decoration.
 *
 * Citation buttons are teal everywhere, regardless of which coloured section
 * they render in. A citation is the same kind of thing in the Ask answer and
 * in the Weekly Summary narrative; giving it the section's accent instead
 * would make the one truly cross-cutting affordance harder to recognise.
 */

// Ordered by precedence: bold before emphasis, so `**x**` is not read as an
// emphasised `*x*` with stray asterisks.
const INLINE = /(\*\*[^*]+\*\*|__[^_]+__|\*[^*\n]+\*|_[^_\n]+_|`[^`]+`|\[\d+\])/g;

function inline(text, onCite, keyPrefix) {
  return text.split(INLINE).map((part, i) => {
    const key = `${keyPrefix}-${i}`;
    if (!part) return null;

    const cite = part.match(/^\[(\d+)\]$/);
    if (cite) {
      return (
        <button
          key={key}
          type="button"
          className="mx-0.5 cursor-pointer border-0 bg-transparent align-[2px] font-sans text-xs text-teal-800 underline decoration-teal-800/50 underline-offset-2 hover:text-teal-950"
          onClick={() => onCite(Number(cite[1]))}
          aria-label={`Show source ${cite[1]}`}
        >
          {part}
        </button>
      );
    }
    if (/^(\*\*|__)/.test(part)) return <strong key={key} className="font-semibold">{part.slice(2, -2)}</strong>;
    if (/^`/.test(part))
      return (
        <code key={key} className="rounded bg-wash px-1 py-0.5 font-mono text-[0.82em]">
          {part.slice(1, -1)}
        </code>
      );
    if (/^(\*|_)/.test(part)) return <em key={key}>{part.slice(1, -1)}</em>;
    return <React.Fragment key={key}>{part}</React.Fragment>;
  });
}

const HEADING = /^(#{1,6})\s+(.*)$/;
const RULE = /^\s*(?:-{3,}|\*{3,}|_{3,})\s*$/;
const BULLET = /^\s*[-*+]\s+(.*)$/;
const NUMBERED = /^\s*\d+[.)]\s+(.*)$/;
const ROW = /^\s*\|(.*)\|\s*$/;
const DIVIDER = /^\s*\|[\s|:-]+\|\s*$/;

function cells(line) {
  return line.replace(/^\s*\|/, "").replace(/\|\s*$/, "").split("|").map((c) => c.trim());
}

/** Group lines into blocks. A line that starts a new block ends the current
 *  paragraph, so a heading immediately after prose is not swallowed by it. */
function blocks(text) {
  const lines = text.replace(/\r\n/g, "\n").split("\n");
  const out = [];
  let i = 0;

  const startsBlock = (line) =>
    !line.trim() ||
    HEADING.test(line) ||
    RULE.test(line) ||
    BULLET.test(line) ||
    NUMBERED.test(line) ||
    ROW.test(line);

  while (i < lines.length) {
    const line = lines[i];

    if (!line.trim()) {
      i++;
    } else if (HEADING.test(line)) {
      const [, hashes, body] = line.match(HEADING);
      out.push({ type: "heading", level: hashes.length, text: body });
      i++;
    } else if (RULE.test(line)) {
      out.push({ type: "rule" });
      i++;
    } else if (ROW.test(line)) {
      const rows = [];
      while (i < lines.length && ROW.test(lines[i])) {
        if (!DIVIDER.test(lines[i])) rows.push(cells(lines[i]));
        i++;
      }
      out.push({ type: "table", head: rows[0] || [], body: rows.slice(1) });
    } else if (BULLET.test(line) || NUMBERED.test(line)) {
      const ordered = !BULLET.test(line);
      const marker = ordered ? NUMBERED : BULLET;
      const items = [];
      // A blank line between items does not end the list. Markdown calls this
      // a loose list and the model writes them constantly; treating the blank
      // as a terminator produced three consecutive one-item <ul>s, which reads
      // as three lists that happen to be adjacent.
      while (i < lines.length) {
        if (marker.test(lines[i])) {
          items.push(lines[i].match(marker)[1]);
          i++;
        } else if (!lines[i].trim() && marker.test(lines[i + 1] || "")) {
          i++;
        } else {
          break;
        }
      }
      out.push({ type: "list", ordered, items });
    } else {
      // A paragraph runs to the next blank line or block opener. Its own line
      // breaks are dropped, which is what makes a hard-wrapped answer reflow
      // to the reader's column width instead of the model's.
      const parts = [];
      while (i < lines.length && !startsBlock(lines[i])) {
        parts.push(lines[i].trim());
        i++;
      }
      out.push({ type: "paragraph", text: parts.join(" ") });
    }
  }
  return out;
}

const HEADING_SIZE = { 3: "text-[15px]", 4: "text-[13.5px]", 5: "text-[13.5px]", 6: "text-[13.5px]" };
const HEADING_COLOR = { 3: "text-ink", 4: "text-slate-600", 5: "text-slate-600", 6: "text-slate-600" };

export function Markdown({ text, onCite }) {
  const parsed = blocks(text);

  // Headings start at h3: the page owns h1 and the rail owns h2, and a screen
  // reader walking the outline should not find the answer outranking the
  // section it sits in.
  //
  // The offset is measured from the answer's own shallowest heading rather than
  // fixed, because models open at "##" about as often as at "#". A fixed +2 made
  // the most common case render its title as an h4 — smaller than the body text
  // underneath it, which looks like a mistake and reads like one.
  const levels = parsed.filter((b) => b.type === "heading").map((b) => b.level);
  const shift = 3 - Math.min(...levels, 3);

  return parsed.map((block, i) => {
    const key = `b${i}`;
    switch (block.type) {
      case "heading": {
        const level = Math.min(block.level + shift, 6);
        const Tag = `h${level}`;
        return (
          <Tag
            key={key}
            className={`mt-6 mb-2 font-sans font-semibold tracking-tight first:mt-0 ${HEADING_SIZE[level] || "text-[13.5px]"} ${HEADING_COLOR[level] || "text-slate-600"}`}
          >
            {inline(block.text, onCite, key)}
          </Tag>
        );
      }
      case "rule":
        return <hr key={key} className="my-6 border-0 border-t border-rule" />;
      case "list": {
        const Tag = block.ordered ? "ol" : "ul";
        return (
          <Tag key={key} className={`mb-3 pl-5 marker:text-slate-500 ${block.ordered ? "list-decimal" : "list-disc"}`}>
            {block.items.map((item, j) => (
              <li key={j} className="mb-1">
                {inline(item, onCite, `${key}-${j}`)}
              </li>
            ))}
          </Tag>
        );
      }
      case "table":
        return (
          // Wrapped so a wide table scrolls itself rather than widening the
          // column and pushing the evidence rail off screen.
          <div className="mb-3 overflow-x-auto" key={key}>
            <table className="w-full min-w-full border-collapse font-sans text-[13.5px] leading-normal">
              <thead>
                <tr>
                  {block.head.map((cell, j) => (
                    <th
                      key={j}
                      className="whitespace-nowrap border-b border-rule py-1.5 pr-3 text-left font-semibold"
                    >
                      {inline(cell, onCite, `${key}-h${j}`)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {block.body.map((row, j) => (
                  <tr key={j}>
                    {row.map((cell, k) => (
                      <td key={k} className="border-b border-rule py-1.5 pr-3 text-left align-top">
                        {inline(cell, onCite, `${key}-${j}-${k}`)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );
      default:
        return (
          <p key={key} className="mb-3 last:mb-0">
            {inline(block.text, onCite, key)}
          </p>
        );
    }
  });
}
