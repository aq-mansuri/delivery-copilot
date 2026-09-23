/**
 * Offline checks for the answer renderer.
 *
 *     npm run check
 *
 * The UI has no test runner and adding one for a single module is not worth the
 * dependency, but this module is exactly the kind that breaks silently: it
 * renders model output, and the failure mode is cosmetic until someone notices
 * the demo is showing literal `##`. So it gets checked, with the bundler and
 * React that are already installed and nothing else. No browser, no network.
 *
 * The cases are the constructs the answering prompt actually produces, plus the
 * two that regressed: loose lists, and the heading level a "##"-led answer lands
 * on. Anything asserting "is not raw markdown" belongs here rather than in a
 * reviewer's eyes.
 */
import { build } from "esbuild";
import { unlinkSync } from "fs";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";

await build({
  entryPoints: ["src/markdown.jsx"],
  bundle: true,
  outfile: "._md.check.mjs",
  format: "esm",
  jsx: "automatic",
  external: ["react", "react/jsx-runtime"],
});
const { Markdown } = await import("./._md.check.mjs");
unlinkSync("._md.check.mjs");

const render = (text) =>
  renderToStaticMarkup(
    React.createElement(Markdown, { text, onCite: () => {} })
  );

const failures = [];
const check = (name, text, assertion) => {
  let html = "";
  try {
    html = render(text);
    const problem = assertion(html);
    if (problem) failures.push(`${name}\n    ${problem}\n    got: ${html}`);
  } catch (e) {
    failures.push(`${name}\n    threw: ${e.message}`);
  }
};

const has = (needle) => (html) =>
  html.includes(needle) ? null : `expected to contain ${JSON.stringify(needle)}`;
const lacks = (needle) => (html) =>
  html.includes(needle) ? `expected NOT to contain ${JSON.stringify(needle)}` : null;
const all = (...fns) => (html) => fns.map((f) => f(html)).find(Boolean) || null;

// Assertions below check structure and content — real element, real text,
// really clickable — rather than an exact class string. The renderer now
// carries Tailwind utility classes that are free to change with the theme;
// coupling this check to their exact spelling would make it fail on every
// restyle for no safety reason, which teaches people to stop reading it.

check(
  "headings are elements, not literal hashes",
  "## INS-101\n\nIt is stale. [1]",
  all(has("<h3"), has(">INS-101</h3>"), lacks("## INS-101"))
);

check(
  "a '#'-led answer also starts at h3",
  "# Summary\n\n## Detail\n\nText. [1]",
  all(has(">Summary</h3>"), has(">Detail</h4>"))
);

check(
  "bold is an element, not literal asterisks",
  "**State: Impeded** — no movement. [1]",
  all(has("<strong"), has(">State: Impeded</strong>"), lacks("**State"))
);

check(
  "citations stay clickable buttons",
  "INS-101 is stale [1] and INS-102 is blocked [2].",
  all(
    has("<button"),
    has(">[1]</button>"),
    has('aria-label="Show source 1"'),
    has('aria-label="Show source 2"')
  )
);

check(
  "a citation inside a heading is still a button",
  "## Findings [1]\n\nBody. [1]",
  all(
    has(">Findings <button"),
    has('aria-label="Show source 1"'),
    has(">[1]</button></h3>")
  )
);

check(
  "blank-separated bullets stay one list",
  "- first [1]\n\n- second [1]\n\n- third [1]",
  (html) => {
    const lists = (html.match(/<ul[^>]*>/g) || []).length;
    const items = (html.match(/<li[^>]*>/g) || []).length;
    return lists === 1 && items === 3
      ? null
      : `expected 1 <ul> with 3 <li>, saw ${lists} and ${items}`;
  }
);

check(
  "numbered lists render as ol",
  "1. first [1]\n2. second [1]",
  all(has("<ol"), lacks("1. first"))
);

check(
  "tables render and scroll in their own box",
  "| Item | Target |\n| --- | --- |\n| Export [1] | Mid Q4 |",
  all(
    has("overflow-x-auto"),
    has(">Item</th>"),
    has(">Mid Q4</td>"),
    has("<button"),
    has('aria-label="Show source 1"')
  )
);

check(
  "a hard-wrapped paragraph reflows",
  "The compliance export\nis proceeding as planned. [1]",
  all(has("export is proceeding"), lacks("\n"))
);

check(
  "model output cannot inject markup",
  'A page says <img src=x onerror="alert(1)"> about it. [1]',
  all(lacks("<img"), has("&lt;img"))
);

check(
  "unsupported constructs degrade to text rather than vanishing",
  "> a blockquote nobody parses [1]",
  has("a blockquote nobody parses")
);

check("empty input does not throw", "", () => null);

if (failures.length) {
  console.error(`\n${failures.length} markdown check(s) failed:\n`);
  for (const f of failures) console.error(`  ✗ ${f}\n`);
  process.exit(1);
}
console.log("markdown: all checks passed");
