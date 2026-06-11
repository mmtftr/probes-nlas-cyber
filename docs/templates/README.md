# Document templates

## `report.typ` — DEFAULT report template

The project-default template for typeset writeups / result reports (PDF).
Use it for any results doc going forward.

**Usage** — import by absolute path and compile with the repo as root:

```typst
#import "/docs/templates/report.typ": report, callout, finding, statrow, accent, accent2, muted
#show: report.with(
  title: "…",
  subtitle: "…",
  author: "exp-NN",
  date: "2026-06-07",
)
= First section
…
```

```bash
typst compile --root <REPO_ROOT> path/to/report.typ path/to/report.pdf
```

**Exports**
- `report(title:, subtitle:, author:, date:, tag:, accent:)` — document show-rule
  (title block, headers, numbered accent headings, hairline tables, mono/code).
- `callout(body, title:, bar:, fill:)` — admonition box (left accent bar).
- `finding(body, label:)` — filled accent box for the thesis / headline.
- `statrow(items)` — KPI row; `items` = array of `(value, label)` pairs.
- colors: `accent` (indigo), `accent2` (brick), `muted`, `soft`, `ink`.

Fonts (all present on this machine): Libertinus Serif (body) · Helvetica Neue
(headings) · Menlo (code).

**Reference render:** `plans/cross-model-probe-generalization/20-fn-fp-token-analysis/report.typ`.
