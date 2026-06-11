// [ai-generated] — Default Typst report template for the probes-nlas-cyber project.
//
// This is the PROJECT-DEFAULT template for typeset writeups / result reports.
// Usage (compile with the repo root so the absolute import resolves):
//
//   #import "/docs/templates/report.typ": report, callout, finding, statrow
//   #show: report.with(
//     title: "…", subtitle: "…", author: "…", date: "2026-06-07")
//   = Section …
//
//   Compile:  typst compile --root <REPO_ROOT> path/to/report.typ
//
// Exports: report (document show-rule), callout (admonition box),
//          finding (accent key-finding box), statrow (inline KPI row).

#let accent = rgb("#2f4b7c")       // deep indigo
#let accent2 = rgb("#b1442e")      // brick (for emphasis / negatives)
#let soft = rgb("#eef2f9")
#let ink = rgb("#1b2330")
#let muted = luma(115)

#let _sans = ("Helvetica Neue", "New Computer Modern Sans")
#let _serif = ("Libertinus Serif", "New Computer Modern")
#let _mono = ("Menlo", "DejaVu Sans Mono")

// ── admonition / callout box ────────────────────────────────────────────────
#let callout(body, title: none, bar: accent, fill: soft) = block(
  width: 100%, inset: (x: 11pt, y: 9pt), radius: 4pt, fill: fill,
  stroke: (left: 3pt + bar), above: 1.0em, below: 1.0em,
)[
  #if title != none [#text(weight: "bold", fill: bar, font: _sans, size: 9.5pt)[#upper(title)]#linebreak()#v(1pt)]
  #body
]

// ── headline "finding" box (accent fill, for the thesis) ────────────────────
#let finding(body, label: "Key finding") = block(
  width: 100%, inset: (x: 12pt, y: 11pt), radius: 4pt, fill: accent,
  above: 1.1em, below: 1.1em,
)[
  #text(fill: rgb("#cdd8ec"), font: _sans, weight: "bold", size: 8.5pt)[#upper(label)]
  #v(3pt)
  #text(fill: white, size: 10.5pt)[#body]
]

// ── inline KPI row: pass an array of (value, label) pairs ────────────────────
#let statrow(items) = {
  grid(
    columns: items.len() * (1fr,),
    column-gutter: 8pt,
    ..items.map(it => block(
      width: 100%, inset: 8pt, radius: 4pt, fill: soft,
      stroke: 0.5pt + luma(220),
    )[
      #align(center)[
        #text(font: _sans, weight: "bold", size: 15pt, fill: accent)[#it.at(0)] \
        #text(size: 8pt, fill: muted)[#it.at(1)]
      ]
    ]),
  )
}

// ── document show-rule ──────────────────────────────────────────────────────
#let report(
  title: "",
  subtitle: none,
  author: none,
  date: none,
  tag: "[ai-generated]",
  accent: accent,
  doc,
) = {
  set document(title: title)
  set page(
    paper: "us-letter",
    margin: (x: 2.1cm, top: 2.2cm, bottom: 1.9cm),
    numbering: "1",
    header: context {
      if counter(page).get().first() > 1 {
        set text(size: 8pt, fill: muted, font: _sans)
        grid(columns: (1fr, auto), [#title], [])
        v(-6pt)
        line(length: 100%, stroke: 0.4pt + luma(215))
      }
    },
  )
  set text(font: _serif, size: 10pt, fill: ink, lang: "en")
  set par(justify: true, leading: 0.62em, spacing: 0.95em)
  show link: set text(fill: accent)
  show raw: set text(font: _mono, size: 8.5pt)
  show raw.where(block: false): box.with(
    fill: luma(243), inset: (x: 2.5pt), outset: (y: 2.5pt), radius: 2pt)

  // headings
  set heading(numbering: "1")
  show heading: set text(font: _sans)
  show heading.where(level: 1): it => block(above: 1.5em, below: 0.7em)[
    #text(fill: accent, size: 13.5pt, weight: "bold")[#it]
    #v(-0.35em)
    #line(length: 100%, stroke: 0.9pt + accent)
  ]
  show heading.where(level: 2): set text(size: 10.5pt, fill: ink)

  // tables: light header, hairline grid
  set table(stroke: 0.5pt + luma(220), inset: (x: 6pt, y: 5pt), align: left + horizon)
  show table.cell.where(y: 0): set text(font: _sans, weight: "bold", size: 8.5pt, fill: ink)
  show table.cell.where(y: 0): set table.cell(fill: luma(236))

  // ── title block ──
  block(width: 100%)[
    #line(length: 38pt, stroke: 3pt + accent)
    #v(5pt)
    #text(font: _sans, weight: "bold", size: 20pt, fill: ink)[#title]
    #if subtitle != none [
      #v(4pt)
      #text(font: _sans, size: 11pt, fill: muted)[#subtitle]
    ]
    #v(6pt)
    #text(size: 8.5pt, fill: muted)[
      #(( author, date, tag ).filter(x => x != none).join("  ·  "))
    ]
  ]
  v(8pt)

  doc
}
