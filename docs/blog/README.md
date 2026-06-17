[ai-generated]

# Blog post — figures & publishing guide

How the "Validating Vulnerability Probes" post and its figures are produced and
shipped to the Quarto site. Read this before regenerating figures or touching
the post.

## What's in this dir

- `outline.typ` / `outline.pdf` — the post outline (Slack-edited); human-facing.
- `user-outline.md` — the user's own outline notes (their framing, verbatim).
- `draft-claims.typ` / `draft-claims.pdf` — claims-&-evidence edition: one
  fig6-style plot per claim, mapped to result JSONs.
- `body-details.md` — long-form notes feeding the prose.
- `figbase.py` — shared styling/theming: light+dark `PALETTES`, `set_theme()`,
  `save()`, the grid helpers (`style_ax` / `style`), `size_axis_arrow()`, and
  `publish()`. `make_figs.py` does `from figbase import *`; re-skin by editing
  this file only.
- `make_figs.py` — **all** figure bodies in one script: narrative **FIG-A..H** →
  `figs/fig_*.png` and the per-claim **CLAIM-*** plots → `figs/claim_*.png`
  (the latter formerly lived in a separate `make_claim_figs.py`). After a run it
  auto-copies every PNG into the post's `figs/` (see `publish()` / `POST_FIGS_DIR`),
  so there is no manual copy step.
- `figs/` — generated PNGs: `<name>.png` (light) + `<name>-dark.png` (dark).
  Gitignored payloads (`logits_*.npz`) are NOT here; see FIG-E below.

The **published post** lives in the personal Quarto site repo, not here:
`posts/vulnerability-probes-lexical-ceiling/index.qmd` (+ a copy of `figs/`,
kept in sync by the auto-copy above).

## Regenerate the figures

```bash
# FIG-E needs the exp-16 token npz + dataset (gitignored); the others don't.
export EXP16_NPZ="plans/cross-model-probe-generalization/16-token-logit-dump/results/logitdump_Qwen_Qwen2.5-Coder-32B-Instruct/logits_layer25.npz"
export DATASET_JSONL="data/dataset.jsonl"   # FIG_E_EID defaults to 105
uv run --with matplotlib --with numpy python docs/blog/make_figs.py
```

The one script renders **every figure twice** (light + dark) and copies them all
into the post in one run. Data is read live from result JSONs under
`plans/cross-model-probe-generalization/*`; the few hardcoded tables (e.g. exp-18
MLP AUCs, exp-25 per-CWE CIs) carry a provenance comment at the value. FIG-E is
**skipped** if its npz is absent. Set `POST_FIGS_DIR` to redirect (or disable, by
pointing at a missing dir) the auto-copy into the site repo.

Figure → source map: `make_figs.py`'s module docstring lists the FIG-A..F
sources; the `claim_*` functions' docstrings carry the claim codes (RES/LNG/SUR)
those plots back.

## House style (matches the post)

- **No em-dashes.** The metric (`tokens_code_auc`) is defined once in prose and
  **never** appears in prose or figure labels — figures say "token-level AUC
  (live code)".
- Agent-drafted in the user's voice, marked `[ai-generated]`, pending the user's
  pass. Keep figures legible at blog width (~780 px body).

## Light + dark variants

Every figure ships a light and a dark PNG so it sits correctly in either site
theme. The dark variant is what made this non-trivial; details below.

- **Light** (`<name>.png`): white bg, original palette. Byte-for-byte stable —
  regenerating must not change it (verify with `cmp`). Only the dark palette and
  shared *ink* knobs are new; the light values are pinned.
- **Dark** (`<name>-dark.png`): **transparent** bg + light ink + lifted accent
  colors, so it floats on the dark page (`#0b0b0b`).

## For agents

### The theming system

- Lives in **`figbase.py`** (imported by `make_figs.py` via `from figbase import
  *`). A `PALETTES = {"light": {...}, "dark": {...}}` dict + module-level color
  globals (`ACCENT`, `BRICK`, `BRICK_LIGHT`, `BRICK_MID`, `GRID`, `FAINT`, …).
  `set_theme(name)` repoints those globals and the matplotlib ink/bg rcParams
  (`text.color`, `axes.edgecolor`, `figure.facecolor`, …). The `__main__` loop
  calls `set_theme` for `"light"` then `"dark"` and re-runs all figures.
- **Cross-module gotcha:** the color globals live in `figbase`, but the figure
  bodies in `make_figs.py` reference the bare names. So `figbase.set_theme(name,
  *mirror_into)` writes the colors into both its own namespace (for its helpers)
  and every namespace passed in; `make_figs.set_theme(name)` forwards its
  `globals()`. A naive `from figbase import ACCENT` would snapshot `None` — the
  mirror is what keeps the bare names live across the import boundary.
- Two grid helpers: `style_ax` (grids one axis, the narrative figures) and
  `style` (grids both, the `claim_*` figures). `BRICK_LIGHT` is fig 3's pale
  qwen-MLP tint; `BRICK_MID` is the distinct qwen-7B line in `claim_capacity` —
  different roles, kept as separate palette entries on purpose.
- **Figure bodies are theme-agnostic**: they reference the globals, never raw
  hex. To re-skin, edit the palette dicts only — do not touch figure bodies.
- `save(fig, name)` writes `<name><SUFFIX>.png` (`SUFFIX=""` light, `"-dark"`
  dark), `transparent=True` for dark, and `bbox_inches="tight"`.
- **Typography:** Inter (bundled TTFs in `fonts/`, registered at import; falls
  back Helvetica Neue → Arial → DejaVu). Palette is Okabe-Ito + ColorBrewer
  (`brick`=vermillion, `brick_mid`=orange, gemma ramp `CAP_RAMP`=Blues), tuned
  for CVD-safety and WCAG contrast on white and on `#0b0b0b`.
- **Width-consistent fonts:** every figure displays at the same ~780px column, so
  a wider figure is downscaled more and its text looks smaller. `save()` rescales
  every text element by `min(SCALE_CAP, FONT_BUMP * width / REF_WIDTH)` so
  displayed size is consistent (and a touch larger). `REF_WIDTH`/`FONT_BUMP`/
  `SCALE_CAP` are the knobs. The size-axis arrow is registered via
  `fig._bbox_extra` so the tight crop keeps it. `save(..., rescale=False)` opts a
  figure out (the monospace code heatmaps).
- **Secondary-metric fill:** in the paired-bar figures the secondary readout (MLP
  in `fig_g`, verbalized in `claim_verbalized`) is a light fill + same-hue `////`
  hatch, so the metric type reads by texture as well as colour. `size_axis_arrow`
  marks an x-axis ordered by `MODEL_ORDER_BY_SIZE`.
- **FIG-E** (token heatmap painted behind monospace code) is special-cased: it
  switches the red colormap per theme (light = constant-value light reds with
  dark text; dark = low-value reds so cool tokens fade into the page and hot
  tokens stay dark enough for light text). Its eid is pinned (105) so the dark
  variant matches the published light frame.

### Publishing the dark figures to the Quarto site

This is the fiddly part. The site toggles `body.quarto-dark` / `body.quarto-light`
(driven by OS preference *and* the manual switch). The swap is **CSS-only** (no
JS, no load flash) and keeps **one `<img>` per figure** so Quarto's figure
numbering / `@fig-` cross-refs survive.

1. **Copy** `figs/*-dark.png` (and any changed light PNG) into the post's
   `figs/` in the site repo. `make_figs.py` does this automatically at the end of
   a run (`publish()` → `POST_FIGS_DIR`); the manual copy is only needed if you
   disabled it. A live `quarto preview` won't pick the new PNGs up until the
   `.qmd` re-renders, so `touch index.qmd` (and hard-refresh the browser).
2. **Each image line** carries the class + the dark URL in a custom property:
   ```
   ![cap](figs/NAME.png){#fig-x .themed-fig style="--dark-fig:url('figs/NAME-dark.png')"}
   ```
   (pandoc passes `style` and `.themed-fig` straight through to the `<img>`.)
3. **Front matter** lists the dark PNGs as resources — Quarto's link scanner
   does **not** see URLs inside a CSS `url()`, so they won't be copied to
   `_site` otherwise:
   ```yaml
   resources:
     - figs/*-dark.png
   ```
4. **The swap rule lives in the site's `assets/head.html`** (document `<head>`),
   NOT in `theme.scss`:
   ```html
   <style> body.quarto-dark img.themed-fig { content: var(--dark-fig); } </style>
   ```

#### Why the rule must be in `<head>`, not `theme.scss` (the gotcha)

A `url()` carried in a CSS custom property is resolved relative to the base URL
of the stylesheet that **consumes** the `var()`, not the page. `theme.scss`
compiles into `/site_libs/bootstrap/…`, so a page-relative `figs/x-dark.png`
resolves to `/site_libs/bootstrap/figs/x-dark.png` → **404**. The `<head>`
`<style>` shares the document's base URL, so the relative path resolves to
`/posts/<slug>/figs/…` correctly. You also can't force an absolute path in the
inline style: **Quarto rewrites absolute `url()` paths back to page-relative**
during render. Keep the inline URL relative; keep the rule in `<head>`.

#### Verify after rendering

```bash
quarto render posts/<slug>/index.qmd
ls _site/posts/<slug>/figs/*-dark.png | wc -l          # all variants copied?
# In a browser on the preview, dark mode, for an img.themed-fig:
#   getComputedStyle(img).content  → url(".../posts/<slug>/figs/NAME-dark.png")
#   that URL must fetch 200 (NOT resolve under /site_libs/…)
```
Sanity: in **light** mode `getComputedStyle(img).content` is `normal` (no swap,
shows the light src); in **dark** mode it is the `-dark.png` URL.
