# [ai-generated]
"""Shared figure styling/theming for the blog figures (used by make_figs.py).

Holds the light/dark palettes, `set_theme()`, `save()`, the grid helpers
(`style_ax` / `style`), the size-axis arrow, and `publish()` (auto-copy to the
post). `make_figs.py` does `from figbase import *` and defines a thin
`set_theme(name)` that forwards its own `globals()` so the figure bodies can keep
referencing the bare color names (ACCENT, GRID, ...).

The cross-module trick: color globals live HERE (so the helpers below see them),
but `set_theme(name, *mirror_into)` also writes them into every namespace passed
in `mirror_into`. make_figs passes its `globals()`, so its figure bodies resolve
the same bare names. Re-skin by editing PALETTES only; never touch figure bodies.
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figs"
FIGS.mkdir(exist_ok=True)

# ---- typography ----------------------------------------------------------
# Register the bundled Inter TTFs (docs/blog/fonts/) so figures render in Inter
# deterministically; fall back through Helvetica Neue / Arial / DejaVu if absent.
for _f in sorted((HERE / "fonts").glob("Inter-*.ttf")):
    try:
        fm.fontManager.addfont(str(_f))
    except Exception:
        pass
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans", "sans-serif"],
    "axes.unicode_minus": False,
})

# Width-consistent text. Figures are authored at different widths but all display
# at the same ~780px blog column, so a wider figure is downscaled more and its
# text ends up looking smaller. save() multiplies every text element's size by
# (figure_width / REF_WIDTH) * FONT_BUMP, so the DISPLAYED size is constant across
# figures (REF_WIDTH = the common single-panel width) and a touch larger overall.
REF_WIDTH = 7.2
FONT_BUMP = 1.12  # width-normalize AND enlarge ~12% (model labels shortened to
                  # "qwen 7B"/"qwen 32B" so the single-panel figures don't collide).
SCALE_CAP = 1.4   # cap the enlargement so the widest 3-panel figures get bigger
                  # text (the actual complaint) without over-growing into crowding.

# Where the published Quarto post keeps its copy of the figures. After a run the
# generated PNGs are copied here automatically (no manual copy step); override
# with POST_FIGS_DIR, skipped silently if the dir is absent (site repo not on
# this machine).
POST_FIGS_DIR = Path(os.environ.get(
    "POST_FIGS_DIR",
    Path.home() / "p/personal/mmtf.dev/posts/vulnerability-probes-lexical-ceiling/figs"))


def publish(src=FIGS, dst=None):
    """Copy every generated PNG from `src` into the post's figs/ (POST_FIGS_DIR)."""
    import shutil
    dst = Path(dst) if dst is not None else POST_FIGS_DIR
    if not dst.exists():
        print(f"(post figs dir {dst} absent - skipping copy)")
        return
    pngs = sorted(src.glob("*.png"))
    for p in pngs:
        shutil.copy2(p, dst / p.name)
    print(f"copied {len(pngs)} PNGs to {dst}")


# ---- theming -------------------------------------------------------------
# Every figure is rendered twice: a light variant ("<name>.png", white bg) and a
# dark variant ("<name>-dark.png", transparent bg + light ink + lifted accents)
# for the dark site theme.
# Palette (2026-06-14): Okabe-Ito + ColorBrewer, picked for CVD-safety + WCAG
# contrast on white and on #0b0b0b. `brick` = Okabe vermillion (the qwen / "red"
# contrast hue), `brick_mid` = Okabe orange (qwen-7B); the gemma size ramp is
# ColorBrewer Blues (CAP_RAMP, in make_figs). Light ink softened off pure black;
# light `faint` clears 4.5:1 for small labels; dark ink stays #f2f2f2 (pure white
# halates on near-black). `accent` blue kept (strong, CVD-distinct from orange).
PALETTES = {
    "light": dict(accent="#2f4b7c", brick="#d55e00", green="#2f7d50", gray="#9aa3af",
                  light="#cdd5e3", brick_light="#f4cdb4", brick_mid="#e69f00", ink="#1a1a1a", muted="#555555",
                  faint="#6e6e6e", ebar="#333333", muted2="#777777", grid="#e3e7ee",
                  bargray="#bbbbbb", bg="white"),
    "dark": dict(accent="#6f97ff", brick="#e8552e", green="#5fb887", gray="#9aa3af",
                 light="#9fb0cc", brick_light="#e0a48a", brick_mid="#f0a35e", ink="#f2f2f2", muted="#c9c9c9",
                 faint="#9c9c9c", ebar="#cfcfcf", muted2="#a9a9a9", grid="#30353f",
                 bargray="#6a6a6a", sep="#2a2f3a", note="#a9a9a9", arrow="#a9a9a9", bg="none"),
}
# light keeps the original one-off shades exactly (fig_d separators, fig_f notes)
PALETTES["light"].update(sep="#dadfe8", note="#666666", arrow="#999999")

THEME, SUFFIX = "light", ""
ACCENT = BRICK = GREEN = GRAY = LIGHT = BRICK_LIGHT = BRICK_MID = MUTED = FAINT = EBAR = MUTED2 = GRID = BARGRAY = SEP = NOTE = ARROW = None


def set_theme(name, *mirror_into):
    """Repoint the color globals + matplotlib ink/bg rcParams to one theme. The
    globals are set HERE (for the helpers below) and mirrored into every namespace
    dict in `mirror_into` (make_figs passes its globals() so its figure bodies see
    the bare names)."""
    p = PALETTES[name]
    vals = dict(
        THEME=name, SUFFIX="" if name == "light" else "-dark",
        ACCENT=p["accent"], BRICK=p["brick"], GREEN=p["green"], GRAY=p["gray"],
        LIGHT=p["light"], BRICK_LIGHT=p["brick_light"], BRICK_MID=p["brick_mid"],
        MUTED=p["muted"], FAINT=p["faint"], EBAR=p["ebar"], MUTED2=p["muted2"],
        GRID=p["grid"], BARGRAY=p["bargray"], SEP=p["sep"], NOTE=p["note"], ARROW=p["arrow"])
    globals().update(vals)
    for ns in mirror_into:
        ns.update(vals)
    plt.rcParams.update({
        "text.color": p["ink"], "axes.labelcolor": p["ink"], "axes.titlecolor": p["ink"],
        "axes.edgecolor": p["ink"], "xtick.color": p["ink"], "ytick.color": p["ink"],
        "figure.facecolor": p["bg"], "axes.facecolor": p["bg"], "savefig.facecolor": p["bg"],
    })


def _rescale_fonts(fig, scale):
    """Multiply every text element's font size by `scale` (so displayed text is
    width-consistent across figures). Called from save()."""
    texts = list(fig.texts)  # suptitle and any figure-level text
    for ax in fig.get_axes():
        texts += [ax.title, ax.xaxis.label, ax.yaxis.label]
        texts += ax.get_xticklabels() + ax.get_yticklabels() + list(ax.texts)
        leg = ax.get_legend()
        if leg is not None:
            texts += list(leg.get_texts())
            lt = leg.get_title()
            if lt is not None and lt.get_text():
                texts.append(lt)
    for t in texts:
        t.set_fontsize(t.get_fontsize() * scale)


def save(fig, name, transparent=None, rescale=True):
    """Write FIGS/<name><suffix>.png; dark variants are transparent so they sit on
    the page's own background. `rescale` applies the width-consistent font scaling
    (off for the monospace code heatmaps, whose text is tied to a char grid)."""
    if transparent is None:
        transparent = THEME == "dark"
    if rescale:
        _rescale_fonts(fig, min(SCALE_CAP, FONT_BUMP * fig.get_size_inches()[0] / REF_WIDTH))
    # bbox="tight" expands the crop to fit rescaled text, but ignores annotation
    # arrows drawn outside the axes (the size-axis arrow) -- pass them explicitly.
    extra = getattr(fig, "_bbox_extra", None)
    fig.savefig(FIGS / f"{name}{SUFFIX}.png", dpi=200, transparent=transparent,
                bbox_inches="tight", pad_inches=0.04, bbox_extra_artists=extra)
    plt.close(fig)


def style_ax(ax):
    """Narrative-figure grid: spines off, single-axis grid (y unless the figure
    has a y-axis category, in which case x), behind data."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y" if ax.get_xlabel() == "" else "x", lw=0.4, color=GRID, zorder=0)
    ax.set_axisbelow(True)


def style(ax):
    """Like style_ax but grids both axes (the CLAIM-* figures' convention)."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(lw=0.4, color=GRID, zorder=0)
    ax.set_axisbelow(True)


def size_axis_arrow(ax, y=-0.30, width=0.34, label="model size"):
    """Small 'increasing model size ->' indicator centered under the x-axis, for
    figures whose x is ordered by parameter count. Subtle (FAINT, short). The
    caller must leave bottom margin (subplots_adjust(bottom=...)) for it to show."""
    x0, x1 = 0.5 - width / 2, 0.5 + width / 2
    ann = ax.annotate("", xy=(x1, y), xytext=(x0, y), xycoords="axes fraction",
                      arrowprops=dict(arrowstyle="-|>", color=FAINT, lw=0.9, mutation_scale=9),
                      annotation_clip=False)
    txt = ax.text(0.5, y + 0.045, label, transform=ax.transAxes, ha="center", va="bottom",
                  fontsize=7, color=FAINT)
    # register with save()'s bbox_extra_artists so bbox="tight" keeps the arrow.
    # The annotation's text is empty, so its own extent ignores the arrow -- pass
    # the FancyArrowPatch (ann.arrow_patch) explicitly.
    fig = ax.figure
    fig._bbox_extra = getattr(fig, "_bbox_extra", []) + [ann.arrow_patch, txt]


__all__ = [
    "FIGS", "POST_FIGS_DIR", "publish", "PALETTES", "set_theme", "save",
    "style_ax", "style", "size_axis_arrow", "THEME", "SUFFIX",
    "ACCENT", "BRICK", "GREEN", "GRAY", "LIGHT", "BRICK_LIGHT", "BRICK_MID",
    "MUTED", "FAINT", "EBAR", "MUTED2", "GRID", "BARGRAY", "SEP", "NOTE", "ARROW",
]
