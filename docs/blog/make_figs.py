# [ai-generated]
"""Generate the blog-post figures (narrative FIG-A..H + per-claim CLAIM-*) into docs/blog/figs/.

Data sources:
- FIG-A: exp-23 results/_summary.json (gate = exp-16 headline repro) + exp-18
  RESULTS.md MLP table (hardcoded below with provenance).
- FIG-B: exp-24 results/design1_general.json + exp-23 language null.
- FIG-C: exp-23 results/_summary.json (within-language).
- FIG-D: exp-25 RESULTS.md per-CWE table (hardcoded below with provenance).
- FIG-E: per-token probe scores for one patched example. Needs the exp-16
  token npz + dataset.jsonl, which are NOT in this repo (gitignored payloads).
  Set EXP16_NPZ and DATASET_JSONL env vars; skipped if absent.
- FIG-F: exp-21 family blocks (hardcoded from its RESULTS) vs exp-24
  results/design4_transfer_matrix.json surface blocks (computed).

Run: python docs/blog/make_figs.py   (from the repo root)
"""

import colorsys
from contextlib import contextmanager
import difflib
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import figbase
from figbase import *  # FIGS, PALETTES, save, style_ax, style, size_axis_arrow,
                       # publish + the theme color globals (set_theme repoints them)

ROOT = Path(__file__).resolve().parents[2]
PLANS = ROOT / "plans/cross-model-probe-generalization"

def set_theme(name):
    """Repoint THIS module's color globals (so the bare color names in the figure
    bodies resolve) plus figbase's own, for one theme. See figbase.set_theme."""
    figbase.set_theme(name, globals())

MODEL_ORDER = [
    ("gemma-3-1b-it", "gemma 1B"),
    ("gemma-3-4b-it", "gemma 4B"),
    ("gemma-3-12b-it", "gemma 12B"),
    ("gemma-3-12b-pt", "gemma 12B-pt"),
    ("gemma-3-27b-it", "gemma 27B"),
    ("Qwen2.5-Coder-7B-Instruct", "qwen 7B"),
    ("Qwen2.5-Coder-32B-Instruct", "qwen 32B"),
]

# Same models ranked by parameter count (gemma and qwen interleaved), used where
# the x-axis should read as a size axis rather than group by family. qwen-7B
# slots between gemma-4B and gemma-12B. (12B-pt is dropped — it has no MLP run.)
MODEL_ORDER_BY_SIZE = [
    ("gemma-3-1b-it", "gemma 1B"),
    ("gemma-3-4b-it", "gemma 4B"),
    ("Qwen2.5-Coder-7B-Instruct", "qwen 7B"),
    ("gemma-3-12b-it", "gemma 12B"),
    ("gemma-3-27b-it", "gemma 27B"),
    ("Qwen2.5-Coder-32B-Instruct", "qwen 32B"),
]

# Validation-selected probe layer per model (repo-layer = output of that block)
# and the model's total block count. Selected layer: exp-23 results/<model>.json
# ["layer"]; totals are each architecture's block count. Printed under each model
# in FIG-A so the depth-of-selection is visible alongside the AUC.
LAYER_SEL = {
    "gemma-3-1b-it": (25, 26),
    "gemma-3-4b-it": (7, 34),
    "gemma-3-12b-it": (15, 48),
    "gemma-3-12b-pt": (13, 48),
    "gemma-3-27b-it": (19, 62),
    "Qwen2.5-Coder-7B-Instruct": (16, 28),
    "Qwen2.5-Coder-32B-Instruct": (25, 64),
}

# exp-18 RESULTS.md: best-of(mlp256, mlp512) test tokens_code_auc per model.
MLP_AUC = {
    "gemma-3-1b-it": 0.801,
    "gemma-3-4b-it": 0.806,
    "gemma-3-12b-it": 0.809,
    "gemma-3-27b-it": 0.824,
    "Qwen2.5-Coder-7B-Instruct": 0.828,
    "Qwen2.5-Coder-32B-Instruct": 0.817,
}

summary = json.loads((PLANS / "23-language-stratified-rescore/results/_summary.json").read_text())

# Relative-path result reader used by the CLAIM-* figures.
J = lambda p: json.loads((PLANS / p).read_text())

# set_theme, save, style_ax, style, size_axis_arrow, publish, FIGS, PALETTES and
# the theme color globals all come from figbase (imported with * above).


def fig_a():
    keys = [k for k, _ in MODEL_ORDER]
    labels = [l for _, l in MODEL_ORDER]
    lin = [summary[k]["gate_line_code_auc"] for k in keys]
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.bar(x, lin, 0.62, color=ACCENT, zorder=2, label="linear probe")
    has_mlp = [(i, MLP_AUC[k]) for i, k in enumerate(keys) if k in MLP_AUC]
    ax.scatter([i for i, _ in has_mlp], [v for _, v in has_mlp], marker="D", s=28,
               color=BRICK, zorder=3, label="MLP head")
    for i, v in enumerate(lin):
        ax.text(i, v - 0.025, f"{v:.3f}", ha="center", va="top", fontsize=7, color="white")
    ax.axhline(0.5, color=FAINT, lw=0.8, ls="--")
    ax.text(0.02, 0.507, "chance", fontsize=7, color=FAINT, transform=ax.get_yaxis_transform())
    ax.set_xticks(x, labels, fontsize=7.5)
    # second, muted line under each model name: validation-selected layer / total
    trans = ax.get_xaxis_transform()
    for i, k in enumerate(keys):
        sel, tot = LAYER_SEL[k]
        ax.text(i, -0.115, f"L{sel}/{tot}", transform=trans, ha="center", va="top",
                fontsize=6.6, color=MUTED)
    ax.set_ylim(0.45, 0.92)
    ax.set_ylabel("token-level AUC (live code)", fontsize=8)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left", ncol=2)
    style_ax(ax)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.965, bottom=0.20)
    save(fig, "fig_a_headline")


def fig_b():
    d1 = json.loads((PLANS / "24-surface-baselines/results/design1_general.json").read_text())
    b = d1["baselines"]
    lang_null = 1 - summary["Qwen2.5-Coder-32B-Instruct"]["general_lang_null_cPos_line"]
    rows = [
        ('language only ("is it Python?")', lang_null, None, BRICK),
        ("token-unigram scanner", b["token_unigram_lr"]["tokens_code_auc"], b["token_unigram_lr"]["ci"], GRAY),
        ("activation probe (qwen 32B)", b["probe_general"]["tokens_code_auc"], b["probe_general"]["ci"], ACCENT),
        ("char-n-gram scanner", b["char_ngram_lr"]["tokens_code_auc"], b["char_ngram_lr"]["ci"], GRAY),
    ]
    fig, ax = plt.subplots(figsize=(6.9, 2.8))
    y = np.arange(len(rows))
    for i, (lab, v, ci, c) in enumerate(rows):
        ax.barh(i, v, 0.6, color=c, zorder=2)
        if ci:
            ax.errorbar(v, i, xerr=[[v - ci[0]], [ci[1] - v]], fmt="none", ecolor=EBAR,
                        elinewidth=0.9, capsize=2, zorder=3)
        ax.text(max(v, ci[1] if ci else v) + 0.006, i, f"{v:.3f}", va="center", fontsize=7.5)
    ax.axvline(0.5, color=FAINT, lw=0.8, ls="--")
    ax.text(0.5, -0.62, "chance", fontsize=7, color=FAINT, ha="center")
    ax.set_yticks(y, [r[0] for r in rows], fontsize=8)
    ax.set_xlim(0.45, 0.92)
    ax.set_xlabel("token-level AUC, identical test tokens", fontsize=8)
    ax.invert_yaxis()
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=GRAY, label="reads only the text"),
                       Patch(color=BRICK, label="reads only the language"),
                       Patch(color=ACCENT, label="reads hidden states")],
              fontsize=7, frameon=False, loc="upper right")
    style_ax(ax)
    fig.tight_layout()
    save(fig, "fig_b_ladder")


def fig_c():
    keys = [k for k, _ in MODEL_ORDER]
    labels = [l for _, l in MODEL_ORDER]
    py = [summary[k]["within_py_line"] for k in keys]
    cc = [summary[k]["within_c_line"] for k in keys]
    pooled = [summary[k]["gate_line_code_auc"] for k in keys]
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(7.4, 2.9))
    ax.bar(x - 0.18, py, 0.34, color=ACCENT, zorder=2, label="within Python")
    ax.bar(x + 0.18, cc, 0.34, color=LIGHT, edgecolor=ACCENT, lw=0.6, zorder=2, label="within C/C++")
    ax.scatter(x, pooled, marker="_", s=220, color=BRICK, lw=1.6, zorder=3,
               label="pooled (languages mixed)")
    for i in range(len(keys)):
        ax.text(i - 0.18, py[i] + 0.008, f"{py[i]:.2f}", ha="center", fontsize=6.5, color=ACCENT)
        ax.text(i + 0.18, cc[i] + 0.008, f"{cc[i]:.2f}", ha="center", fontsize=6.5, color=MUTED)
    ax.axhline(0.5, color=FAINT, lw=0.8, ls="--")
    ax.text(0.02, 0.507, "chance", fontsize=7, color=FAINT, transform=ax.get_yaxis_transform())
    ax.set_xticks(x, labels, fontsize=7.5, rotation=12)
    ax.set_ylim(0.45, 0.95)
    ax.set_ylabel("token-level AUC (live code)", fontsize=8)
    ax.legend(fontsize=7.5, frameon=False, ncol=3, loc="upper left")
    style_ax(ax)
    fig.tight_layout()
    save(fig, "fig_c_withinlang")


# exp-25 RESULTS.md tables (allclean point; conly / matchedpatch with 95% CI).
EXP25 = {
    "qwen 32B": {
        "CWE-125": {"allclean": 0.732, "conly": (0.675, 0.613, 0.745), "matched": (0.633, 0.575, 0.701)},
        "CWE-416": {"allclean": 0.766, "conly": (0.714, 0.595, 0.824), "matched": (0.610, 0.514, 0.709)},
        "CWE-476": {"allclean": 0.640, "conly": (0.579, 0.517, 0.676), "matched": (0.544, 0.483, 0.633)},
    },
    "gemma 1B": {
        "CWE-125": {"allclean": 0.734, "conly": (0.668, 0.599, 0.741), "matched": (0.657, 0.593, 0.731)},
        "CWE-416": {"allclean": 0.769, "conly": (0.697, 0.631, 0.756), "matched": (0.603, 0.525, 0.677)},
        "CWE-476": {"allclean": 0.619, "conly": (0.560, 0.506, 0.605), "matched": (0.507, 0.454, 0.552)},
    },
}


CWE_NAMES = {"CWE-125": "out-of-bounds read", "CWE-416": "use-after-free",
             "CWE-476": "NULL dereference"}


def fig_d():
    fig, ax = plt.subplots(figsize=(6.9, 3.2))
    rows = [(m, c) for c in ("CWE-125", "CWE-416", "CWE-476") for m in EXP25]
    for i, (m, c) in enumerate(rows):
        e = EXP25[m][c]
        ax.scatter(e["allclean"], i, marker="o", facecolors="none", edgecolors=GRAY, s=30,
                   zorder=3, label="all clean (confounded)" if i == 0 else None)
        v, lo, hi = e["conly"]
        ax.errorbar(v, i + 0.14, xerr=[[v - lo], [hi - v]], fmt="s", color=GRAY, ms=4,
                    elinewidth=0.9, capsize=2, zorder=3,
                    label="clean C/C++ only" if i == 0 else None)
        v, lo, hi = e["matched"]
        ax.errorbar(v, i - 0.14, xerr=[[v - lo], [hi - v]], fmt="D", color=ACCENT, ms=5,
                    elinewidth=1.2, capsize=2, zorder=4,
                    label="own fixed version (strictest)" if i == 0 else None)
    for sep in (1.5, 3.5):
        ax.axhline(sep, color=SEP, lw=0.7, zorder=1)
    ax.axvline(0.5, color=FAINT, lw=0.9, ls="--")
    ax.text(0.5, -0.75, "chance", fontsize=7, color=FAINT, ha="center")
    ax.set_yticks(range(len(rows)), [f"{CWE_NAMES[c]} · {m}" for m, c in rows], fontsize=8)
    ax.set_xlim(0.42, 0.85)
    ax.set_xlabel("token-level AUC", fontsize=8)
    ax.invert_yaxis()
    ax.legend(fontsize=7, frameon=False, loc="lower right", title="negative pool", title_fontsize=7)
    style_ax(ax)
    fig.tight_layout()
    save(fig, "fig_d_matchedpatch")


def fig_e():
    npz = os.environ.get("EXP16_NPZ", "")
    ds = os.environ.get("DATASET_JSONL", "")
    if not (npz and ds and Path(npz).exists() and Path(ds).exists()):
        print("FIG-E skipped (set EXP16_NPZ / DATASET_JSONL)")
        return
    with _mono_no_hinting():
        eid = int(os.environ.get("FIG_E_EID", "105"))
        rows = [json.loads(l) for l in open(ds)]
        code = rows[eid]["code"]
        d = np.load(npz)
        m = d["example_id"] == eid
        toks = sorted(zip(d["char_start"][m], d["char_end"][m], d["prob"][m]))
        cmap = _heat_cmap()  # red ramp keyed to P(vulnerable), theme-aware
        lines = code.split("\n")
        fig, ax = plt.subplots(figsize=(7.0, 0.34 * len(lines) + 0.5))
        ax.set_xlim(-0.5, max(len(l) for l in lines) + 2)
        ax.set_ylim(-len(lines), 1)
        ax.axis("off")
        _, txts = _draw_code_block(ax, code, toks, 0, 0, cmap, 8.5)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
        cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.01)
        cb.set_label("probe P(vulnerable)", fontsize=7)
        cb.ax.tick_params(labelsize=6)
        fig.tight_layout()
        fs_fit = _fit_mono_fontsize(fig, ax)
        for t in txts:
            t.set_fontsize(fs_fit)
        save(fig, "fig_e_tokenheat", rescale=False)


# FIG-FLIPS: the matched-patch "non-flip". Each pair is a held-out SVEN function
# and its own security fix; the probe scores the SQL/command sink string at ~1.0
# in BOTH, so the vuln->patched label flip leaves the probe's max score put. Same
# Qwen-32B L25 logit dump as FIG-E. eids verified against data/dataset.jsonl and
# the exp-20 audit (all three vuln members are in the held-out subtractive test
# split). Provenance: probes-nlas-cyber-clean/plans/.../20-fn-fp-token-analysis.
FLIP_PAIRS = [
    # (label, CWE, vuln eid, patch eid, fix shown in the code)
    dict(fam="SQL injection", cwe="CWE-089", func="get_article", v=67, s=846),
    dict(fam="SQL injection", cwe="CWE-089", func="delete_playlist", v=1130, s=1192),
    dict(fam="OS command injection", cwe="CWE-078", func="_remove_volume_from_volume_set", v=149, s=221),
]


def _heat_cmap():
    """The FIG-E red ramp keyed to P(vulnerable), theme-aware (see fig_e)."""
    from matplotlib.colors import ListedColormap, hsv_to_rgb
    _t = np.linspace(0, 1, 256)
    if THEME == "dark":
        return ListedColormap(hsv_to_rgb(np.stack(
            [np.full_like(_t, 0.02), 0.25 + 0.70 * _t, 0.16 + 0.46 * _t], axis=1)))
    return ListedColormap(hsv_to_rgb(np.stack(
        [np.full_like(_t, 0.02), 0.04 + 0.88 * _t, np.full_like(_t, 0.98)], axis=1)))


@contextmanager
def _mono_no_hinting():
    """Keep monospace advances fractional so the char grid can fit exactly."""
    old_hinting = plt.rcParams["text.hinting"]
    plt.rcParams["text.hinting"] = "no_hinting"
    try:
        yield
    finally:
        plt.rcParams["text.hinting"] = old_hinting


def _fit_mono_fontsize(fig, ax, fs0=10.0):
    """Font size (pt) at which one monospace character's advance equals exactly 1
    data unit in `ax`, so the per-token heat rectangles (drawn 1 unit per char) line
    up with the rendered code text. Without this the glyph advance and the box grid
    drift apart across a line. Call AFTER the axes box is final (post tight_layout);
    the measurement is a pure pixel ratio, so it is dpi-independent and holds at the
    higher save dpi."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    x0, x1 = ax.get_xlim()
    xspan = abs(x1 - x0)
    axw_px = ax.get_window_extent(r).width
    target_adv_px = axw_px / xspan

    def measure_adv_px(fontsize):
        # Measure advance by differencing two strings. A single text bbox can
        # include fixed side bearings/padding; the delta cancels that constant.
        n_base, n_delta = 20, 40
        y = sum(ax.get_ylim()) / 2.0
        probe_base = ax.text(x0, y, "0" * n_base, family="monospace",
                             fontsize=fontsize, ha="left")
        probe_delta = ax.text(x0, y, "0" * (n_base + n_delta),
                              family="monospace", fontsize=fontsize, ha="left")
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        adv = (
            probe_delta.get_window_extent(renderer).width
            - probe_base.get_window_extent(renderer).width
        ) / n_delta
        probe_base.remove()
        probe_delta.remove()
        return adv

    fontsize = fs0 * target_adv_px / measure_adv_px(fs0)
    for _ in range(4):
        fontsize *= target_adv_px / measure_adv_px(fontsize)
    return fontsize


def _draw_code_block(ax, code, toks, y_top, x0, cmap, fontsize, mark_spans=None,
                     mark_color=None):
    """Paint per-token heat behind monospace code; return (n_lines, text_artists).
    `toks` is an iterable of (char_start, char_end, prob). Top line sits at y_top
    and lines descend by one unit each; columns are character positions + x0. If
    `mark_spans` (character (cs, ce) ranges of the actual fix diff) is given,
    underline EACH span individually in `mark_color` -- split per line and trimmed
    of surrounding whitespace, so the marks track the real edited regions rather
    than the whole labeled line."""
    lines = code.split("\n")
    starts, pos = [], 0
    for ln in lines:
        starts.append(pos)
        pos += len(ln) + 1

    def line_col(cs):
        li = max(i for i, s in enumerate(starts) if s <= cs)
        return li, cs - starts[li]

    for cs, ce, p in toks:
        li, col = line_col(cs)
        w = min(ce, starts[li] + len(lines[li])) - cs
        if w <= 0:
            continue
        ax.add_patch(plt.Rectangle((x0 + col, y_top - li - 0.42), w, 0.84,
                                   color=cmap(float(p)), lw=0, zorder=1))
    for cs, ce in (mark_spans or []):
        cur = cs
        while cur < ce:
            li, col = line_col(cur)
            line_end = starts[li] + len(lines[li])
            b = min(ce, line_end) - starts[li]
            seg = lines[li][col:b]
            a = col + (len(seg) - len(seg.lstrip()))
            b -= len(seg) - len(seg.rstrip())
            if b > a:
                ax.add_patch(plt.Rectangle((x0 + a, y_top - li - 0.62), b - a, 0.13,
                                           color=mark_color, lw=0, zorder=3))
            cur = line_end + 1  # past the newline into the next line
    txts = [ax.text(x0, y_top - i, ln, fontsize=fontsize, family="monospace",
                    ha="left",
                    va="center", zorder=2) for i, ln in enumerate(lines)]
    return len(lines), txts


def fig_flips():
    """Three vuln<->patch pairs: the probe fires on the sink string in both, while
    the underline marks the tokens the fix actually changed (the labeled diff)."""
    npz = os.environ.get("EXP16_NPZ", "")
    ds = os.environ.get("DATASET_JSONL", "")
    if not (npz and ds and Path(npz).exists() and Path(ds).exists()):
        print("FIG-FLIPS skipped (set EXP16_NPZ / DATASET_JSONL)")
        return
    with _mono_no_hinting():
        rows = [json.loads(l) for l in open(ds)]
        d = np.load(npz)
        eids, prob, csa, cea = (d["example_id"], d["prob"], d["char_start"], d["char_end"])
        cmap = _heat_cmap()
        FS = 8.0
        LEFT = 13          # left margin (char units) for the vulnerable/patched tag
        HEADER, MIDGAP, PAIRGAP = 1.25, 0.95, 1.7

        def toks_for(eid):
            m = eids == eid
            return sorted(zip(csa[m], cea[m], prob[m]))

        def diff_spans(veid, seid):
            """Character ranges in the VULNERABLE code that the fix removed or changed
            (difflib opcodes vs the patched version) -- the real, individual edits, not
            the whole labeled line."""
            sm = difflib.SequenceMatcher(a=rows[veid]["code"], b=rows[seid]["code"],
                                         autojunk=False)
            return [(i1, i2) for tag, i1, i2, _, _ in sm.get_opcodes()
                    if tag in ("replace", "delete") and i2 > i1]

        maxchars = max(max(len(l) for l in rows[e].get("code", "").split("\n"))
                       for pr in FLIP_PAIRS for e in (pr["v"], pr["s"]))
        RIGHT = maxchars + LEFT + 2     # right margin for the function-name label

        # total height in line-units (one dry pass)
        total = 0.4
        for pr in FLIP_PAIRS:
            nv = len(rows[pr["v"]]["code"].split("\n"))
            ns = len(rows[pr["s"]]["code"].split("\n"))
            total += HEADER + nv + MIDGAP + ns + PAIRGAP

        fig, ax = plt.subplots(figsize=(7.3, 0.205 * total + 0.55))
        ax.set_xlim(-0.5, RIGHT + 2)
        ax.axis("off")

        txts = []
        y = 0.0
        for pr in FLIP_PAIRS:
            nv = len(rows[pr["v"]]["code"].split("\n"))
            ns = len(rows[pr["s"]]["code"].split("\n"))
            # pair header
            ax.text(0, y, f"{pr['fam']}  ·  {pr['cwe']}", fontsize=8.8, weight="bold",
                    va="center", color=ACCENT, zorder=3)
            ax.text(RIGHT + 2, y, f"{pr['func']}()", fontsize=7.2, style="italic",
                    va="center", ha="right", color=FAINT, zorder=3)
            y -= HEADER
            # vulnerable block (underline the actual fix diff)
            ax.text(LEFT - 2, y - (nv - 1) / 2, "vulnerable", fontsize=7.6, weight="bold",
                    va="center", ha="right", color=BRICK, zorder=3, rotation=0)
            _, t = _draw_code_block(ax, rows[pr["v"]]["code"], toks_for(pr["v"]), y, LEFT,
                                    cmap, FS, mark_spans=diff_spans(pr["v"], pr["s"]),
                                    mark_color=ACCENT)
            txts += t
            y -= (nv - 1) + MIDGAP
            # patched block (the fix's own side -> nothing removed, no underline)
            ax.text(LEFT - 2, y - (ns - 1) / 2, "patched", fontsize=7.6, weight="bold",
                    va="center", ha="right", color=GREEN, zorder=3)
            _, t = _draw_code_block(ax, rows[pr["s"]]["code"], toks_for(pr["s"]), y, LEFT,
                                    cmap, FS)
            txts += t
            y = y - (ns - 1) - PAIRGAP

        ax.set_ylim(y + 0.6, 1.0)
        # legend: what the underline means
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([0], [0], color=ACCENT, lw=3,
                                  label="code the fix removed or changed")],
                  loc="lower right", fontsize=7, frameon=False, handlelength=1.3,
                  borderaxespad=0.2)
        # colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
        cb = fig.colorbar(sm, ax=ax, fraction=0.022, pad=0.01)
        cb.set_label("probe P(vulnerable)", fontsize=7)
        cb.ax.tick_params(labelsize=6)
        fig.tight_layout()
        fs_fit = _fit_mono_fontsize(fig, ax)
        for t in txts:
            t.set_fontsize(fs_fit)
        save(fig, "fig_flips", rescale=False)


# exp-21 RESULTS: family-block means of the cross-CWE transfer matrix
# (train CWE-X probe, eval CWE-Y vuln-vs-all-clean), off-diagonal cells.
PROBE_BLOCKS = {"inj→inj": 0.60, "mem→mem": 0.57, "inj→mem": 0.41, "mem→inj": 0.34}
INJ = ["CWE-089", "CWE-078", "CWE-022", "CWE-079"]
MEM = ["CWE-125", "CWE-416", "CWE-476"]


def fig_f():
    d4 = json.loads((PLANS / "24-surface-baselines/results/design4_transfer_matrix.json").read_text())
    auc = d4["auc"]

    def block(src, dst):
        cells = [auc[a][b] for a in src for b in dst if a != b]
        return float(np.mean(cells))

    surf = {"inj→inj": block(INJ, INJ), "mem→mem": block(MEM, MEM),
            "inj→mem": block(INJ, MEM), "mem→inj": block(MEM, INJ)}
    labels = list(PROBE_BLOCKS)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    ax.bar(x - 0.19, [PROBE_BLOCKS[k] for k in labels], 0.36, color=ACCENT, zorder=2,
           label="probe (hidden states)")
    ax.bar(x + 0.19, [surf[k] for k in labels], 0.36, color=GRAY, zorder=2,
           label="char-n-gram (text only)")
    for i, k in enumerate(labels):
        ax.text(i - 0.19, PROBE_BLOCKS[k] + 0.008, f"{PROBE_BLOCKS[k]:.2f}", ha="center", fontsize=7, color=ACCENT)
        ax.text(i + 0.19, surf[k] + 0.008, f"{surf[k]:.2f}", ha="center", fontsize=7, color=MUTED)
    ax.axhline(0.5, color=FAINT, lw=0.8, ls="--")
    ax.annotate("the one block the text\nbaseline doesn't reproduce",
                (1.19, surf["mem→mem"] + 0.015), (1.48, 0.64), fontsize=7, color=NOTE,
                ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=ARROW, lw=0.8,
                                connectionstyle="arc3,rad=0.15"))
    ax.set_xticks(x, [f"{k}\n(train→eval family)" for k in labels], fontsize=8)
    ax.set_ylim(0.25, 0.72)
    ax.set_ylabel("token-level AUC (cross-CWE transfer)", fontsize=8)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    style_ax(ax)
    fig.tight_layout()
    save(fig, "fig_f_blocks")


def fig_g():
    """Linear vs MLP head, each at its OWN validation-selected layer (exp-18
    proper comparison; the layer-policy artifact version lives in claim_scaling).
    Models are ordered by parameter count, and the bar hue marks the family
    (gemma blue, qwen brick) to match the color convention in claim_capacity:
    within each model the solid bar is the linear probe and the light-tint bar
    (same hue, family-colored edge) is the MLP head."""
    keys = [k for k, _ in MODEL_ORDER_BY_SIZE if k in MLP_AUC]
    labels = [l for k, l in MODEL_ORDER_BY_SIZE if k in MLP_AUC]
    lin = [summary[k]["gate_line_code_auc"] for k in keys]
    mlp = [MLP_AUC[k] for k in keys]
    # per-model family color: qwen brick, gemma accent-blue (solid = linear,
    # tint = MLP head); same family convention as CAP_MODELS/_cap_color below.
    is_qwen = ["Qwen" in k for k in keys]
    base = [BRICK if q else ACCENT for q in is_qwen]
    tint = [BRICK_LIGHT if q else LIGHT for q in is_qwen]
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    ax.bar(x - 0.18, lin, 0.34, color=base, zorder=2)
    # MLP bar: light tint + same-hue hatch so "MLP head" reads by fill texture too
    ax.bar(x + 0.18, mlp, 0.34, color=tint, edgecolor=base, lw=0.6, hatch="////", zorder=2)
    for i in range(len(keys)):
        ax.text(i - 0.18, lin[i] + 0.006, f"{lin[i]:.2f}", ha="center", fontsize=6.5, color=base[i])
        ax.text(i + 0.18, mlp[i] + 0.006, f"{mlp[i]:.2f}", ha="center", fontsize=6.5, color=MUTED)
    ax.axhline(0.5, color=FAINT, lw=0.8, ls="--")
    ax.text(0.02, 0.507, "chance", fontsize=7, color=FAINT, transform=ax.get_yaxis_transform())
    ax.set_xticks(x, labels, fontsize=7.5, rotation=12)
    ax.set_ylim(0.45, 0.92)
    ax.set_ylabel("token-level AUC (live code)", fontsize=8)
    # neutral-grey fill legend (solid = linear, tint = MLP); hue is read off the
    # family-colored bars + x labels, so the legend stays color-agnostic.
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=MUTED, label="linear probe"),
                       Patch(facecolor=BARGRAY, edgecolor=MUTED, lw=0.6, hatch="////", label="MLP head")],
              fontsize=7.5, frameon=False, ncol=2, loc="upper left")
    # x-axis is ordered by parameter count: small "model size ->" cue under the ticks
    size_axis_arrow(ax)
    style_ax(ax)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.965, bottom=0.30)
    save(fig, "fig_g_mlp")


# FIG-H: the headline AUC split into vulnerability families. Per model, the
# general probe's token-level AUC on injection-class CWEs vs memory-safety CWEs,
# read from exp-23 per-model `per_cwe` (same source/probe as FIG-A's pooled bar).
# Each CWE's `b_probe_auc` ranks that CWE's positive code tokens against the full
# test pool, so a positive-token-weighted mean over a family ≈ the pooled family
# AUC (family positives vs the rest). Untrusted CWEs (exp-23's flag: <10 positive
# examples; here CWE-190/787 at 4 and 5) are dropped — matching the INJ/MEM
# grouping used in FIG-F.
INJ_CWES = ["CWE-089", "CWE-078", "CWE-022", "CWE-079"]
MEM_CWES = ["CWE-125", "CWE-416", "CWE-476"]


def _family_auc(per_cwe, fam_cwes):
    """Positive-token-weighted mean of per-CWE token-AUC over a family's trusted
    CWEs (≈ pooled family-vs-rest AUC)."""
    rows = [(per_cwe[c]["b_probe_auc"], per_cwe[c]["n_pos_tokens"])
            for c in fam_cwes if c in per_cwe and not per_cwe[c].get("untrusted")]
    tot = sum(n for _, n in rows)
    return sum(a * n for a, n in rows) / tot


def _desat(hexc, scale):
    """Scale a hex color's HLS saturation by `scale` (hue + lightness fixed) — a
    softer version of the same hue. Theme-agnostic: applied to the active GREEN/
    BRICK, it softens the light or dark palette consistently."""
    h = hexc.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    hh, ll, ss = colorsys.rgb_to_hls(r, g, b)
    rgb = colorsys.hls_to_rgb(hh, ll, ss * scale)
    return "#%02x%02x%02x" % tuple(int(round(c * 255)) for c in rgb)


def _darken(hexc, f=0.78):
    """Multiply a hex color toward black by `f` (for value labels: a touch darker
    than the bar so the number reads on white)."""
    h = hexc.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return "#%02x%02x%02x" % (int(r * f), int(g * f), int(b * f))


def fig_h():
    keys = [k for k, _ in MODEL_ORDER]
    labels = [l for _, l in MODEL_ORDER]
    res = {k: json.loads((PLANS / f"23-language-stratified-rescore/results/{k}.json").read_text())
           for k in keys}

    def fam_val(k, fam_name, fam_cwes):
        # Prefer the TRUE pooled family-vs-rest AUC (rescore_language.py's
        # `family_pooled`); fall back to the positive-token-weighted mean of the
        # per-CWE AUCs only if a model's JSON predates that recompute (warns —
        # the mean under-reports injection by ~0.008; re-run rescore to refresh).
        fp = res[k].get("family_pooled")
        if fp and fam_name in fp:
            return fp[fam_name]["pooled_auc"]
        print(f"  [fig_h] WARN {k}: no family_pooled[{fam_name}] -> weighted-mean "
              f"fallback (re-run rescore_language.py for the true pooled AUC)")
        return _family_auc(res[k]["per_cwe"], fam_cwes)

    inj = [fam_val(k, "injection", INJ_CWES) for k in keys]
    mem = [fam_val(k, "memory", MEM_CWES) for k in keys]
    pooled = [summary[k]["gate_line_code_auc"] for k in keys]
    # Family hues = softened (x0.85 saturation) green / vermillion so injection and
    # memory read as distinct colors; the pooled marker is ACCENT (FIG-A's bar
    # color) so the same number matches across the two figures. Figure-local —
    # GREEN/BRICK stay untouched globally, so no other figure changes.
    inj_c, mem_c = _desat(GREEN, 0.85), _desat(BRICK, 0.85)
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    ax.bar(x - 0.18, inj, 0.34, color=inj_c, zorder=2, label="injection CWEs")
    ax.bar(x + 0.18, mem, 0.34, color=mem_c, zorder=2, label="memory-safety CWEs")
    ax.scatter(x, pooled, marker="_", s=220, color=ACCENT, lw=1.8, zorder=4,
               label="pooled (all CWEs)")
    for i in range(len(keys)):
        ax.text(i - 0.18, inj[i] + 0.008, f"{inj[i]:.2f}", ha="center", fontsize=6.5, color=_darken(inj_c))
        ax.text(i + 0.18, mem[i] + 0.008, f"{mem[i]:.2f}", ha="center", fontsize=6.5, color=_darken(mem_c))
    ax.axhline(0.5, color=FAINT, lw=0.8, ls="--")
    ax.text(0.02, 0.507, "chance", fontsize=7, color=FAINT, transform=ax.get_yaxis_transform())
    ax.set_xticks(x, labels, fontsize=7.5, rotation=12)
    ax.set_ylim(0.45, 0.97)
    ax.set_ylabel("token-level AUC (live code)", fontsize=8)
    ax.legend(fontsize=7.5, frameon=False, ncol=3, loc="upper center")
    style_ax(ax)
    fig.tight_layout()
    save(fig, "fig_h_family")


def fig_stringmatcher():
    """exp-20 (DAT5) — cross-probe detection is bimodal and family-split: per
    held-out vuln function, how many of the 7 probes catch it (>=1 true vuln
    token over threshold), stacked by CWE family. fn_corpus.json gives the
    per-bucket injection/memory split; totals reproduce analysis.json detect_hist.
    Family hues match fig_h (injection green / memory vermillion, x0.85 sat).

    NB the FP-lexical-breakdown panel was DROPPED (2026-06-16, user): its bucket
    taxonomy was a catch-all regex (51% bucket = the regex else-branch, named
    "SQL identifier" but holding generic vars/literals) over a 127/130-Python
    curated sample, and the population file (fp_corpus.json) is gitignored/absent
    so the 51/25/15 split can't be re-verified. The robust FP evidence is the
    safe-alarm result (96% of cross-model false alarms on patched injection
    code), not the lexical breakdown. See project-log §6."""
    E = PLANS / "20-fn-fp-token-analysis"
    corpus = json.loads((E / "fn_corpus.json").read_text())

    inj_set = set(INJ_CWES)  # CWE-089/078/022/079 (string-sink); else = memory
    buckets = list(range(8))
    inj = [sum(1 for r in corpus if r["n_detect"] == b and r["cwe"] in inj_set) for b in buckets]
    mem = [sum(1 for r in corpus if r["n_detect"] == b and r["cwe"] not in inj_set) for b in buckets]
    inj_c, mem_c = _desat(GREEN, 0.85), _desat(BRICK, 0.85)

    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    x = np.arange(8)
    ax.bar(x, inj, 0.74, color=inj_c, zorder=2, label="injection CWEs")
    ax.bar(x, mem, 0.74, bottom=inj, color=mem_c, zorder=2, label="memory-safety CWEs")
    tot = [inj[i] + mem[i] for i in range(8)]
    for i, t in enumerate(tot):
        if t:
            ax.text(i, t + 1.0, str(t), ha="center", va="bottom", fontsize=6.6, color=MUTED)
    ax.set_xticks(x, [str(b) for b in buckets], fontsize=7.5)
    ax.set_xlabel("#probes/7", fontsize=8)
    ax.set_ylabel("#vulnerable functions caught", fontsize=8)
    ax.set_ylim(0, 62)
    ax.set_title("Distribution of model probe performance", fontsize=8.5, fontweight="bold")
    ax.legend(fontsize=7, frameon=False, loc=(0.025, 0.80))
    style_ax(ax)
    fig.tight_layout()
    save(fig, "fig_stringmatcher")


# ===========================================================================
# CLAIM-* figures: one plot per claim in the blog outline (formerly the
# standalone make_claim_figs.py). House style is the per-claim diagnostic one:
# a bold claim-title suptitle and style() (grids both axes). Values read from
# result JSONs except where a comment cites the hardcoded source table.
# ===========================================================================


def claim_scaling():
    """RES4/RES6 — MLP 'scaling' is a layer-policy artifact."""
    sizes = ["1b", "4b", "12b", "27b"]
    lin = [J(f"09-ensemble-linear-probes/results/summary_google_gemma-3-{s}-it.json")
           ["baseline_overall_tokens_code_auc"] for s in sizes]
    # exp-09 MLP at the LINEAR-selected layer (HANDOFF.md item 1 / fig6, 2026-06-01)
    mlp_linlayer = [0.759, 0.7785, 0.805, 0.814]
    # exp-18 RESULTS.md: MLP at its OWN val-selected layer (best of mlp256/512)
    mlp_ownlayer = [0.801, 0.806, 0.809, 0.824]
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.plot(x, lin, "o-", color=ACCENT, label="linear probe")
    ax.plot(x, mlp_linlayer, "s-", color=BRICK, label="MLP at the linear probe's layer")
    ax.plot(x, mlp_ownlayer, "D-", color=GREEN, label="MLP at its own swept layer")
    for xi, (a, b) in enumerate(zip(mlp_linlayer, mlp_ownlayer)):
        ax.annotate("", (xi, b - 0.002), (xi, a + 0.002),
                    arrowprops=dict(arrowstyle="->", color=MUTED2, lw=0.8))
    ax.set_xticks(x, [f"gemma-3 {s}" for s in sizes], fontsize=9)
    ax.set_ylabel("token-level AUC", fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle("MLP probe AUC: fixed operating layer vs each head's best layer",
                 fontsize=11, fontweight="bold")
    style(ax)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "claim_scaling")


# Does extra probe capacity beat a single linear probe? Two panels (overall |
# memory-safety CWEs 125/416/476), per model:
#   - K-ensemble of linear directions (solid line): exp-09 sweep AT THE OPERATING
#     LAYER, aggregation chosen by validation AUC at each K. K=1 IS the single
#     linear probe (== the baseline cell), so the left end of each line is the
#     plain linear probe. Qwen2.5-Coder-7B's ensemble comes from the 2026-06-14
#     exp-09 addon on cv_* operating-layer acts.
#   - MLP head (dashed horizontal line, same colour): a 2-layer head AT ITS OWN
#     swept best layer (NOT the operating layer) — the SAME layer policy fig_g_mlp
#     / MLP_AUC use, so the overall dashed line equals fig-3's MLP bar for that
#     model. Overall = exp-18 best-of(mlp256,mlp512) test tokens_code_auc; memory
#     = that same best head's per-CWE token AUC recomputed at its best layer
#     (exp-18 per_cwe_bestlayer.py -> mlp_memory_bestlayer.json; identical by_cwe
#     contrast to the ensemble). Both panels' dashed line is therefore ONE MLP at
#     ONE layer.
# Gemma sizes get a blue ramp; the two Qwen-Coder models the brick pair (7B =
# brick_mid, 32B = brick). A model is skipped until its ensemble summary lands.
CAP_MODELS = [
    ("google_gemma-3-1b-it", "gemma 1B"),
    ("google_gemma-3-4b-it", "gemma 4B"),
    ("google_gemma-3-12b-it", "gemma 12B"),
    ("google_gemma-3-27b-it", "gemma 27B"),
    ("Qwen_Qwen2.5-Coder-7B-Instruct", "qwen 7B"),
    ("Qwen_Qwen2.5-Coder-32B-Instruct", "qwen 32B"),
]
# ColorBrewer "Blues" (CVD-safe sequential); dark variant lifted so the darkest
# size step still reads on #0b0b0b.
CAP_RAMP = {"light": ["#9ecae1", "#4292c6", "#2171b5", "#08519c"],
            "dark": ["#c6dbef", "#82b3e6", "#4a90d9", "#2b5d9c"]}
RES09 = PLANS / "09-ensemble-linear-probes/results"


def _cap_color(i):
    # 4 gemmas: blue ramp; qwen-7B: mid brick; qwen-32B: brick
    if i < 4:
        return CAP_RAMP[THEME][i]
    return BRICK_MID if i == 4 else BRICK


def _ens_curve(slug):
    """exp-09: per K, the aggregation with the best validation AUC; return
    (Ks, overall test AUC, memory-CWE-mean test AUC), or None if no summary yet.
    K=1 == single linear probe."""
    p = RES09 / f"summary_{slug}.json"
    if not p.exists():
        return None
    rows = json.loads(p.read_text())["rows"]
    ks = sorted({r["K"] for r in rows})
    best = {k: max((r for r in rows if r["K"] == k), key=lambda r: r["val_tokens_code_auc"])
            for k in ks}
    ov = [best[k]["overall"] for k in ks]
    mem = [float(np.mean([best[k]["by_cwe"][c] for c in MEM_CWES])) for k in ks]
    return ks, ov, mem


_MLP_MEM_BESTLAYER = None


def _mlp_mem_table():
    """exp-18 mlp_memory_bestlayer.json (per_cwe_bestlayer.py): per model, the
    best MLP head's per-CWE token AUC AT ITS OWN BEST LAYER. Cached; {} if absent."""
    global _MLP_MEM_BESTLAYER
    if _MLP_MEM_BESTLAYER is None:
        p = PLANS / "18-mlp-logit-dump/results/mlp_memory_bestlayer.json"
        _MLP_MEM_BESTLAYER = json.loads(p.read_text()) if p.exists() else {}
    return _MLP_MEM_BESTLAYER


def _mlp_overall(slug):
    """MLP overall token-code AUC at the MLP's OWN swept best layer: max over
    heads of exp-18 test_tokens_code_auc — identical to fig_g_mlp's MLP_AUC, so
    fig-3 and fig-4 agree on the MLP overall number."""
    try:
        return max(J(f"18-mlp-logit-dump/results/{slug}/{h}/metrics_mlp.json")["test_tokens_code_auc"]
                   for h in ("mlp256", "mlp512"))
    except Exception:
        return None


def _mlp_mem(slug):
    """Memory-CWE-mean token AUC of the best MLP head AT ITS BEST LAYER (same
    layer the overall number uses; same by_cwe contrast as the ensemble)."""
    return _mlp_mem_table().get(slug, {}).get("mem_mean")


def claim_capacity():
    """RES7 — does extra probe capacity beat a single linear probe (K=1)? Plotted
    as Δ AUC vs the MLP head (MLP = 0 line): the K-ensemble of linear directions
    sits BELOW the MLP for every model on both panels, and adding members (K)
    doesn't close the gap. K=1 is the plain single linear probe. The only upward
    motion is qwen 32B creeping toward its MLP with K on memory CWEs."""
    fig, (ax_ov, ax_mem) = plt.subplots(1, 2, figsize=(9.8, 3.8), width_ratios=[1.05, 1])
    for i, (slug, lab) in enumerate(CAP_MODELS):
        ec = _ens_curve(slug)
        if ec is None:
            continue
        ks, ov, mem = ec
        c = _cap_color(i)
        mo, mm = _mlp_overall(slug), _mlp_mem(slug)
        if mo is not None:
            ax_ov.plot(ks, [o - mo for o in ov], "o-", color=c, ms=4.5, lw=1.6, zorder=3, label=lab)
        if mm is not None:
            ax_mem.plot(ks, [m - mm for m in mem], "o-", color=c, ms=4.5, lw=1.6, zorder=3, label=lab)

    for ax, (lo, hi), title in [(ax_ov, (-0.075, 0.022), "overall"),
                                (ax_mem, (-0.20, 0.035), "memory CWEs (125 / 416 / 476 mean)")]:
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8], ["K=1\n(linear\nprobe)", "2", "4", "8"], fontsize=8)
        ax.set_xlabel("K  (ensemble size)", fontsize=8)
        ax.set_ylim(lo, hi)
        ax.axhline(0.0, color=MUTED, lw=1.2, zorder=1)
        ax.text(0.985, 0.0, "  MLP head", fontsize=6.8, color=MUTED,
                transform=ax.get_yaxis_transform(), ha="right", va="bottom")
        ax.set_ylabel("token-level AUC  −  MLP head", fontsize=8)
        ax.set_title(title, fontsize=9)
        style(ax)

    ax_ov.legend(fontsize=7, frameon=False, loc="lower left", ncol=2,
                 handlelength=1.4, columnspacing=1.0)
    fig.suptitle("Linear K-ensemble AUC minus MLP-head AUC, by ensemble size",
                 fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "claim_capacity")


def claim_verbalized():
    """RES8 — verbalized weak + scales with size; probe flat above it."""
    # ordered by parameter count (size axis), matching fig 3 / fig_g_mlp
    order = [("gemma-3-1b-it", "google_gemma-3-1b-it", "gemma 1B"),
             ("gemma-3-4b-it", "google_gemma-3-4b-it", "gemma 4B"),
             ("Qwen2.5-Coder-7B-Instruct", "Qwen_Qwen2.5-Coder-7B-Instruct", "qwen 7B"),
             ("gemma-3-12b-it", "google_gemma-3-12b-it", "gemma 12B"),
             ("gemma-3-27b-it", "google_gemma-3-27b-it", "gemma 27B"),
             ("Qwen2.5-Coder-32B-Instruct", "Qwen_Qwen2.5-Coder-32B-Instruct", "qwen 32B")]
    summ = J("23-language-stratified-rescore/results/_summary.json")
    probe = [summ[k]["gate_line_code_auc"] for k, _, _ in order]
    verb = [J(f"17-verbalized-logit-dump/results/{slug}/metrics_verbalized_logits.json")
            ["verbalized_auc_test"] for _, slug, _ in order]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.bar(x - 0.18, probe, 0.34, color=ACCENT, zorder=2, label="probe (token AUC)")
    # verbalized bar: light fill + same-hue hatch (the "secondary readout" style)
    ax.bar(x + 0.18, verb, 0.34, color=BRICK_LIGHT, edgecolor=BRICK, lw=0.6, hatch="////",
           zorder=2, label="verbalized yes/no (example AUC)")
    for i, (p, v) in enumerate(zip(probe, verb)):
        ax.text(i - 0.18, p + 0.005, f"{p:.2f}", ha="center", fontsize=7, color=ACCENT)
        ax.text(i + 0.18, v + 0.005, f"{v:.2f}", ha="center", fontsize=7, color=BRICK)
    ax.axhline(0.5, color=FAINT, lw=0.8, ls="--")
    ax.set_xticks(x, [l for _, _, l in order], fontsize=8)
    ax.set_ylim(0.4, 0.95)
    ax.legend(fontsize=8, frameon=False, loc="upper left", ncol=2)
    size_axis_arrow(ax)
    fig.suptitle("Probe vs the model's own verbalized answer", fontsize=11, fontweight="bold")
    style(ax)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.26)
    save(fig, "claim_verbalized")


def claim_langmethod():
    """LNG1/LNG2/LNG3 — the language-baseline methodology, in one figure."""
    d = J("23-language-stratified-rescore/results/Qwen2.5-Coder-32B-Instruct.json")
    py, cc = d["within_language"]["py_line"], d["within_language"]["c_line"]
    summ = J("23-language-stratified-rescore/results/_summary.json")["Qwen2.5-Coder-32B-Instruct"]
    lang_null = 1 - summ["general_lang_null_cPos_line"]
    probe = summ["gate_line_code_auc"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))

    # (a) composition of the eval
    ax = axes[0]
    langs = ["Python", "C/C++"]
    pos = [py["n_pos_tokens"], cc["n_pos_tokens"]]
    neg = [py["n_tokens"] - py["n_pos_tokens"], cc["n_tokens"] - cc["n_pos_tokens"]]
    x = np.arange(2)
    ax.bar(x, neg, 0.5, color=LIGHT, zorder=2, label="negative (clean) tokens")
    ax.bar(x, pos, 0.5, bottom=neg, color=BRICK, zorder=2, label="positive (vuln-span) tokens")
    for i in range(2):
        ax.text(i, neg[i] + pos[i] + 800, f"{100*pos[i]/(pos[i]+neg[i]):.1f}% pos", ha="center", fontsize=8)
    ax.set_xticks(x, langs, fontsize=9)
    ax.set_ylabel("test live-code tokens", fontsize=8)
    ax.set_title("(a) positive-token density by language", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    style(ax)

    # (b) null on the identical eval
    ax = axes[1]
    vals = [0.5, lang_null, probe]
    labs = ["chance", 'score := "is Python"\n(language null)', "probe"]
    cols = [BARGRAY, BRICK, ACCENT]
    ax.bar(range(3), vals, 0.55, color=cols, zorder=2)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.008, f"{v:.3f}", ha="center", fontsize=9)
    frac = (lang_null - 0.5) / (probe - 0.5)
    ax.annotate(f"language alone recovers\n{frac:.0%} of the margin",
                (1, lang_null), (0.4, 0.86), fontsize=8,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
    ax.set_xticks(range(3), labs, fontsize=8)
    ax.set_ylim(0.45, 0.95)
    ax.set_title("(b) language-only score, same eval", fontsize=9)
    style(ax)

    # (c) stratified: cross-language pairs removed
    ax = axes[2]
    vals = [py["pooled_auc"], cc["pooled_auc"]]
    los = [py["boot95_lo"], cc["boot95_lo"]]
    his = [py["boot95_hi"], cc["boot95_hi"]]
    ax.bar(range(2), vals, 0.5, color=[ACCENT, LIGHT], edgecolor=ACCENT, lw=0.6, zorder=2)
    ax.errorbar(range(2), vals, yerr=[np.subtract(vals, los), np.subtract(his, vals)],
                fmt="none", ecolor=EBAR, elinewidth=1, capsize=3, zorder=3)
    ax.axhline(probe, color=MUTED, lw=0.8, ls=":")
    ax.text(1.32, probe, "pooled 0.776", fontsize=7, va="center", color=MUTED)
    ax.axhline(0.5, color=FAINT, lw=0.8, ls="--")
    for i, v in enumerate(vals):
        ax.text(i, his[i] + 0.012, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xticks(range(2), ["within Python", "within C/C++"], fontsize=9)
    ax.set_ylim(0.45, 0.95)
    ax.set_title("(c) within-language AUC", fontsize=9)
    style(ax)

    fig.suptitle("The language baseline (Qwen2.5-Coder 32B)", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "claim_langmethod")


def claim_family26():
    """SUR3 — no transferable memory-family direction within C/C++."""
    d = J("26-primevul-within-family/results/pv_within.json")
    b, ci = d["family_blocks"], d["family_blocks_ci"]
    keys = ["mem->mem_offdiag", "mem->other", "other->mem", "other->other_offdiag"]
    labs = ["mem→mem\n(off-diag)", "mem→other", "other→mem", "other→other\n(off-diag)"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3), width_ratios=[1, 1.3])
    ax = axes[0]
    vals = [b[k] for k in keys]
    ax.bar(range(4), vals, 0.55, color=[BRICK, GRAY, GRAY, GRAY], zorder=2)
    ax.errorbar(range(4), vals, yerr=[[b[k] - ci[k][0] for k in keys], [ci[k][1] - b[k] for k in keys]],
                fmt="none", ecolor=EBAR, elinewidth=1, capsize=3, zorder=3)
    ax.axhline(0.5, color=FAINT, lw=0.8, ls="--")
    ax.set_xticks(range(4), labs, fontsize=8)
    ax.set_ylim(0.42, 0.62)
    ax.set_ylabel("token-level AUC (transfer)", fontsize=8)
    ax.set_title("family-transfer blocks", fontsize=9)
    style(ax)

    ax = axes[1]
    diag = {c: v for c, v in d["diagonal"].items() if v["family"] == "mem"}
    cs = sorted(diag, key=lambda c: -diag[c]["auc"])
    vals = [diag[c]["auc"] for c in cs]
    los = [diag[c]["ci"][0] for c in cs]
    his = [diag[c]["ci"][1] for c in cs]
    ax.bar(range(len(cs)), vals, 0.55, color=ACCENT, zorder=2)
    ax.errorbar(range(len(cs)), vals, yerr=[np.subtract(vals, los), np.subtract(his, vals)],
                fmt="none", ecolor=EBAR, elinewidth=0.9, capsize=2, zorder=3)
    ax.axhline(0.5, color=FAINT, lw=0.8, ls="--")
    ax.set_xticks(range(len(cs)), [c.replace("CWE-", "") for c in cs], fontsize=8)
    ax.set_ylim(0.3, 1.0)
    ax.set_title("per-CWE diagonals", fontsize=9)
    style(ax)
    fig.suptitle("PrimeVul (C/C++): family-transfer blocks and per-CWE diagonals",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "claim_family26")


def claim_additive():
    """RES9 — additive fixes undetectable; subtractive subset costless."""
    # exp-19 RESULTS.md mean row: base→sub 0.756, sub→sub 0.755, base→base 0.769,
    # pairAcc-sub 0.760, pairAcc-add 0.429.
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.0))
    ax = axes[0]
    vals = [0.760, 0.429]
    ax.bar([0, 1], vals, 0.5, color=[ACCENT, BRICK], zorder=2)
    ax.axhline(0.5, color=FAINT, lw=0.8, ls="--")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks([0, 1], ["subtractive pairs\n(vuln tokens removed by fix)", "additive pairs\n(fix only ADDS a check)"], fontsize=8)
    ax.set_ylabel("pair-ranking accuracy", fontsize=8)
    ax.set_title("pair ranking by max token score", fontsize=9)
    style(ax)
    ax = axes[1]
    vals = [0.769, 0.756, 0.755]
    ax.bar(range(3), vals, 0.5, color=[GRAY, GRAY, ACCENT], zorder=2)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xticks(range(3), ["base→base", "base→sub", "sub→sub"], fontsize=8)
    ax.set_ylim(0.70, 0.80)
    ax.set_title("token AUC by train/eval subset", fontsize=9)
    style(ax)
    fig.suptitle("Additive vs subtractive pairs: pair-ranking and token AUC",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "claim_additive")


def claim_steering():
    """SUR4 — steering the probe direction at fair magnitude moves nothing."""
    s = J("13-causal-steering/results/steer_v2/steer_13_Qwen_Qwen2.5-Coder-32B-Instruct.json")
    alphas = s["alpha_grid"]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0), sharey=True)
    subsets = [("memory_pos", "P(yes) on memory-vuln code"),
               ("injection_pos", "P(yes) on injection-vuln code"),
               ("negative", "P(yes) on clean code")]
    for ax, (sub, title) in zip(axes, subsets):
        for dname, col in [("memory", BRICK), ("injection", ACCENT)]:
            ax.plot(alphas, s["by_direction"][dname]["by_subset"][sub]["p_yes"], "o-", ms=3, color=col,
                    label=f"{dname} probe direction")
        rnd = np.array([s["by_direction"][f"random_{i}"]["by_subset"][sub]["p_yes"] for i in (0, 1)])
        ax.fill_between(alphas, rnd.min(0), rnd.max(0), color=BARGRAY, alpha=0.4, label="2 random directions")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("steering strength α (·σ along direction)", fontsize=8)
        style(ax)
    axes[0].set_ylabel("model's verbalized P(yes)", fontsize=8)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Verbalized P(yes) shift under probe-direction steering",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save(fig, "claim_steering")


# exp-29 key format: "logitdump_<org>_<model>" (the token-logit dump dir name);
# MODEL_ORDER_BY_SIZE keys are the bare model names, so map them explicitly.
E29KEY = {
    "gemma-3-1b-it": "logitdump_google_gemma-3-1b-it",
    "gemma-3-4b-it": "logitdump_google_gemma-3-4b-it",
    "gemma-3-12b-it": "logitdump_google_gemma-3-12b-it",
    "gemma-3-27b-it": "logitdump_google_gemma-3-27b-it",
    "Qwen2.5-Coder-7B-Instruct": "logitdump_Qwen_Qwen2.5-Coder-7B-Instruct",
    "Qwen2.5-Coder-32B-Instruct": "logitdump_Qwen_Qwen2.5-Coder-32B-Instruct",
}


def fig_example_chance():
    """Upfront honesty (Initial results) — exp-29. The token probe is near chance
    at telling which WHOLE function is vulnerable (example-level AUC 0.51-0.57),
    even though its token-level AUC is 0.74-0.81. Bars = two example-level reads of
    the same deployed probe (per-model family hue, solid max-pool vs tint+hatch
    final-token); green dashed marks = the token-level headline, floating far above.
    Source: 29-last-token-readout/results/last_token_readout.json (full test,
    n=292; bootstrap CIs)."""
    d = J("29-last-token-readout/results/last_token_readout.json")["models"]
    order = MODEL_ORDER_BY_SIZE
    x = np.arange(len(order)); w = 0.34
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    # per-model family hue (gemma blue, qwen brick), same convention as fig_g /
    # claim_capacity; the two example-level reads are distinguished by fill, solid
    # (max-pool) vs light-tint + hatch (final code token), read off the grey legend.
    keys = [k for k, _ in order]
    is_qwen = ["Qwen" in k for k in keys]
    base = [BRICK if q else ACCENT for q in is_qwen]
    tint = [BRICK_LIGHT if q else LIGHT for q in is_qwen]
    reads = [("max_pool", base, None, -0.5),
             ("last_code_token", tint, "////", 0.5)]
    for key, cols, hatch, mul in reads:
        vals, lo, hi = [], [], []
        for slug in keys:
            m = d[E29KEY[slug]]["full_test"][key]
            v = m["auc"]; ci = m["ci"]
            vals.append(v); lo.append(v - ci[0]); hi.append(ci[1] - v)
        off = mul * w
        ax.bar(x + off, vals, w, color=cols, edgecolor=base, lw=0.6, hatch=hatch, zorder=2)
        ax.errorbar(x + off, vals, yerr=[lo, hi], fmt="none", ecolor=EBAR,
                    elinewidth=0.8, capsize=2, alpha=0.6, zorder=3)
        for i, v in enumerate(vals):
            ax.text(i + off, v + hi[i] + 0.008, f"{v:.2f}", ha="center", fontsize=6.4, color=base[i])
    # token-level AUC: the headline number, a floating dashed mark far above.
    tok = [d[E29KEY[slug]]["token_gate"]["tokens_code_auc_recomputed"] for slug in keys]
    ax.hlines(tok, x - 0.36, x + 0.36, colors=GREEN, linestyles=(0, (4, 2)), lw=2.0, zorder=4)
    for i, v in enumerate(tok):
        ax.text(i, v + 0.013, f"{v:.2f}", ha="center", fontsize=6.4, color=GREEN)
    ax.axhline(0.5, color=FAINT, lw=0.8, ls="--", zorder=1)
    ax.text(len(order) - 0.5, 0.508, "chance", fontsize=7, color=FAINT, va="bottom", ha="right")
    ax.set_xticks(x, [l for _, l in order], fontsize=8)
    ax.set_ylim(0.4, 0.97)
    ax.set_ylabel("AUC", fontsize=8)
    # color-agnostic legend: fill marks the read (family hue is read off the x
    # labels); the dashed green line is the token-level headline.
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    ax.legend(handles=[Patch(facecolor=MUTED, label="max-pool (rank by the function's top token)"),
                       Patch(facecolor=BARGRAY, edgecolor=MUTED, lw=0.6, hatch="////", label="final code token"),
                       Line2D([0], [0], color=GREEN, ls="--", lw=2.0, label="token-level AUC (the headline number)")],
              fontsize=6.8, frameon=False, loc="upper center", ncol=3)
    size_axis_arrow(ax)
    fig.suptitle("Token-level AUC vs example-level reads, per model",
                 fontsize=11, fontweight="bold")
    style(ax)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.88, bottom=0.26)
    save(fig, "fig_example_chance")


def fig_commit_honest():
    """Replaces fig-verbalized (figure 5). Honest example-level comparison at the
    commit position: a linear probe on the answer-commit hidden state beats the
    model's own verbalized yes/no under BOTH a vulnerability-specific and a neutral
    prompt, but a char-n-gram on the raw text (dashed background line) matches or
    beats it on almost every model. Per-model family hue (gemma blue, qwen brick);
    the two probes are the two shades, verbalized is the grey hatched baseline.
    Sources: 31-.../surface_vs_probe.json (primed/neutral probe + char ceiling) and
    30-.../introspection_probe.json (verbalized)."""
    S = J("31-neutral-prompt-and-surface/results/surface_vs_probe.json")
    V = J("30-last-token-introspection/results/introspection_probe.json")["models"]
    char = S["surface"][S["strongest_char"]]["auc"]
    order = MODEL_ORDER_BY_SIZE
    keys = [k for k, _ in order]
    x = np.arange(len(order)); w = 0.26
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    # per-model family hue (gemma blue, qwen brick); the two probes are the two
    # shades (primed = solid base, neutral = light tint), as in fig_example_chance.
    # Verbalized is the asking baseline: a grey hatched fill, set apart from the
    # family-colored probe bars (same secondary fill convention as fig_g).
    is_qwen = ["Qwen" in k for k in keys]
    base = [BRICK if q else ACCENT for q in is_qwen]
    tint = [BRICK_LIGHT if q else LIGHT for q in is_qwen]
    # verbalized = the model's own answer, drawn in the family hue (an apparent
    # "model" tone, not a faint grey) but hatched (in the light tint, so the weave
    # reads on the solid fill) to mark it as asking, not reading.
    series = [("verbalized", base, tint, r"\\\\", -1),
              ("neutral", tint, base, None, 0),
              ("primed", base, base, None, 1)]
    for key, cols, edges, hatch, mul in series:
        vals, lo, hi = [], [], []
        for slug in keys:
            if key == "verbalized":
                v = V[slug]["verbalized"]["test_auc"]; ci = V[slug]["verbalized"]["test_ci"]
            else:
                m = S["models"][slug][key]; v = m["probe_auc"]; ci = m["probe_ci"]
            vals.append(v); lo.append(v - ci[0]); hi.append(ci[1] - v)
        off = mul * w
        ax.bar(x + off, vals, w, color=cols, edgecolor=edges, lw=0.6, hatch=hatch, zorder=2)
        ax.errorbar(x + off, vals, yerr=[lo, hi], fmt="none", ecolor=EBAR,
                    elinewidth=0.7, capsize=2, alpha=0.55, zorder=3)
        for i, v in enumerate(vals):
            ax.text(i + off, v + 0.012, f"{v:.2f}", ha="center", fontsize=5.9, color=base[i])
    # char-n-gram ceiling: a dashed reference pushed to the background (behind bars).
    ax.axhline(char, color=MUTED, lw=1.3, ls="--", zorder=1)
    ax.axhline(0.5, color=FAINT, lw=0.8, ls="--", zorder=1)
    ax.text(len(order) - 0.5, 0.508, "chance", fontsize=7, color=FAINT, va="bottom", ha="right")
    ax.set_xticks(x, [l for _, l in order], fontsize=8)
    ax.set_ylabel("example-level AUC", fontsize=8)
    # extra top headroom so the legend clears the bars (margin between them).
    ax.set_ylim(0.4, 1.04)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Patch(facecolor=ACCENT, label="probe (vulnerability prompt)"),
        Patch(facecolor=LIGHT, edgecolor=ACCENT, lw=0.6, label="probe (neutral prompt)"),
        Patch(facecolor=ACCENT, edgecolor=LIGHT, lw=0.6, hatch=r"\\\\", label="ask the model (verbalized yes/no)"),
        Line2D([0], [0], color=MUTED, ls="--", lw=1.3, label=f"char n-gram on the text ({char:.2f})")],
        fontsize=6.4, frameon=False, loc="upper left", ncol=2, borderaxespad=0.4)
    size_axis_arrow(ax)
    fig.suptitle("Commit-position probe vs verbalized vs char n-gram (example-level AUC)",
                 fontsize=11, fontweight="bold")
    style(ax)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.88, bottom=0.26)
    save(fig, "fig_commit_honest")


def fig_example_verbalized():
    """Example-level (SECONDARY) like-with-like — exp-29. The two example-level
    reads of the deployed code-token probe (max-pool over the function, and the read
    at the final code token) against the model's own verbalized yes/no, one score
    per function on the held-out 292. All three hug chance: asking the model is at
    least as good as any probe read at the example level. Per-model family hue
    (gemma blue, qwen brick); verbalized = base + tint hatch (the model's own
    answer), matching fig_commit_honest. Source:
    29-last-token-readout/results/last_token_readout.json (full test; bootstrap CIs)."""
    d = J("29-last-token-readout/results/last_token_readout.json")["models"]
    order = MODEL_ORDER_BY_SIZE
    keys = [k for k, _ in order]
    is_qwen = ["Qwen" in k for k in keys]
    base = [BRICK if q else ACCENT for q in is_qwen]
    tint = [BRICK_LIGHT if q else LIGHT for q in is_qwen]
    x = np.arange(len(order)); w = 0.26
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    # max-pool = solid base; final code token = light tint; verbalized = base hue
    # + tint hatch (asking the model, not reading it).
    series = [("max_pool", base, base, None, -1),
              ("last_code_token", tint, base, None, 0),
              ("verbalized", base, tint, r"\\\\", 1)]
    for key, cols, edges, hatch, mul in series:
        vals, lo, hi = [], [], []
        for slug in keys:
            m = d[E29KEY[slug]]["full_test"][key]
            v = m["auc"]; ci = m["ci"]
            vals.append(v); lo.append(v - ci[0]); hi.append(ci[1] - v)
        off = mul * w
        ax.bar(x + off, vals, w, color=cols, edgecolor=edges, lw=0.6, hatch=hatch, zorder=2)
        ax.errorbar(x + off, vals, yerr=[lo, hi], fmt="none", ecolor=EBAR,
                    elinewidth=0.7, capsize=2, alpha=0.55, zorder=3)
        for i, v in enumerate(vals):
            ax.text(i + off, v + hi[i] + 0.006, f"{v:.2f}", ha="center", fontsize=5.9, color=base[i])
    ax.axhline(0.5, color=FAINT, lw=0.8, ls="--", zorder=1)
    ax.text(len(order) - 0.5, 0.508, "chance", fontsize=7, color=FAINT, va="bottom", ha="right")
    ax.set_xticks(x, [l for _, l in order], fontsize=8)
    ax.set_ylabel("example-level AUC", fontsize=8)
    ax.set_ylim(0.4, 0.82)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=ACCENT, label="probe, max-pool over the function"),
        Patch(facecolor=LIGHT, edgecolor=ACCENT, lw=0.6, label="probe, final code token"),
        Patch(facecolor=ACCENT, edgecolor=LIGHT, lw=0.6, hatch=r"\\\\", label="ask the model (verbalized yes/no)")],
        fontsize=6.6, frameon=False, loc="upper center", ncol=3, borderaxespad=0.4)
    size_axis_arrow(ax)
    fig.suptitle("Example-level AUC: code-token probe reads vs verbalized",
                 fontsize=11, fontweight="bold")
    style(ax)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.88, bottom=0.26)
    save(fig, "fig_example_verbalized")


# exp-33 operating-point: slice display order + short labels. Injection CWEs first,
# then memory; x-labels colored by family (green inj / vermillion mem), like fig_h.
OP_SLICES = [
    ("overall", "all", "agg"), ("python", "Python", "lang"), ("c_cpp", "C/C++", "lang"),
    ("CWE-089", "SQLi", "inj"), ("CWE-078", "cmd", "inj"),
    ("CWE-022", "path", "inj"), ("CWE-079", "XSS", "inj"),
    ("CWE-125", "OOB-rd", "mem"), ("CWE-416", "UAF", "mem"), ("CWE-476", "NULL", "mem"),
]


def fig_operating_point(source="global", name="fig_operating_point",
                        mode_note="single deployed threshold: 1% FPR on the full held-out pool"):
    """The operating-point view (exp-33, SECONDARY example-level). TPR at a frozen
    1%-FPR threshold, broken out by language and CWE, for the two flagship models.
    `source="global"` (the blog default): one threshold per scorer chosen on the
    full held-out negative pool, recall by slice (deployable + comparable).
    `source="perslice"`: each slice sets its own 1%-FPR threshold on its CWE-matched
    negatives (brittle at small n -- a single negative can swing the number; companion
    only). The story the AUC figures tell, restated as recall: at a 1% false-positive
    budget detection is Python/SQL-injection only; memory CWEs and C/C++ are ~0 for the
    probe, the char-n-gram, AND the model's own yes/no. The probe's one CI-separated
    edge over char (Qwen CWE-089, *) is cleaner top-ranking of the SAME lexical signal
    (within-089 AUC is a tie), not above-lexical. Source:
    33-operating-point-tpr/results/operating_point.json."""
    d = J("33-operating-point-tpr/results/operating_point.json")
    blk_key = "global_threshold" if source == "global" else "methods"
    pd_key = "global_paired_delta" if source == "global" else "paired_delta"
    panels = [("Qwen2.5-Coder-32B-Instruct", "qwen 32B", BRICK),
              ("gemma-3-27b-it", "gemma 27B", ACCENT)]
    fam_color = {"agg": None, "lang": None, "inj": GREEN, "mem": BRICK}
    keys = [k for k, _, _ in OP_SLICES]
    x = np.arange(len(keys)); w = 0.27
    fig, axes = plt.subplots(2, 1, figsize=(7.8, 5.6), sharex=True)
    for ax, (slug, disp, hue) in zip(axes, panels):
        M = d["models"][slug]; gt = M[blk_key]
        gp = M[pd_key]["probe_minus_char"]
        # probe = family hue solid; char = grey lexical baseline; verbalized = family
        # hue + light-tint hatch (asking the model, not reading it).
        series = [("probe", hue, hue, None, -1),
                  ("char_ngram", BARGRAY, MUTED, None, 0),
                  ("verbalized", hue, BRICK_LIGHT if hue == BRICK else LIGHT, r"\\\\", 1)]
        for key, col, edge, hatch, mul in series:
            vals = [gt[key][k]["tpr"] for k in keys]
            ax.bar(x + mul * w, vals, w, color=col, edgecolor=edge, lw=0.6, hatch=hatch, zorder=2)
            for i, v in enumerate(vals):
                ax.text(i + mul * w, v + 0.02, "1.0" if v >= 0.995 else f"{v:.2f}"[1:],
                        ha="center", fontsize=5.3, color=MUTED)
        # mark probe bars where the probe>char paired CI excludes 0
        for i, k in enumerate(keys):
            if gp.get(k, {}).get("ci_excludes_0"):
                ax.text(i - w, gt["probe"][k]["tpr"] + 0.1, "*", ha="center", fontsize=13,
                        color=hue, fontweight="bold")
        for xi in (2.5, 6.5):                       # lang | injection | memory separators
            ax.axvline(xi, color=SEP, lw=0.8, zorder=1)
        ax.set_ylim(0, 1.14); ax.set_yticks([0, 0.5, 1.0])
        ax.set_ylabel("TPR @ 1% FPR", fontsize=8)
        ax.text(0.012, 1.02, disp, transform=ax.transAxes, fontsize=10, fontweight="bold",
                color=hue, va="top")
        style(ax)
    npos = d["models"]["Qwen2.5-Coder-32B-Instruct"][blk_key]["probe"]
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([disp for _, disp, _ in OP_SLICES], fontsize=7.8)
    for tick, (_, _, grp) in zip(axes[1].get_xticklabels(), OP_SLICES):
        if fam_color[grp]:
            tick.set_color(fam_color[grp])
    for i, k in enumerate(keys):
        axes[1].text(i, -0.20, f"n={npos[k]['n_pos']}", ha="center", va="top", fontsize=5.8,
                     color=FAINT, transform=axes[1].get_xaxis_transform())
    axes[0].text(1.5, 1.14, "by language", ha="center", fontsize=6.6, color=FAINT, style="italic")
    axes[0].text(4.5, 1.14, "injection CWEs", ha="center", fontsize=6.6, color=GREEN, style="italic")
    axes[0].text(8.0, 1.14, "memory CWEs", ha="center", fontsize=6.6, color=BRICK, style="italic")
    from matplotlib.patches import Patch
    # legend exemplars in the gemma-blue hue (panels use their own family hue, read
    # off the panel label); char is the model-independent grey baseline.
    fig.legend(handles=[
        # same hero probe as Fig-7 (exp-31 `primed` == exp-30 `deployable`, refit
        # here): the commit-position read under the vulnerability prompt. Label
        # matches fig_commit_honest so the two adjacent figures name it identically.
        Patch(facecolor=ACCENT, label="probe (vulnerability prompt)"),
        Patch(facecolor=BARGRAY, edgecolor=MUTED, lw=0.6, label="char n-gram (lexical baseline)"),
        Patch(facecolor=ACCENT, edgecolor=LIGHT, lw=0.6, hatch=r"\\\\", label="ask the model (verbalized)")],
        fontsize=7, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.945))
    fig.suptitle("Example-level TPR at a 1% false-positive rate, by language and CWE",
                 fontsize=11, fontweight="bold", y=0.995)
    # fig.text(0.5, 0.012, mode_note, ha="center", fontsize=6.4, color=FAINT, style="italic")
    fig.subplots_adjust(left=0.085, right=0.99, top=0.86, bottom=0.155, hspace=0.18)
    save(fig, name)


if __name__ == "__main__":
    for _theme in ("light", "dark"):
        set_theme(_theme)
        # narrative FIG-A..H
        fig_a(); fig_b(); fig_c(); fig_d(); fig_e(); fig_flips(); fig_f(); fig_g(); fig_h()
        # exp-20 (DAT5) string-matcher: cross-probe agreement + FP composition
        fig_stringmatcher()
        # per-claim CLAIM-* figures
        claim_scaling(); claim_capacity(); claim_verbalized()
        claim_langmethod(); claim_family26(); claim_additive(); claim_steering()
        # exp-29/30/31 example-level honesty + commit-position figures
        fig_example_chance(); fig_commit_honest(); fig_example_verbalized()
        # exp-33 operating-point (TPR @ 1% FPR by language + CWE)
        fig_operating_point()
    print("figures written to", FIGS)
    publish()
