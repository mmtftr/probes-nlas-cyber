# [ai-generated]
"""Figure: per-CWE matched-patch AUC, probe vs lexical baselines.

Reads results/percwe_matchedpatch.json (from consolidate.py) and draws, for one
model, a grouped horizontal bar chart: each CWE gets probe / char-ngram /
token-unigram bars with 95% bootstrap CI whiskers, a chance line at 0.5, the
injection block separated from the memory block, and n<10 CWEs greyed + labeled.

CI whiskers overlap heavily between probe and lexical in the memory cells; that
is intentional and honest -- no probe-over-lexical contrast is CI-separated.

Usage: uv run --with matplotlib --with numpy python make_fig.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"

# injection block then memory block; within each, the order to display top->bottom
ORDER = ["CWE-089", "CWE-078", "CWE-022", "CWE-079",
         "CWE-125", "CWE-416", "CWE-476", "CWE-190", "CWE-787"]
METHODS = [("probe", "probe", "#C44E52"),
           ("char_ngram_lr", "char n-gram", "#4C72B0"),
           ("token_unigram_lr", "token unigram", "#9AA7B8")]
BAR_H = 0.26


def plot_model(ax, model: dict) -> None:
    rows = model["rows"]
    # y positions: leave a gap between injection and memory blocks
    ypos, ylabels, gap = [], [], 0.0
    y = 0.0
    prev_fam = None
    pos_of = {}
    for c in ORDER:
        fam = rows[c]["family"]
        if prev_fam is not None and fam != prev_fam:
            y += 1.0  # block gap
        pos_of[c] = y
        ypos.append(y)
        n = rows[c]["n_test_pos"]
        lown = "" if rows[c]["trust"] else f"  (n={n})"
        ylabels.append(f"{c}\n{rows[c]['name']}{lown}")
        y += 1.0
        prev_fam = fam

    for c in ORDER:
        base = pos_of[c]
        trust = rows[c]["trust"]
        for j, (key, _, color) in enumerate(METHODS):
            yy = base + (1 - j) * BAR_H
            auc = rows[c][key]["auc"]
            lo, hi = rows[c][key]["ci"]
            a = 1.0 if trust else 0.4
            ax.barh(yy, auc - 0.4, left=0.4, height=BAR_H, color=color, alpha=a,
                    edgecolor="white", linewidth=0.5, zorder=2)
            ax.plot([lo, hi], [yy, yy], color="#333333", lw=1.0, alpha=a * 0.8, zorder=3)
            ax.plot([auc], [yy], "|", color="#222222", ms=5, alpha=a, zorder=4)

    ax.axvline(0.5, color="#888888", ls="--", lw=1.0, zorder=1)
    ax.text(0.5, max(ypos) + 1.0, "chance", color="#888888", fontsize=8,
            ha="center", va="bottom")
    ax.set_yticks(ypos)
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel("token-level AUC (live code), matched-patch", fontsize=9)
    ax.set_title(model["model"], fontsize=11, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(axis="x", color="#dddddd", lw=0.6, zorder=0)
    # family band labels
    inj_y = [pos_of[c] for c in ORDER if rows[c]["family"] == "inj"]
    mem_y = [pos_of[c] for c in ORDER if rows[c]["family"] == "mem"]
    ax.text(0.405, min(inj_y) - 0.45, "INJECTION", fontsize=8, color="#555",
            weight="bold", va="bottom")
    ax.text(0.405, min(mem_y) - 0.45, "MEMORY", fontsize=8, color="#555",
            weight="bold", va="bottom")


def main() -> None:
    payload = json.loads((OUT / "percwe_matchedpatch.json").read_text())
    models = {m["model"]: m for m in payload["models"]}
    legend = [Patch(facecolor=c, label=lab) for _, lab, c in METHODS]

    # headline: Qwen alone
    qwen = models["Qwen2.5-Coder-32B"]
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    plot_model(ax, qwen)
    ax.legend(handles=legend, loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_percwe_matchedpatch_qwen.png", dpi=170, bbox_inches="tight")

    # both models, side by side
    fig2, axes = plt.subplots(1, 2, figsize=(13.5, 6.2), sharey=True)
    for ax, name in zip(axes, ["Qwen2.5-Coder-32B", "Gemma-3-1B"]):
        plot_model(ax, models[name])
    axes[1].legend(handles=legend, loc="lower right", fontsize=8, frameon=False)
    fig2.tight_layout()
    fig2.savefig(OUT / "fig_percwe_matchedpatch_both.png", dpi=170, bbox_inches="tight")
    print("[fig] wrote results/fig_percwe_matchedpatch_{qwen,both}.png")


if __name__ == "__main__":
    main()
