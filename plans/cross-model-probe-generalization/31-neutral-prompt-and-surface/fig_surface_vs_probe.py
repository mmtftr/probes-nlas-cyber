# [ai-generated]
"""Commit-probe (primed + neutral) vs the char-n-gram lexical ceiling, example level."""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
S = json.loads((HERE / "results/surface_vs_probe.json").read_text())
OUT = HERE / "results"
ORDER = [("gemma-3-1b-it", "gemma 1B"), ("gemma-3-4b-it", "gemma 4B"),
         ("gemma-3-12b-it", "gemma 12B"), ("gemma-3-27b-it", "gemma 27B"),
         ("Qwen2.5-Coder-7B-Instruct", "qwen 7B"), ("Qwen2.5-Coder-32B-Instruct", "qwen 32B")]
char = S["surface"][S["strongest_char"]]["auc"]
uni = S["surface"]["token_unigram"]["auc"]

TH = {"light": dict(green="#2f9e44", teal="#1098ad", brick="#d9480f", ink="#222", bg="white", faint="#bbb", sfx=""),
      "dark": dict(green="#5fb887", teal="#56c5d6", brick="#e8896f", ink="#e6e6e6", bg="none", faint="#555", sfx="-dark")}


def make(th):
    t = TH[th]
    plt.rcParams.update({"text.color": t["ink"], "axes.labelcolor": t["ink"], "xtick.color": t["ink"],
                         "ytick.color": t["ink"], "axes.edgecolor": t["ink"], "figure.facecolor": t["bg"],
                         "axes.facecolor": t["bg"], "savefig.facecolor": t["bg"]})
    x = np.arange(len(ORDER)); w = 0.36
    fig, ax = plt.subplots(figsize=(7.8, 3.6))
    for j, (tag, col, lab) in enumerate([("primed", t["green"], "probe (primed: vuln Q)"),
                                         ("neutral", t["teal"], "probe (neutral: 'what do you think?')")]):
        vals, lo, hi = [], [], []
        for k, _ in ORDER:
            m = S["models"][k].get(tag, {})
            v = m.get("probe_auc", np.nan); ci = m.get("probe_ci", [v, v])
            vals.append(v); lo.append(v - ci[0]); hi.append(ci[1] - v)
        off = (j - 0.5) * w
        ax.bar(x + off, vals, w, color=col, zorder=2, label=lab)
        ax.errorbar(x + off, vals, yerr=[lo, hi], fmt="none", ecolor=t["ink"], elinewidth=0.7, capsize=2, alpha=0.55, zorder=3)
        for i, v in enumerate(vals):
            if v == v:
                ax.text(i + off, v + 0.012, f"{v:.2f}", ha="center", fontsize=6.3, color=col)
    ax.axhline(char, color=t["brick"], lw=1.4, zorder=4, label=f"char-n-gram ceiling ({char:.2f})")
    ax.axhline(uni, color=t["brick"], lw=0.9, ls=":", zorder=4, label=f"token-unigram ({uni:.2f})")
    ax.axhline(0.5, color=t["faint"], lw=0.9, ls="--", zorder=1)
    ax.text(len(ORDER) - 0.5, 0.505, "chance", fontsize=7, color=t["faint"], va="bottom", ha="right")
    ax.set_xticks(x, [l for _, l in ORDER], fontsize=8)
    ax.set_ylabel("example-level AUC", fontsize=8); ax.set_ylim(0.4, 0.9)
    ax.grid(axis="y", lw=0.4, color=t["faint"], zorder=0)
    ax.legend(fontsize=7, frameon=False, loc="upper left", ncol=2)
    fig.suptitle("Commit-position probe vs the lexical ceiling (example level)", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / f"fig_surface_vs_probe{t['sfx']}.png", dpi=200, transparent=(th == "dark"))
    plt.close(fig)


for th in ("light", "dark"):
    make(th)
print("wrote", OUT / "fig_surface_vs_probe.png", "+ -dark")
