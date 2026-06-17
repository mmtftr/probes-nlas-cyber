# [ai-generated]
"""Corrected verbalized comparison — everything at the EXAMPLE level.

Replaces the blog's fig-verbalized metric mismatch (probe token-AUC vs verbalized
example-AUC) with a like-with-like example-level comparison, and adds the
last-code-token probe read the user asked for. Light + '-dark' variants, matching
docs/blog/make_claim_figs.py theming so it can drop into the post.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = json.loads((HERE / "results/last_token_readout.json").read_text())
OUT = HERE / "results"

# 6 models with a verbalized run, in the blog's claim_verbalized order.
ORDER = [("logitdump_google_gemma-3-1b-it", "gemma 1B"),
         ("logitdump_google_gemma-3-4b-it", "gemma 4B"),
         ("logitdump_google_gemma-3-12b-it", "gemma 12B"),
         ("logitdump_google_gemma-3-27b-it", "gemma 27B"),
         ("logitdump_Qwen_Qwen2.5-Coder-7B-Instruct", "qwen 7B"),
         ("logitdump_Qwen_Qwen2.5-Coder-32B-Instruct", "qwen 32B")]

THEMES = {
    "light": dict(accent="#3b5bdb", brick="#d9480f", green="#2f9e44", bg="white",
                  ink="#222", faint="#bbb", suffix=""),
    "dark": dict(accent="#6f97ff", brick="#e8896f", green="#5fb887", bg="none",
                 ink="#e6e6e6", faint="#555", suffix="-dark"),
}


def make(theme):
    t = THEMES[theme]
    plt.rcParams.update({"text.color": t["ink"], "axes.labelcolor": t["ink"],
                         "xtick.color": t["ink"], "ytick.color": t["ink"],
                         "axes.edgecolor": t["ink"],
                         "figure.facecolor": t["bg"], "axes.facecolor": t["bg"],
                         "savefig.facecolor": t["bg"]})
    keys = [("last_code_token", t["green"], "probe @ last code token"),
            ("max_pool", t["accent"], "probe, max-pool (example)"),
            ("verbalized", t["brick"], "verbalized yes/no")]
    x = np.arange(len(ORDER))
    w = 0.26
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    for j, (k, col, lab) in enumerate(keys):
        vals, lo, hi = [], [], []
        for s, _ in ORDER:
            cell = RES["models"][s]["full_test"][k]
            vals.append(cell["auc"]); lo.append(cell["auc"] - cell["ci"][0]); hi.append(cell["ci"][1] - cell["auc"])
        off = (j - 1) * w
        ax.bar(x + off, vals, w, color=col, zorder=2, label=lab)
        ax.errorbar(x + off, vals, yerr=[lo, hi], fmt="none", ecolor=t["ink"],
                    elinewidth=0.8, capsize=2, zorder=3, alpha=0.6)
        for i, v in enumerate(vals):
            ax.text(i + off, v + 0.012, f"{v:.2f}", ha="center", fontsize=6.5, color=col)
    ax.axhline(0.5, color=t["faint"], lw=0.9, ls="--", zorder=1)
    ax.text(len(ORDER) - 0.5, 0.505, "chance", fontsize=7, color=t["faint"], va="bottom", ha="right")
    ax.set_xticks(x, [l for _, l in ORDER], fontsize=8)
    ax.set_ylabel("example-level AUC", fontsize=8)
    ax.set_ylim(0.4, 0.78)
    ax.grid(axis="y", lw=0.4, color=t["faint"], zorder=0)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left", ncol=3)
    fig.suptitle("Probe vs verbalized, like-with-like (example level)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / f"fig_last_token{t['suffix']}.png", dpi=200,
                transparent=(theme == "dark"))
    plt.close(fig)


for th in ("light", "dark"):
    make(th)
print("wrote", OUT / "fig_last_token.png", "+ -dark")
