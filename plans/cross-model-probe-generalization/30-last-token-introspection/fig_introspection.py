# [ai-generated]
"""Last-token introspection probe vs verbalized vs code-token probe (example level).

Per model: deployable introspection-probe test AUC (with CI), verbalized P(yes),
and the exp-29 max-pool code-token read — all example-level. Overlays each model's
label-permutation-null p95 (the "is it just overfit?" ceiling) as a caret; a probe
bar above its caret + above 0.5 is the headline signal. Light + '-dark' variants.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = json.loads((HERE / "results/introspection_probe.json").read_text())["models"]
OUT = HERE / "results"

ORDER = [("gemma-3-1b-it", "gemma 1B"), ("gemma-3-4b-it", "gemma 4B"),
         ("gemma-3-12b-it", "gemma 12B"), ("gemma-3-27b-it", "gemma 27B"),
         ("Qwen2.5-Coder-7B-Instruct", "qwen 7B"), ("Qwen2.5-Coder-32B-Instruct", "qwen 32B")]
ORDER = [(k, l) for k, l in ORDER if k in RES]

THEMES = {"light": dict(green="#2f9e44", accent="#3b5bdb", brick="#d9480f", gray="#868e96",
                        ink="#222", bg="white", faint="#bbb", suffix=""),
          "dark": dict(green="#5fb887", accent="#6f97ff", brick="#e8896f", gray="#9aa3af",
                       ink="#e6e6e6", bg="none", faint="#555", suffix="-dark")}


def make(theme):
    t = THEMES[theme]
    plt.rcParams.update({"text.color": t["ink"], "axes.labelcolor": t["ink"],
                         "xtick.color": t["ink"], "ytick.color": t["ink"], "axes.edgecolor": t["ink"],
                         "figure.facecolor": t["bg"], "axes.facecolor": t["bg"], "savefig.facecolor": t["bg"]})
    bars = [("probe", t["green"], "introspection probe (deployable)"),
            ("verb", t["brick"], "verbalized yes/no"),
            ("e29", t["gray"], "code-token probe (max-pool, exp-29)")]
    x = np.arange(len(ORDER)); w = 0.26
    fig, ax = plt.subplots(figsize=(7.8, 3.6))
    for j, (key, col, lab) in enumerate(bars):
        vals, lo, hi = [], [], []
        for k, _ in ORDER:
            m = RES[k]
            if key == "probe":
                v = m["deployable"]["test_auc"]; ci = m["deployable"]["test_ci"]
            elif key == "verb":
                v = m["verbalized"]["test_auc"]; ci = m["verbalized"]["test_ci"]
            else:
                v = m["exp29"].get("max_pool") or np.nan; ci = [v, v]
            vals.append(v); lo.append(v - ci[0]); hi.append(ci[1] - v)
        off = (j - 1) * w
        ax.bar(x + off, vals, w, color=col, zorder=2, label=lab)
        ax.errorbar(x + off, vals, yerr=[lo, hi], fmt="none", ecolor=t["ink"], elinewidth=0.7, capsize=2, alpha=0.55, zorder=3)
        for i, v in enumerate(vals):
            if v == v:
                ax.text(i + off, v + 0.012, f"{v:.2f}", ha="center", fontsize=6.3, color=col)
    # null ceilings over the probe bar: perm-null p95 (filled) + random-dir p95 (hollow, higher at d>>n)
    for i, (k, _) in enumerate(ORDER):
        ax.plot([i - w], [RES[k]["perm_null"]["p95"]], marker="v", ms=6, color=t["ink"], zorder=4)
        ax.plot([i - w], [RES[k]["random_dir_null"]["p95"]], marker="v", ms=6, mfc="none",
                mec=t["ink"], mew=1.1, zorder=4)
    ax.plot([], [], marker="v", ls="none", color=t["ink"], label="perm-null p95")
    ax.plot([], [], marker="v", ls="none", mfc="none", mec=t["ink"], label="random-dir p95 (d≫n ceiling)")
    ax.axhline(0.5, color=t["faint"], lw=0.9, ls="--", zorder=1)
    ax.text(len(ORDER) - 0.5, 0.505, "chance", fontsize=7, color=t["faint"], va="bottom", ha="right")
    ax.set_xticks(x, [l for _, l in ORDER], fontsize=8)
    ax.set_ylabel("example-level AUC", fontsize=8)
    ax.set_ylim(0.4, 0.92)
    ax.grid(axis="y", lw=0.4, color=t["faint"], zorder=0)
    ax.legend(fontsize=7, frameon=False, loc="upper left", ncol=2)
    fig.suptitle("Last-token introspection probe vs verbalized (example level)", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / f"fig_introspection{t['suffix']}.png", dpi=200, transparent=(theme == "dark"))
    plt.close(fig)


for th in ("light", "dark"):
    make(th)
print("wrote", OUT / "fig_introspection.png", "+ -dark")
