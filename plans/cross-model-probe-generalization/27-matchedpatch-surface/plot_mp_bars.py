# [ai-generated]
"""Quick bar plots: matched-patch only, per CWE — probe variants vs best surface baseline."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "plots"
REGIME = "matchedpatch"

BASELINE_KEYS = (
    "char_ngram_lr",
    "combined_abd_lr",
    "token_unigram_lr",
    "conlytrained_char_ngram_lr",
    "conlytrained_combined_abd_lr",
    "keyword_lr",
)
BASELINE_LABELS = {
    "char_ngram_lr": "char-ngram",
    "combined_abd_lr": "combined",
    "token_unigram_lr": "unigram",
    "conlytrained_char_ngram_lr": "conlytr-char",
    "conlytrained_combined_abd_lr": "conlytr-comb",
    "keyword_lr": "keyword-LR",
}
AXES = (
    ("qwen32b", "Qwen2.5-Coder-32B"),
    ("gemma1b", "gemma-3-1b"),
)


def load_axis(slug: str) -> dict:
    return json.load(open(HERE / f"results/exp27_{slug}_axis.json"))["rows"]


def cell(row: dict, key: str) -> dict:
    return row["regimes"][REGIME][key]


def best_baseline(row: dict) -> tuple[str, dict]:
    best_k, best = None, None
    for k in BASELINE_KEYS:
        c = cell(row, k)
        if best is None or c["auc"] > best["auc"]:
            best_k, best = k, c
    return best_k, best


def ci_err(auc: float, ci: list) -> tuple[float, float]:
    lo, hi = ci[0], ci[1]
    return auc - lo, hi - auc


def plot_cwe(cwe: str, rows_q: dict, rows_g: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    series = []
    for slug, axis_label in AXES:
        rows = rows_q if slug == "qwen32b" else rows_g
        if cwe not in rows:
            continue
        row = rows[cwe]
        probe = row["exp25_probe"][REGIME]
        probe_g = cell(row, "probe_general")
        bk, bl = best_baseline(row)
        series.append(
            dict(
                axis=axis_label,
                probe=probe,
                probe_g=probe_g,
                baseline=bl,
                baseline_name=BASELINE_LABELS[bk],
                n=row["n_test_vuln_ex"],
                trust=row.get("trust", True),
            )
        )

    if not series:
        plt.close(fig)
        return

    labels = ["probe (exp-25)", "probeG", "best surface"]
    x = np.arange(len(series))
    width = 0.24
    colors = ["#2563eb", "#7c3aed", "#64748b"]

    for j, (color, getter) in enumerate(
        zip(
            colors,
            (
                lambda s: s["probe"],
                lambda s: s["probe_g"],
                lambda s: s["baseline"],
            ),
        )
    ):
        aucs = [getter(s)["auc"] for s in series]
        yerr_lo, yerr_hi = zip(
            *[ci_err(getter(s)["auc"], getter(s)["ci"]) for s in series]
        )
        offset = (j - 1) * width
        bars = ax.bar(
            x + offset,
            aucs,
            width,
            label=labels[j],
            color=color,
            edgecolor="white",
            linewidth=0.6,
            yerr=np.array([yerr_lo, yerr_hi]),
            capsize=3,
            error_kw=dict(linewidth=1, capthick=1),
        )
        if j == 2:
            for bar, s in zip(bars, series):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    s["baseline_name"],
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                )

    ax.axhline(0.5, color="#94a3b8", ls="--", lw=1, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([s["axis"] for s in series])
    trust = series[0]["trust"]
    n = series[0]["n"]
    suffix = "" if trust else " (untrusted n<10)"
    ax.set_title(f"{cwe} — matched-patch{suffix}  (n={n} test vuln ex)")
    ax.set_ylabel("tokens_code_auc")
    ymax = max(
        max(s["probe"]["ci"][1], s["probe_g"]["ci"][1], s["baseline"]["ci"][1])
        for s in series
    )
    ax.set_ylim(0.35, min(1.05, ymax + 0.12))
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"mp_{cwe.replace('CWE-', '')}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(out)


def main() -> None:
    rows_q = load_axis("qwen32b")
    rows_g = load_axis("gemma1b")
    cwes = sorted(set(rows_q) | set(rows_g), key=lambda c: (c.split("-")[1].zfill(4)))
    for cwe in cwes:
        plot_cwe(cwe, rows_q, rows_g)


if __name__ == "__main__":
    main()
