# [ai-generated]
"""exp-21 re-exec figures: 9x9 cross-CWE token-AUC transfer heatmaps (both models)
+ a 2x2 injection/memory block-mean summary with bootstrap CIs. Tiny-n test CWEs
(n_test<10) are hatched and never to be read as individual evidence."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figs"; FIGS.mkdir(exist_ok=True)
ORDER = ["CWE-022", "CWE-078", "CWE-079", "CWE-089", "CWE-125", "CWE-190", "CWE-416", "CWE-476", "CWE-787"]
INJ = {"CWE-089", "CWE-078", "CWE-022", "CWE-079"}
MODELS = [("qwen32b", "Qwen2.5-Coder-32B (L25)"), ("gemma1b", "Gemma-3-1b-it (L25)")]
MIN_TRUST = 10


def load(key):
    return json.load(open(HERE / "results" / key / "matrix_tokenauc.json"))


def heatmap():
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4))
    for ax, (key, name) in zip(axes, MODELS):
        d = load(key)
        cwes = [c for c in ORDER if c in d["cwes"]]
        M = d["auc_natural_own"]; nt = d["n_test_vuln_examples"]
        A = np.array([[M[cx][cy] if M[cx][cy] is not None else np.nan for cy in cwes] for cx in cwes])
        im = ax.imshow(A, cmap="RdBu_r", vmin=0.2, vmax=0.95, aspect="auto")
        ax.set_xticks(range(len(cwes))); ax.set_yticks(range(len(cwes)))
        ax.set_xticklabels([c[-3:] for c in cwes]); ax.set_yticklabels([c[-3:] for c in cwes])
        ax.set_xlabel("TEST CWE"); ax.set_ylabel("TRAIN-probe CWE")
        ninj = sum(c in INJ for c in cwes)
        for b in (ninj - 0.5,):
            ax.axhline(b, color="k", lw=1.6); ax.axvline(b, color="k", lw=1.6)
        for i, cx in enumerate(cwes):
            for j, cy in enumerate(cwes):
                v = A[i, j]
                if np.isnan(v):
                    continue
                tiny = nt[cy] < MIN_TRUST
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if (v > 0.78 or v < 0.4) else "black",
                        alpha=0.55 if tiny else 1.0)
                if i == j:
                    ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, edgecolor="lime", lw=2))
                if tiny:
                    ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, edgecolor="grey",
                                           lw=0.6, hatch="///", alpha=0.35))
        ax.set_title(f"{name}\ntoken-AUC; green=self, hatched=test n<10 (noise)", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="tokens_code_auc")
    fig.suptitle("Cross-CWE transfer on tokens_code_auc — injection block (top-left) hot, memory cold",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGS / "transfer_tokenauc_heatmaps.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def blocks():
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    order = ["inj->inj", "inj->mem", "mem->inj", "mem->mem"]
    x = np.arange(len(order)); w = 0.38
    for k, (key, name) in enumerate(MODELS):
        d = load(key)
        bm = d["block_auc_natural_own"]; ci = d["block_auc_natural_own_ci"]
        vals = [bm[o] for o in order]
        lo = [bm[o] - ci[o][0] for o in order]; hi = [ci[o][1] - bm[o] for o in order]
        ax.bar(x + (k - .5) * w, vals, w, yerr=[lo, hi], capsize=4,
               label=name, color=["#2b6", "#36c"][k], alpha=0.85)
    ax.axhline(0.5, color="k", ls="--", lw=1, label="chance")
    ax.set_xticks(x); ax.set_xticklabels(order)
    ax.set_ylabel("block-mean tokens_code_auc (95% bootstrap CI)")
    ax.set_ylim(0.30, 0.85)
    ax.set_title("Transfer blocks: only inj→inj clears chance; memory blocks at/below it")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "transfer_tokenauc_blocks.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def diagonal():
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    cwes = ORDER
    xpos = np.arange(len(cwes)); w = 0.38
    for k, (key, name) in enumerate(MODELS):
        d = load(key); dd = d["diagonal_natural"]
        vals = [dd[c]["auc"] if c in dd else np.nan for c in cwes]
        los = [dd[c]["auc"] - dd[c]["ci"][0] if c in dd else 0 for c in cwes]
        his = [dd[c]["ci"][1] - dd[c]["auc"] if c in dd else 0 for c in cwes]
        ax.bar(xpos + (k - .5) * w, vals, w, yerr=[los, his], capsize=3,
               label=name, color=["#2b6", "#36c"][k], alpha=0.85)
    ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.axhline(0.73, color="purple", ls=":", lw=1, label="exp-10 memory (full-SVEN regime)")
    ax.set_xticks(xpos); ax.set_xticklabels([c[-3:] for c in cwes])
    ax.axvline(3.5, color="k", lw=1.2)
    ax.text(1.5, 0.32, "INJECTION", ha="center", fontsize=9, weight="bold")
    ax.text(6, 0.32, "MEMORY (all test n<10)", ha="center", fontsize=9, weight="bold")
    ax.set_ylabel("self-detection tokens_code_auc")
    ax.set_ylim(0.30, 1.0)
    ax.set_title("Per-CWE self-detection: injection strong, memory near-chance IN THIS (subtractive) REGIME")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGS / "self_detection_diagonal.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    heatmap(); blocks(); diagonal()
    print("wrote figs/transfer_tokenauc_heatmaps.png, transfer_tokenauc_blocks.png, self_detection_diagonal.png")
