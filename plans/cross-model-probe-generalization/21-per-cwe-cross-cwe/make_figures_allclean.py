# [ai-generated]
"""exp-21 (corrected) figures: cross-CWE token-AUC transfer heatmaps (exp-10
recipe, train vs all-clean) + diagonal-vs-exp-10 reproduction bars + 2x2
within/cross-family block means with bootstrap CIs. Tiny-n test CWEs (n<10)
hatched (noise)."""
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
EXP10 = {"CWE-089": 0.983, "CWE-078": 0.923, "CWE-022": 0.943, "CWE-079": 0.863,
         "CWE-125": 0.732, "CWE-476": 0.640, "CWE-787": 0.670, "CWE-416": 0.766, "CWE-190": 0.767}


def load(key):
    return json.load(open(HERE / "results" / key / "transfer_allclean.json"))


def heatmap():
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.6))
    for ax, (key, name) in zip(axes, MODELS):
        d = load(key); M = d["auc"]; nt = d["n_test_pos"]
        cwes = [c for c in ORDER if c in d["cwes"]]
        A = np.array([[M[cx][cy] if M[cx][cy] is not None else np.nan for cy in cwes] for cx in cwes])
        im = ax.imshow(A, cmap="RdBu_r", vmin=0.2, vmax=0.95, aspect="auto")
        ax.set_xticks(range(len(cwes))); ax.set_yticks(range(len(cwes)))
        ax.set_xticklabels([c[-3:] for c in cwes]); ax.set_yticklabels([c[-3:] for c in cwes])
        ax.set_xlabel("TEST CWE"); ax.set_ylabel("TRAIN-probe CWE")
        ninj = sum(c in INJ for c in cwes)
        ax.axhline(ninj - 0.5, color="k", lw=1.6); ax.axvline(ninj - 0.5, color="k", lw=1.6)
        for i, cx in enumerate(cwes):
            for j, cy in enumerate(cwes):
                v = A[i, j]
                if np.isnan(v):
                    continue
                tiny = nt[cy] < 10
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if (v > 0.78 or v < 0.4) else "black", alpha=0.5 if tiny else 1.0)
                if i == j:
                    ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, edgecolor="lime", lw=2))
                if tiny:
                    ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, edgecolor="grey", lw=0.6, hatch="///", alpha=0.35))
        ax.set_title(f"{name}\ntrained vs ALL-clean (exp-10 recipe); green=self, hatched=test n<10", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="tokens_code_auc")
    fig.suptitle("Cross-CWE transfer (exp-10 recipe): block-diagonal — within-family transfers, cross-family at/below chance",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGS / "allclean_transfer_heatmaps.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def diagonal_repro():
    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    cwes = ORDER; x = np.arange(len(cwes)); w = 0.26
    qd = load("qwen32b")["diagonal"]; gd = load("gemma1b")["diagonal"]
    ax.bar(x - w, [EXP10[c] for c in cwes], w, label="exp-10 (Qwen-32B, full-SVEN)", color="#999")
    qv = [qd[c]["auc"] if c in qd else np.nan for c in cwes]
    ql = [qd[c]["auc"] - qd[c]["ci"][0] if c in qd else 0 for c in cwes]
    qh = [qd[c]["ci"][1] - qd[c]["auc"] if c in qd else 0 for c in cwes]
    ax.bar(x, qv, w, yerr=[ql, qh], capsize=3, label="this re-exec (Qwen-32B)", color="#2b6")
    gv = [gd[c]["auc"] if c in gd else np.nan for c in cwes]
    ax.bar(x + w, gv, w, label="this re-exec (Gemma-1b)", color="#36c", alpha=0.85)
    ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.axvline(3.5, color="k", lw=1.2)
    ax.text(1.5, 0.30, "INJECTION", ha="center", fontsize=9, weight="bold")
    ax.text(6, 0.30, "MEMORY", ha="center", fontsize=9, weight="bold")
    ax.set_xticks(x); ax.set_xticklabels([c[-3:] for c in cwes])
    ax.set_ylabel("self-detection tokens_code_auc"); ax.set_ylim(0.28, 1.0)
    ax.set_title("Self-detection reproduces exp-10 exactly (Qwen Δ=±0.000): memory IS learnable on its own data")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGS / "allclean_diagonal_repro.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def blocks():
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    order = ["inj->inj", "inj->mem", "mem->inj", "mem->mem"]
    x = np.arange(len(order)); w = 0.38
    for k, (key, name) in enumerate(MODELS):
        d = load(key); bm = d["block_auc_trusted"]; ci = d["block_auc_trusted_ci"]
        vals = [bm[o] for o in order]
        lo = [max(0, bm[o] - ci[o][0]) for o in order]; hi = [max(0, ci[o][1] - bm[o]) for o in order]
        ax.bar(x + (k - .5) * w, vals, w, yerr=[lo, hi], capsize=4, label=name,
               color=["#2b6", "#36c"][k], alpha=0.85)
    ax.axhline(0.5, color="k", ls="--", lw=1, label="chance")
    ax.set_xticks(x); ax.set_xticklabels(order)
    ax.set_ylabel("block-mean tokens_code_auc (trusted n≥10, 95% boot CI)")
    ax.set_ylim(0.25, 0.85)
    ax.set_title("Family-structured transfer: within-family > chance, cross-family at/below chance")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "allclean_blocks.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    heatmap(); diagonal_repro(); blocks()
    print("wrote figs/allclean_transfer_heatmaps.png, allclean_diagonal_repro.png, allclean_blocks.png")
