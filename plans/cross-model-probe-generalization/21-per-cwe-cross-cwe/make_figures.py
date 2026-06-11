# [ai-generated]
"""exp-21 figures: CWE×CWE transfer heatmaps (pair-accuracy, natural + balanced)
per model, with injection/memory block dividers and small-test-n cells hatched.
Run: uv run --with matplotlib python make_figures.py"""
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
MODELS = {"qwen32b": "Qwen2.5-Coder-32B  (L25)", "gemma1b": "Gemma-3-1b-it  (L25)"}
INJ = ["CWE-022", "CWE-078", "CWE-079", "CWE-089"]
MEM = ["CWE-125", "CWE-190", "CWE-416", "CWE-476", "CWE-787"]
ORDER = INJ + MEM
SHORT = {c: c.replace("CWE-", "") for c in ORDER}


def heat(ax, M, cwes, nt, title):
    n = len(cwes)
    A = np.full((n, n), np.nan)
    for i, c in enumerate(cwes):
        for j, cp in enumerate(cwes):
            v = M[c].get(cp)
            A[i, j] = v if (v is not None and v == v) else np.nan
    im = ax.imshow(A, cmap="RdBu_r", vmin=0.2, vmax=0.8, aspect="equal")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([SHORT[c] + f"\nn={nt[c]}" for c in cwes], fontsize=6.5)
    ax.set_yticklabels([SHORT[c] for c in cwes], fontsize=7)
    ax.set_xlabel("test CWE (held-out)", fontsize=8)
    ax.set_ylabel("train-CWE probe", fontsize=8)
    ax.set_title(title, fontsize=9)
    for i, c in enumerate(cwes):
        for j, cp in enumerate(cwes):
            if np.isnan(A[i, j]):
                continue
            small = nt[cp] < 5
            ax.text(j, i, f"{A[i,j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(A[i, j] - 0.5) > 0.22 else "black",
                    alpha=0.5 if small else 1.0)
    # injection/memory dividers
    k = len(INJ)
    for xy in (k - 0.5,):
        ax.axhline(xy, color="k", lw=1.2); ax.axvline(xy, color="k", lw=1.2)
    ax.add_patch(Rectangle((-.5, -.5), k, k, fill=False, ec="#2f4b7c", lw=2))
    return im


def main():
    fig, axes = plt.subplots(2, 2, figsize=(11, 10.5))
    for r, key in enumerate(MODELS):
        m = json.load(open(HERE / "results" / key / "matrix.json"))
        cwes = [c for c in ORDER if c in m["cwes"]]
        nt = m["n_test_pairs"]
        im = heat(axes[r, 0], m["pairacc_natural"], cwes, nt, f"{MODELS[key]} — natural")
        heat(axes[r, 1], m["pairacc_balanced"], cwes, nt, f"{MODELS[key]} — balanced-15")
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.03)
    cbar.set_label("pair-accuracy (vuln vs its own patch)  ·  0.5 = chance", fontsize=8)
    fig.suptitle("Cross-CWE transfer: per-CWE probe (row) detecting each CWE (col)\n"
                 "injection block (blue box) self-detects + partial 089↔078 transfer; "
                 "memory block ≈ chance even on its own diagonal",
                 fontsize=10, y=0.98)
    out = FIGS / "cross_cwe_heatmaps.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")

    # block summary bar (natural) with CIs, both models
    import math
    def wilson(p, nn, z=1.96):
        if nn == 0: return (np.nan, np.nan)
        d = 1 + z*z/nn; c = (p + z*z/(2*nn))/d
        h = z*math.sqrt(p*(1-p)/nn + z*z/(4*nn*nn))/d
        return c-h, c+h
    def blocks(M, nt, cwes):
        out = {}
        for gr, rows in (("inj", INJ), ("mem", MEM)):
            for gc, cols in (("inj", INJ), ("mem", MEM)):
                s = t = 0
                for c in rows:
                    if c not in cwes: continue
                    for cp in cols:
                        if cp not in cwes: continue
                        v = M[c].get(cp); nn = nt[cp]
                        if v is not None and v == v and nn: s += round(v*nn); t += nn
                p = s/t if t else np.nan
                lo, hi = wilson(p, t); out[f"{gr}→{gc}"] = (p, lo, hi, t)
        return out
    fig2, ax = plt.subplots(figsize=(8, 4))
    labels = ["inj→inj", "inj→mem", "mem→inj", "mem→mem"]
    x = np.arange(len(labels)); w = 0.38
    for i, key in enumerate(MODELS):
        m = json.load(open(HERE / "results" / key / "matrix.json"))
        cwes = [c for c in ORDER if c in m["cwes"]]
        bk = blocks(m["pairacc_natural"], m["n_test_pairs"], cwes)
        ys = [bk[l][0] for l in labels]
        errlo = [bk[l][0]-bk[l][1] for l in labels]; errhi = [bk[l][2]-bk[l][0] for l in labels]
        ax.bar(x + (i-0.5)*w, ys, w, yerr=[errlo, errhi], capsize=3,
               label=MODELS[key], color=["#2f4b7c", "#b1442e"][i], alpha=0.85)
    ax.axhline(0.5, color="k", ls="--", lw=1, label="chance")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0.3, 0.7)
    ax.set_ylabel("pooled pair-accuracy"); ax.legend(fontsize=8)
    ax.set_title("Injection/memory block transfer (natural, 95% Wilson CI)", fontsize=10)
    out2 = FIGS / "block_summary.png"
    fig2.savefig(out2, dpi=140, bbox_inches="tight")
    print(f"wrote {out2}")


if __name__ == "__main__":
    main()
