# [ai-generated]
"""Plot AUC-vs-layer with mean +/- std error bands from the repeated-split runs.

Reads metrics_variance.json for one or more models and draws, per model, a panel
with example-AUC and token-AUC mean curves + shaded +/-1 std bands across the K
seeds, plus the per-seed-averaged trivial baselines as dashed lines.

    uv run python plot_variance.py \
        --in google_gemma-3-27b-it=metrics_variance_gemma3-27b.json \
             Qwen=metrics_variance_qwen25coder32b.json \
        --out auc_vs_layer_variance.png
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _series(layers, mean_key, std_key):
    xs = [d["layer"] for d in layers if d[mean_key] is not None]
    ms = np.array([d[mean_key] for d in layers if d[mean_key] is not None])
    ss = np.array([d[std_key] if d[std_key] is not None else 0.0
                   for d in layers if d[mean_key] is not None])
    return np.array(xs), ms, ss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inputs", nargs="+", required=True,
                    help="label=path.json entries (label shown as the panel title)")
    ap.add_argument("--out", default="auc_vs_layer_variance.png")
    args = ap.parse_args()

    entries = []
    for spec in args.inputs:
        label, path = spec.split("=", 1)
        entries.append((label, json.loads(Path(path).read_text())))

    n = len(entries)
    fig, axes = plt.subplots(1, n, figsize=(7.5 * n, 5.2), squeeze=False)
    axes = axes[0]

    for ax, (label, rec) in zip(axes, entries):
        layers = rec["layers"]
        seeds = rec.get("seeds", [])
        for key_m, key_s, color, name in [
            ("ex_auc_mean", "ex_auc_std", "tab:blue", "example-AUC"),
            ("tok_auc_mean", "tok_auc_std", "tab:orange", "token-AUC"),
        ]:
            xs, ms, ss = _series(layers, key_m, key_s)
            ax.plot(xs, ms, "-o", ms=3, color=color, label=name)
            ax.fill_between(xs, ms - ss, ms + ss, color=color, alpha=0.20)

        # Baselines (mean across seeds) as horizontal dashed lines.
        base = rec.get("baseline_auc", {})
        for bname, style in [("length", ":"), ("regex", "-."), ("random", "--")]:
            b = base.get(bname, {})
            if b.get("mean") is not None:
                ax.axhline(b["mean"], color="gray", ls=style, lw=1,
                           label=f"{bname} baseline ({b['mean']:.2f})")

        # Mark best mean-example-AUC layer.
        if rec.get("best_layer") is not None:
            ax.axvline(rec["best_layer"], color="tab:blue", ls=":", lw=1, alpha=0.6)
            ax.annotate(
                f"best L{rec['best_layer']} (frac {rec['best_layer_frac']:.2f})\n"
                f"ex-AUC {rec['best_ex_auc_mean']:.3f}±{rec['best_ex_auc_std']:.3f}",
                xy=(rec["best_layer"], rec["best_ex_auc_mean"]),
                xytext=(0.45, 0.08), textcoords="axes fraction", fontsize=8,
                arrowprops=dict(arrowstyle="->", color="tab:blue", lw=0.8))

        ax.set_title(f"{label}  ({rec['n_layers']} layers, {len(seeds)} splits {seeds})", fontsize=9)
        ax.set_xlabel("layer index")
        ax.set_ylabel("held-out AUC")
        ax.set_ylim(0.40, 0.90)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("Per-layer probe AUC — repeated group-clean splits (mean ± 1 std)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
