# [ai-generated]
"""Plot example-AUC vs alpha for base vs neg_incl span-max, per layer, per model.

Grid: one row per model, one column per swept layer. Each subplot has two lines
(base, neg_incl) with +/-1 std bands over the seeds; a dashed line marks the
length baseline (0.575).

    uv run --with matplotlib --with numpy python plot_loss_alpha.py \
        --in "Gemma-3-27B=metrics_loss_alpha_gemma.json" \
             "Qwen2.5-Coder-32B=metrics_loss_alpha_qwen.json" \
        --out loss_alpha_sweep.png
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LENGTH_BASELINE = 0.575
COLORS = {"base": "tab:gray", "neg_incl": "tab:red"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inputs", nargs="+", required=True)
    ap.add_argument("--out", default="loss_alpha_sweep.png")
    args = ap.parse_args()

    models = [(spec.split("=", 1)[0], json.loads(Path(spec.split("=", 1)[1]).read_text()))
              for spec in args.inputs]
    n_layers_max = max(len({r["layer"] for r in rec["rows"]}) for _, rec in models)
    fig, axes = plt.subplots(len(models), n_layers_max,
                             figsize=(3.6 * n_layers_max, 3.6 * len(models)),
                             squeeze=False)

    for mi, (label, rec) in enumerate(models):
        layers = sorted({r["layer"] for r in rec["rows"]})
        for ci in range(n_layers_max):
            ax = axes[mi][ci]
            if ci >= len(layers):
                ax.axis("off")
                continue
            li = layers[ci]
            for loss in ("base", "neg_incl"):
                pts = sorted([r for r in rec["rows"] if r["layer"] == li and r["loss"] == loss
                              and r["ex_auc_mean"] is not None], key=lambda r: r["alpha"])
                if not pts:
                    continue
                xs = [p["alpha"] for p in pts]
                ms = np.array([p["ex_auc_mean"] for p in pts])
                ss = np.array([p["ex_auc_std"] or 0.0 for p in pts])
                ax.plot(xs, ms, "-o", ms=4, color=COLORS[loss], label=loss)
                ax.fill_between(xs, ms - ss, ms + ss, color=COLORS[loss], alpha=0.18)
            ax.axhline(LENGTH_BASELINE, color="k", ls=":", lw=0.8, label="length base")
            ax.set_xscale("log")
            ax.set_title(f"{label}  L{li}", fontsize=9)
            ax.set_xlabel("alpha (in-span weight)")
            if ci == 0:
                ax.set_ylabel("held-out example-AUC")
            ax.grid(alpha=0.25)
            ax.set_ylim(0.55, 0.80)
            if mi == 0 and ci == 0:
                ax.legend(fontsize=7, loc="lower left")

    fig.suptitle("Span-max loss sweep: example-AUC vs α, base vs neg_incl (mean ± 1 std over 5 splits)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
