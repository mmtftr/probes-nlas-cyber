# [ai-generated]
"""Plot held-out example-AUC per (feature_set x arch) config, per model.

One subplot per model: a point-with-errorbar (ex-AUC mean ±1 std over the 5
splits) for each (feature_set, arch) config along the x-axis, labelled e.g.
"L19/linear", "9-19-26-61/mlp256". Two reference lines: the linear_single_best
mean (dashed) and the length baseline 0.575 (dotted).

    uv run --with matplotlib --with numpy python plot_richer.py \
        --in "Gemma-3-27B=metrics_richer_gemma.json" \
             "Qwen2.5-Coder-32B=metrics_richer_qwen.json" \
        --out richer_probe_sweep.png
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LENGTH_BASELINE = 0.575
ARCH_COLORS = {"linear": "tab:blue", "mlp256": "tab:orange", "mlp512": "tab:red"}


def _config_label(row) -> str:
    layers = row.get("layers") or [int(x) for x in row["feature_set"].split(",")]
    fs = f"L{layers[0]}" if len(layers) == 1 else "-".join(str(li) for li in layers)
    return f"{fs}/{row['arch']}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inputs", nargs="+", required=True,
                    help="label=metrics_richer.json specs")
    ap.add_argument("--out", default="richer_probe_sweep.png")
    args = ap.parse_args()

    models = [(spec.split("=", 1)[0], json.loads(Path(spec.split("=", 1)[1]).read_text()))
              for spec in args.inputs]
    fig, axes = plt.subplots(1, len(models),
                             figsize=(max(6.0, 4.0 * len(models)), 4.8),
                             squeeze=False)

    for mi, (label, rec) in enumerate(models):
        ax = axes[0][mi]
        rows = [r for r in rec["rows"] if r["ex_auc_mean"] is not None]
        # stable order: by feature-set width, then layers, then arch
        rows.sort(key=lambda r: (len(r.get("layers") or r["feature_set"].split(",")),
                                 r["feature_set"], r["arch"]))
        xs = np.arange(len(rows))
        labels = [_config_label(r) for r in rows]
        ms = np.array([r["ex_auc_mean"] for r in rows])
        ss = np.array([r["ex_auc_std"] or 0.0 for r in rows])
        colors = [ARCH_COLORS.get(r["arch"], "tab:gray") for r in rows]
        for x, m, s, c in zip(xs, ms, ss, colors):
            ax.errorbar(x, m, yerr=s, fmt="o", ms=6, color=c, capsize=3)

        lsb = rec.get("linear_single_best")
        if lsb is not None and lsb.get("ex_auc_mean") is not None:
            ax.axhline(lsb["ex_auc_mean"], color="k", ls="--", lw=0.9,
                       label=f"linear single-best ({_config_label(lsb)})")
        ax.axhline(LENGTH_BASELINE, color="gray", ls=":", lw=0.9, label="length base (0.575)")

        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_title(label, fontsize=10)
        if mi == 0:
            ax.set_ylabel("held-out example-AUC")
        ax.grid(alpha=0.25, axis="y")
        ax.set_ylim(0.55, 0.80)
        ax.legend(fontsize=7, loc="lower left")

    fig.suptitle("Richer probes: example-AUC per (feature-set × arch)  "
                 "(α=1, max-pool, mean ± 1 std over 5 splits)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
