# [ai-generated]
"""Plot probe-AUC vs. the model's own verbalized P(yes) AUC, per model.

Reads one metrics_verbalized.json per model (label=path pairs). For each model:
  - a grouped bar of probe-AUC vs verbalized-AUC (mean ±1 std over splits), with
    a dotted length-baseline reference line;
  - if seed42_arrays are present, a scatter of probe score vs P(yes) colored by
    true label (the qualitative disagreement view).

Local-only matplotlib:
    uv run --with matplotlib --with numpy \
        python plot_verbalized.py \
        --in "Gemma-3-27B=/path/metrics_verbalized_gemma.json" \
        --in "Qwen2.5-Coder-32B=/path/metrics_verbalized_qwen.json" \
        --out probe_vs_verbalized.png
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inputs", action="append", required=True,
                    help="label=path/to/metrics_verbalized.json (repeatable)")
    ap.add_argument("--out", default="probe_vs_verbalized.png")
    args = ap.parse_args()

    items = []
    for spec in args.inputs:
        label, _, path = spec.partition("=")
        items.append((label, json.loads(Path(path).read_text())))

    n = len(items)
    have_scatter = any(m.get("seed42_arrays") for _, m in items)
    ncols = 2 if have_scatter else 1
    fig, axes = plt.subplots(n, ncols, figsize=(5.5 * ncols, 3.6 * n), squeeze=False)

    for r, (label, m) in enumerate(items):
        base = float(m.get("length_baseline", 0.575))
        # --- left: grouped bar ---
        axb = axes[r][0]
        means = [m["probe_auc_mean"], m["verbalized_auc_mean"]]
        stds = [m["probe_auc_std"], m["verbalized_auc_std"]]
        x = np.arange(2)
        axb.bar(x, means, yerr=stds, capsize=5,
                color=["#2a6f97", "#bc4749"], width=0.6)
        axb.axhline(base, ls=":", color="gray", lw=1.2,
                    label=f"length baseline {base:.3f}")
        axb.set_xticks(x)
        axb.set_xticklabels(["probe", "verbalized\nP(yes)"])
        axb.set_ylim(0.4, 1.0)
        axb.set_ylabel("example-AUC")
        d, ds = m.get("delta_mean"), m.get("delta_std")
        dtxt = f"  Δ={d:+.3f}±{ds:.3f}" if d is not None else ""
        axb.set_title(f"{label}  (L{m['model_layer']}){dtxt}", fontsize=10)
        axb.legend(fontsize=7, loc="lower right")
        for xi, (mu, sd) in enumerate(zip(means, stds)):
            axb.text(xi, mu + sd + 0.01, f"{mu:.3f}", ha="center", fontsize=8)

        # --- right: scatter (probe score vs p_yes), if arrays stored ---
        if have_scatter:
            axs = axes[r][1]
            arr = m.get("seed42_arrays") or {}
            if arr:
                ps = np.array(arr["probe_score"])
                py = np.array(arr["p_yes"])
                lab = np.array(arr["ex_label"])
                for cls, color, name in [(0, "#9aa0a6", "clean"), (1, "#d62728", "vulnerable")]:
                    sel = lab == cls
                    axs.scatter(ps[sel], py[sel], s=14, alpha=0.6, c=color, label=name)
                axs.axhline(0.5, ls=":", color="gray", lw=0.8)
                axs.set_xlabel("probe example score")
                axs.set_ylabel("model P(yes)")
                rho = m.get("spearman")
                rtxt = f"  ρ={rho:.2f}" if rho is not None else ""
                axs.set_title(f"seed-42 test{rtxt}", fontsize=10)
                axs.legend(fontsize=7, loc="best")
            else:
                axs.axis("off")

    fig.suptitle(
        "Probe vs. the model's own verbalized judgment\n"
        "(input-stream, P(yes); mean±std over 5 splits)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
