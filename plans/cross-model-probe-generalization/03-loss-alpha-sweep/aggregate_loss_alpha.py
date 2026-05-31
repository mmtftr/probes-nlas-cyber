# [ai-generated]
"""Combine loss_alpha_sweep cell JSONs -> mean/std over seeds per (layer, loss, alpha).

Writes metrics_loss_alpha.json: a flat list of {layer, loss, alpha, ex_auc_mean,
ex_auc_std, tok_auc_mean, tok_auc_std, n, per_seed}. Also reports the best
(loss, alpha) per layer by mean example-AUC and the base-vs-neg_incl delta.
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def _ms(vals):
    a = np.array([v for v in vals if v is not None and v == v], dtype=float)
    return (float(a.mean()), float(a.std(ddof=0)), int(a.size)) if a.size else (None, None, 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="")
    args = ap.parse_args()

    cells = [json.loads(f.read_text()) for f in Path(args.cell_dir).glob("cell_*.json")]
    by_key = defaultdict(list)
    for c in cells:
        by_key[(c["layer"], c["loss"], c["alpha"])].append(c)

    rows = []
    for (li, loss, alpha), cs in sorted(by_key.items()):
        ex_m, ex_s, n = _ms([c.get("test_ex_auc") for c in cs])
        tok_m, tok_s, _ = _ms([c.get("test_tok_auc") for c in cs])
        rows.append({
            "layer": li, "loss": loss, "alpha": alpha,
            "ex_auc_mean": ex_m, "ex_auc_std": ex_s, "n": n,
            "tok_auc_mean": tok_m, "tok_auc_std": tok_s,
            "ex_auc_per_seed": [(c.get("seed"), c.get("test_ex_auc")) for c in sorted(cs, key=lambda c: c.get("seed", 0))],
        })

    # best (loss, alpha) per layer + base-vs-neg_incl delta at matched alpha.
    layers = sorted({r["layer"] for r in rows})
    summary = []
    for li in layers:
        lr = [r for r in rows if r["layer"] == li and r["ex_auc_mean"] is not None]
        if not lr:
            continue
        best = max(lr, key=lambda r: r["ex_auc_mean"])
        # neg_incl improvement averaged over shared alphas
        deltas = []
        for alpha in sorted({r["alpha"] for r in lr}):
            b = next((r for r in lr if r["loss"] == "base" and r["alpha"] == alpha), None)
            n = next((r for r in lr if r["loss"] == "neg_incl" and r["alpha"] == alpha), None)
            if b and n:
                deltas.append(n["ex_auc_mean"] - b["ex_auc_mean"])
        summary.append({
            "layer": li,
            "best_loss": best["loss"], "best_alpha": best["alpha"],
            "best_ex_auc_mean": best["ex_auc_mean"], "best_ex_auc_std": best["ex_auc_std"],
            "mean_neg_incl_delta": float(np.mean(deltas)) if deltas else None,
        })

    record = {"model": args.model, "n_cells": len(cells), "rows": rows, "per_layer_summary": summary}
    Path(args.out).write_text(json.dumps(record, indent=2))
    for s in summary:
        d = s["mean_neg_incl_delta"]
        print(f"[agg] L{s['layer']:02d}: best={s['best_loss']} a={s['best_alpha']:g} "
              f"ex={s['best_ex_auc_mean']:.3f}±{s['best_ex_auc_std']:.3f}  "
              f"mean neg_incl-base delta={d:+.3f}" if d is not None else
              f"[agg] L{s['layer']:02d}: best={s['best_loss']} a={s['best_alpha']:g}", file=sys.stderr)


if __name__ == "__main__":
    main()
