# [ai-generated]
"""Combine richer_probe_sweep cell JSONs -> mean/std over seeds per (feature_set, arch).

Writes metrics_richer.json: a list of {feature_set, arch, in_dim, ex_auc_mean,
ex_auc_std, n, tok_auc_mean, tok_auc_std, ex_auc_per_seed}, the overall best
config by mean example-AUC, and `linear_single_best` -- the (single-layer,
linear) config with the highest mean example-AUC (the exp-03 baseline this
experiment must beat to justify a richer probe).

Also prints a stderr summary table: per-config ex-AUC mean±std and each config's
delta vs linear_single_best.
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


def _is_single_linear(row) -> bool:
    return row["arch"] == "linear" and len(row.get("layers") or row["feature_set"].split(",")) == 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="")
    args = ap.parse_args()

    cells = [json.loads(f.read_text()) for f in Path(args.cell_dir).glob("cell_*.json")]
    by_key = defaultdict(list)
    for c in cells:
        by_key[(c["feature_set"], c["arch"])].append(c)

    rows = []
    for (fs, arch), cs in sorted(by_key.items()):
        ex_m, ex_s, n = _ms([c.get("test_ex_auc") for c in cs])
        tok_m, tok_s, _ = _ms([c.get("test_tok_auc") for c in cs])
        # Honest live-code-only token AUC (present only on runs that recorded it).
        code_m, code_s, _ = _ms([c.get("tokens_code_auc") for c in cs])
        in_dim = next((c.get("in_dim") for c in cs if c.get("in_dim") is not None), None)
        layers = next((c.get("layers") for c in cs if c.get("layers") is not None),
                      [int(x) for x in fs.split(",") if x.strip()])
        rows.append({
            "feature_set": fs, "arch": arch, "layers": layers, "in_dim": in_dim,
            "ex_auc_mean": ex_m, "ex_auc_std": ex_s, "n": n,
            "tok_auc_mean": tok_m, "tok_auc_std": tok_s,
            "tokens_code_auc_mean": code_m, "tokens_code_auc_std": code_s,
            "ex_auc_per_seed": [(c.get("seed"), c.get("test_ex_auc"))
                                for c in sorted(cs, key=lambda c: c.get("seed", 0))],
        })

    scored = [r for r in rows if r["ex_auc_mean"] is not None]
    best = max(scored, key=lambda r: r["ex_auc_mean"]) if scored else None
    single_linear = [r for r in scored if _is_single_linear(r)]
    linear_single_best = (max(single_linear, key=lambda r: r["ex_auc_mean"])
                          if single_linear else None)

    record = {"model": args.model, "n_cells": len(cells), "rows": rows,
              "best": best, "linear_single_best": linear_single_best}
    Path(args.out).write_text(json.dumps(record, indent=2))

    base_m = linear_single_best["ex_auc_mean"] if linear_single_best else None
    if linear_single_best is not None:
        print(f"[agg] linear_single_best = {linear_single_best['feature_set']}/linear "
              f"ex={base_m:.3f}±{linear_single_best['ex_auc_std']:.3f}", file=sys.stderr)
    for r in sorted(scored, key=lambda r: -r["ex_auc_mean"]):
        delta = (r["ex_auc_mean"] - base_m) if base_m is not None else None
        ds = f"  Δ_vs_linbest={delta:+.3f}" if delta is not None else ""
        print(f"[agg] {r['feature_set']}/{r['arch']} (d={r['in_dim']}): "
              f"ex={r['ex_auc_mean']:.3f}±{r['ex_auc_std']:.3f} "
              f"tok={r['tok_auc_mean']:.3f}±{r['tok_auc_std']:.3f} n={r['n']}{ds}", file=sys.stderr)
    if best is not None:
        print(f"[agg] best = {best['feature_set']}/{best['arch']} "
              f"ex={best['ex_auc_mean']:.3f}±{best['ex_auc_std']:.3f}", file=sys.stderr)


if __name__ == "__main__":
    main()
