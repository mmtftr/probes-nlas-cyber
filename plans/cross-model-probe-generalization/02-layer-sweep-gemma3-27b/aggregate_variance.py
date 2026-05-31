# [ai-generated]
"""Aggregate the repeated-split variance run into mean +/- std per layer.

Reads layer_{NN}.json written by splits_variance.py (each holds a per-seed list)
and produces metrics_variance.json: for every layer the mean/std/n of example-
and token-AUC across the K seeds, plus the trivial baselines computed PER SEED
(so the baseline bands resample with the same splits). No plotting here
(matplotlib isn't in the container) -- plot locally from the JSON.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from src.eval.baselines import all_baselines  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402


def _load_mod(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mean_std(vals):
    arr = np.array([v for v in vals if v == v], dtype=float)  # drop NaN
    if arr.size == 0:
        return None, None, 0
    return float(arr.mean()), float(arr.std(ddof=0)), int(arr.size)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--layer-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="42,43,44,45,46")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    meta = json.loads((Path(args.acts_dir) / "meta.json").read_text())
    here = Path(__file__).resolve().parent
    te_mod = _load_mod(REPO / "src" / "remotes" / "clariden" / "train_eval.py", "clariden_train_eval")
    sv = _load_mod(here / "splits_variance.py", "splits_variance")

    # ---- per-layer mean/std across seeds ----
    layer_dir = Path(args.layer_dir)
    layers = []
    for f in sorted(layer_dir.glob("layer_*.json")):
        rec = json.loads(f.read_text())
        seed_recs = rec.get("seeds")
        if not seed_recs:
            continue
        ex = [s.get("test_ex_auc", float("nan")) for s in seed_recs]
        tok = [s.get("test_tok_auc", float("nan")) for s in seed_recs]
        ex_m, ex_s, ex_n = _mean_std(ex)
        tok_m, tok_s, tok_n = _mean_std(tok)
        layers.append({
            "layer": rec["layer"], "layer_frac": rec["layer_frac"],
            "ex_auc_mean": ex_m, "ex_auc_std": ex_s, "ex_auc_n": ex_n,
            "tok_auc_mean": tok_m, "tok_auc_std": tok_s, "tok_auc_n": tok_n,
            "ex_auc_per_seed": [s.get("test_ex_auc") for s in seed_recs],
            "tok_auc_per_seed": [s.get("test_tok_auc") for s in seed_recs],
        })
    layers.sort(key=lambda d: d["layer"])

    # ---- baselines, recomputed for each seed's held-out split ----
    rows = [json.loads(l) for l in Path(args.dataset).open()]
    eid_to_group = {i: te_mod.pair_group_key(r) for i, r in enumerate(rows)}
    base_per_seed = {bl.name: [] for bl in all_baselines()}
    for seed in seeds:
        _tr, te_eids = sv.make_split_for_seed(eid_to_group, seed)
        test_rows = [rows[i] for i in sorted(te_eids)]
        test_lab = np.array([int(r.get("label", r.get("vulnerable", 0))) for r in test_rows])
        if len(np.unique(test_lab)) < 2:
            continue
        for bl in all_baselines():
            try:
                base_per_seed[bl.name].append(float(roc_auc_score(test_lab, bl.score(test_rows))))
            except Exception:  # noqa: BLE001 -- a baseline must never sink aggregation
                pass
    baseline = {name: {"mean": _mean_std(v)[0], "std": _mean_std(v)[1], "n": _mean_std(v)[2]}
                for name, v in base_per_seed.items()}

    # ---- best layer by mean example-AUC ----
    valid = [d for d in layers if d["ex_auc_mean"] is not None]
    best = max(valid, key=lambda d: d["ex_auc_mean"]) if valid else None

    record = {
        "model": meta["model"], "n_layers": meta["n_layers"], "hidden": meta["hidden"],
        "seeds": seeds, "n_seeds": len(seeds),
        "layers_done": len(layers), "layers_total": meta["n_layers"],
        "best_layer": best["layer"] if best else None,
        "best_layer_frac": best["layer_frac"] if best else None,
        "best_ex_auc_mean": best["ex_auc_mean"] if best else None,
        "best_ex_auc_std": best["ex_auc_std"] if best else None,
        "baseline_auc": baseline,
        "layers": layers,
    }
    Path(args.out).write_text(json.dumps(record, indent=2))
    if best:
        print(f"[aggregate-var] {len(layers)}/{meta['n_layers']} layers, {len(seeds)} seeds; "
              f"best layer {best['layer']} (frac {best['layer_frac']:.3f}) "
              f"ex_auc={best['ex_auc_mean']:.3f}+/-{best['ex_auc_std']:.3f}; "
              f"baselines={ {k: round(v['mean'], 3) if v['mean'] else None for k, v in baseline.items()} }",
              file=sys.stderr)


if __name__ == "__main__":
    main()
