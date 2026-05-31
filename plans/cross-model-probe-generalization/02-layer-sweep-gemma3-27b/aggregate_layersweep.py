# [ai-generated]
"""Combine per-layer JSONs into one AUC-vs-depth record + the shared baselines.

Reads layer_{NN}.json written by train_all_layers.py, computes the trivial
baselines once on the held-out split, and writes metrics_layersweep.json. No
plotting here (matplotlib isn't in the container) — plot locally from the JSON.
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


def _load_train_eval():
    p = REPO / "src" / "remotes" / "the cluster" / "train_eval.py"
    spec = importlib.util.spec_from_file_location("remote_train_eval", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--layer-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    meta = json.loads((Path(args.acts_dir) / "meta.json").read_text())
    layer_dir = Path(args.layer_dir)
    layers = []
    for f in sorted(layer_dir.glob("layer_*.json")):
        rec = json.loads(f.read_text())
        if "test_ex_auc" in rec:
            layers.append(rec)
    layers.sort(key=lambda d: d["layer"])
    done = len(layers)

    te_mod = _load_train_eval()
    rows, _train_eids, test_eids = te_mod.load_or_make_split(Path(args.dataset), Path(args.split))
    test_rows = [rows[i] for i in sorted(test_eids)]
    test_lab = np.array([int(r.get("label", r.get("vulnerable", 0))) for r in test_rows])
    base_auc = {}
    if len(np.unique(test_lab)) > 1:
        for bl in all_baselines():
            try:
                base_auc[bl.name] = float(roc_auc_score(test_lab, bl.score(test_rows)))
            except Exception as e:  # noqa: BLE001
                base_auc[bl.name] = f"err:{e}"

    valid = [d for d in layers if d["test_ex_auc"] == d["test_ex_auc"]]  # drop NaN
    best = max(valid, key=lambda d: d["test_ex_auc"]) if valid else None
    record = {
        "model": meta["model"],
        "n_layers": meta["n_layers"],
        "hidden": meta["hidden"],
        "layers_done": done,
        "layers_total": meta["n_layers"],
        "best_layer": best["layer"] if best else None,
        "best_layer_frac": best["layer_frac"] if best else None,
        "best_test_ex_auc": best["test_ex_auc"] if best else None,
        "best_test_tok_auc": best["test_tok_auc"] if best else None,
        "baseline_auc": base_auc,
        "layers": layers,
    }
    Path(args.out).write_text(json.dumps(record, indent=2))
    print(f"[aggregate] {done}/{meta['n_layers']} layers; "
          f"best layer {record['best_layer']} (frac {record['best_layer_frac']}) "
          f"ex_auc={record['best_test_ex_auc']}; baselines={base_auc}", file=sys.stderr)


if __name__ == "__main__":
    main()
