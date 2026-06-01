# [ai-generated]
"""Combine 06 per-layer JSONs into one AUC-vs-depth record + baselines.

Like exp-02's aggregator, but the HONEST metric (`tokens_code_auc`) is the
headline, and layer selection avoids test leakage:

  best_layer        = argmax over layers of VALIDATION ex-AUC (`val_ex_auc`),
                      the same val signal exp-02 used for selection. No test
                      metric touches selection.
  best_tokens_code_auc / best_tokens_auc
                    = the honest + inflated TEST token AUCs read off at the
                      val-selected layer. `best_tokens_code_auc` is the
                      deployable number.
  oracle_*          = argmax over layers of TEST `tokens_code_auc`. This is an
                      UPPER BOUND (peeks at test) — reported ONLY to measure the
                      val-vs-oracle gap (the §8 layer-policy question), never for
                      deployment.

Baselines (random/length/regex) are example-level, as in exp-02 — kept for
continuity; they are NOT directly comparable to the token-level honest AUC, so
the headline comparison is `tokens_code_auc` vs `tokens_auc` (the inflation gap).

No plotting here (matplotlib isn't in the container) — plot locally from JSON.
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
    p = REPO / "src" / "remotes" / "clariden" / "train_eval.py"
    spec = importlib.util.spec_from_file_location("clariden_train_eval", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _finite(x) -> bool:
    try:
        return x == x and abs(float(x)) != float("inf")
    except (TypeError, ValueError):
        return False


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
        if "test_ex_auc" in rec:  # skip skipped/errored layers
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

    # --- selection: by VALIDATION tokens_code AUC (no test leakage) ---
    # (val_ex_auc was near-chance on this data and selected near-random layers;
    #  the honest token signal is the right selection target.)
    val_valid = [d for d in layers if _finite(d.get("val_tokens_code_auc"))]
    best = max(val_valid, key=lambda d: d["val_tokens_code_auc"]) if val_valid else None

    # --- oracle: argmax TEST tokens_code_auc (upper bound, not for deployment) ---
    tc_valid = [d for d in layers if _finite(d.get("tokens_code_auc"))]
    oracle = max(tc_valid, key=lambda d: d["tokens_code_auc"]) if tc_valid else None

    record = {
        "model": meta["model"],
        "n_layers": meta["n_layers"],
        "hidden": meta["hidden"],
        "layers_done": done,
        "layers_total": meta["n_layers"],
        # val-selected (deployable) operating point
        "best_layer": best["layer"] if best else None,
        "best_layer_frac": best["layer_frac"] if best else None,
        "selected_by": "val_tokens_code_auc",
        "best_val_tokens_code_auc": best.get("val_tokens_code_auc") if best else None,
        "best_val_ex_auc": best.get("val_ex_auc") if best else None,
        "best_tokens_code_auc": best.get("tokens_code_auc") if best else None,
        "best_tokens_auc": best.get("tokens_auc", best.get("test_tok_auc")) if best else None,
        "best_test_ex_auc": best.get("test_ex_auc") if best else None,  # rides along
        "best_dropped_fraction": best.get("dropped_fraction") if best else None,
        # oracle (test-peeking upper bound) — for the val-vs-oracle gap only
        "oracle_tokens_code_layer": oracle["layer"] if oracle else None,
        "oracle_tokens_code_frac": oracle["layer_frac"] if oracle else None,
        "oracle_tokens_code_auc": oracle.get("tokens_code_auc") if oracle else None,
        "baseline_auc": base_auc,
        "baseline_note": "baselines are example-level (exp-02 carryover); compare "
                         "tokens_code_auc vs tokens_auc for the inflation gap.",
        "layers": layers,
    }
    Path(args.out).write_text(json.dumps(record, indent=2))
    print(f"[aggregate] {done}/{meta['n_layers']} layers; "
          f"val-best layer {record['best_layer']} (frac {record['best_layer_frac']}) "
          f"tokens_code_auc={record['best_tokens_code_auc']} "
          f"tokens_auc={record['best_tokens_auc']}; "
          f"oracle tokens_code layer {record['oracle_tokens_code_layer']} "
          f"auc={record['oracle_tokens_code_auc']}; baselines={base_auc}", file=sys.stderr)


if __name__ == "__main__":
    main()
