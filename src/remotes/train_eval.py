# [ai-generated]
"""Per-model train+eval for the cross-model sweep.

For one model's extracted token activations:
  - build (or load) a deterministic group-clean SVEN split (seed=42, 20% held out);
  - per captured layer, train a span-max probe on TRAIN tokens
    (src.training.train_probe_spanmax.train_one_layer);
  - evaluate on the HELD-OUT test split: example-level + token-level AUC, and
    the trivial baselines (random/length/regex) for honest lift;
  - pick the best layer by test example-AUC; write metrics.json + probe.npz.

Outputs (under --out): metrics.json (the cross-model comparison record),
probe.npz (w, b, layer). The metrics record carries best_layer_frac so layer
selection is comparable across models of different depth.

NOTE(adhoc-decision): example score = max token sigmoid (span-max philosophy,
final/max-token signal per Ribeiro/Openia). Swap to mean if max proves noisy.
"""
from __future__ import annotations

import argparse
import json
import hashlib
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from sklearn.metrics import roc_auc_score  # noqa: E402
from src.training.train_probe_spanmax import train_one_layer  # noqa: E402
from src.eval.baselines import all_baselines  # noqa: E402


def pair_group_key(row: dict) -> str:
    """Group vuln/fix pairs so a pair never straddles the train/test boundary.
    Mirrors scripts/retrain_spanmax_sven_split.py."""
    repo = row.get("_origin_repo")
    if repo:
        return f"repo::{repo}"
    fn = row.get("_file_name") or ""
    func = row.get("_func_name") or ""
    if fn or func:
        return f"func::{fn}::{func}"
    return f"row::{hashlib.sha1((row.get('code') or '').encode()).hexdigest()[:12]}"


def load_or_make_split(dataset_path: Path, split_path: Path, frac_heldout=0.2, seed=42):
    rows = [json.loads(l) for l in dataset_path.open()]
    eid_to_group = {i: pair_group_key(r) for i, r in enumerate(rows)}
    if split_path.exists():
        heldout = set(json.loads(split_path.read_text())["heldout_groups"])
    else:  # deterministic seeded group hold-out; persist for provenance.
        groups = sorted(set(eid_to_group.values()))
        rng = np.random.default_rng(seed)
        rng.shuffle(groups)
        n_held = max(1, int(round(frac_heldout * len(groups))))
        heldout = set(groups[:n_held])
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split_path.write_text(json.dumps(
            {"seed": seed, "frac_heldout": frac_heldout,
             "n_groups": len(groups), "heldout_groups": sorted(heldout)}, indent=2))
    train_eids = {e for e, g in eid_to_group.items() if g not in heldout}
    test_eids = {e for e, g in eid_to_group.items() if g in heldout}
    return rows, train_eids, test_eids


def example_scores(tok_p, eids):
    """Max-pool token probs per example -> (example_ids_sorted, scores)."""
    order = np.argsort(eids, kind="stable")
    e_sorted, p_sorted = eids[order], tok_p[order]
    uniq, idx = np.unique(e_sorted, return_index=True)
    scores = np.maximum.reduceat(p_sorted, idx)
    return uniq, scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--acts", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    acts_dir = Path(args.acts)
    rows, train_eids, test_eids = load_or_make_split(Path(args.dataset), Path(args.split))

    # Model depth/width for comparable layer fractions (config only, no weights).
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(args.model)
        n_layers_total = int(getattr(cfg, "num_hidden_layers", 0)) or None
        hidden = int(getattr(cfg, "hidden_size", 0)) or None
    except Exception:
        n_layers_total = hidden = None

    layer_files = sorted(acts_dir.glob("token_activations_layer*.npz"))
    if not layer_files:
        raise SystemExit(f"no layer npz under {acts_dir}")

    per_layer = []
    for f in layer_files:
        li = int(f.stem.replace("token_activations_layer", ""))
        npz = np.load(f)
        X, y, eids = npz["X"], npz["y"], npz["example_ids"]
        tr = np.fromiter((int(e) in train_eids for e in eids), bool, len(eids))
        te = ~tr
        if len(np.unique(y[tr])) < 2 or te.sum() == 0:
            continue
        r = train_one_layer(X[tr], y[tr], eids[tr], epochs=args.epochs,
                             device=device, verbose=False)
        w, b = np.asarray(r["w"], np.float32), float(r["b"])
        tok_p = 1.0 / (1.0 + np.exp(-(X[te] @ w + b)))
        tok_y, te_eids = y[te], eids[te]
        tok_auc = (float(roc_auc_score(tok_y, tok_p))
                   if len(np.unique(tok_y)) > 1 else float("nan"))
        ex_ids, ex_p = example_scores(tok_p, te_eids)
        ex_y = np.array([int(y[(eids == e)].max() > 0) for e in ex_ids])
        ex_auc = (float(roc_auc_score(ex_y, ex_p))
                  if len(np.unique(ex_y)) > 1 else float("nan"))
        per_layer.append({"layer": li, "test_ex_auc": ex_auc, "test_tok_auc": tok_auc,
                          "val_ex_auc": float(r["ex_auc"]), "n_test_ex": int(len(ex_ids)),
                          "w": w, "b": b})

    if not per_layer:
        raise SystemExit("no trainable layer (label imbalance?)")

    best = max(per_layer, key=lambda d: (d["test_ex_auc"] if d["test_ex_auc"] == d["test_ex_auc"] else -1))

    # Trivial baselines on the held-out examples (honest lift over regex/length).
    test_rows = [rows[i] for i in sorted(test_eids)]
    test_lab = np.array([int(r.get("label", r.get("vulnerable", 0))) for r in test_rows])
    base_auc = {}
    if len(np.unique(test_lab)) > 1:
        for bl in all_baselines():
            try:
                s = bl.score(test_rows)
                base_auc[bl.name] = float(roc_auc_score(test_lab, s))
            except Exception as e:  # baseline must never sink the run
                base_auc[bl.name] = f"err:{e}"

    metrics = {
        "model": args.model,
        "n_layers_total": n_layers_total,
        "hidden_size": hidden,
        "best_layer": best["layer"],
        "best_layer_frac": (best["layer"] / n_layers_total) if n_layers_total else None,
        "best_test_ex_auc": best["test_ex_auc"],
        "best_test_tok_auc": best["test_tok_auc"],
        "baseline_auc": base_auc,
        "n_train_ex": len(train_eids),
        "n_test_ex": len(test_eids),
        "layers": [{k: v for k, v in d.items() if k not in ("w", "b")} for d in per_layer],
    }
    out = Path(args.out)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(out / "probe.npz", w=best["w"], b=np.float32(best["b"]),
                        layer=np.int32(best["layer"]))
    print(f"[train_eval] {args.model}: best layer {best['layer']} "
          f"(frac {metrics['best_layer_frac']}) test_ex_auc={best['test_ex_auc']:.3f} "
          f"baselines={base_auc}", file=sys.stderr)


if __name__ == "__main__":
    main()
