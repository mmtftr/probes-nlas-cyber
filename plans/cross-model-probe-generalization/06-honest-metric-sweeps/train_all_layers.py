# [ai-generated]
"""Honest per-layer span-max sweep: like exp-02's train_all_layers.py but ALSO
records the honest `tokens_code_auc` (live-code-only token AUC) alongside the
inflated all-token `tokens_auc` for the TEST tokens.

Why a separate copy: exp-02's files are the archived inflated-metric record and
must not change. This 06 copy keeps the exact same train/split/GPU-sharding /
resumable logic, and augments each per-layer JSON with three honest fields:
    tokens_auc        all-token TEST AUC (== exp-02's test_tok_auc, the inflated
                      reference). Also written as `test_tok_auc` for backward
                      compatibility with the aggregator.
    tokens_code_auc   TEST AUC restricted to live-code tokens (tree-sitter mask
                      via src/eval/honest_scoring.py).
    dropped_fraction  1 - live_code_mask.mean() over the TEST tokens.

The mask needs per-row char offsets (offsets.npz in the acts dir) + the dataset
rows (code + lang). See src/eval/honest_scoring.py and src/eval/code_mask.py.

Resumable + GPU-shardable: writes one layer_{NN}.json per layer; skips any that
exist. With --n-gpus G --gpu-id i, a worker handles layers where li % G == i.
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

from src.training.train_probe_spanmax import train_one_layer  # noqa: E402
from src.eval.honest_scoring import (  # noqa: E402
    honest_token_aucs,
    load_dataset_rows,
    load_offsets_npz,
)
from sklearn.metrics import roc_auc_score  # noqa: E402


def _load_train_eval():
    """Import the canonical train_eval module by path (no __init__ in its dir)."""
    p = REPO / "src" / "remotes" / "clariden" / "train_eval.py"
    spec = importlib.util.spec_from_file_location("clariden_train_eval", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--offsets", default=None,
                    help="Per-row char offsets npz (offsets_row_NNNN keys). "
                         "Defaults to <acts-dir>/offsets.npz.")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--n-gpus", type=int, default=1)
    ap.add_argument("--gpu-id", type=int, default=0)
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    te_mod = _load_train_eval()
    acts = Path(args.acts_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    offsets_path = Path(args.offsets) if args.offsets else (acts / "offsets.npz")
    offsets_by_eid = load_offsets_npz(offsets_path)
    dataset_rows_by_eid = load_dataset_rows(Path(args.dataset))

    meta = json.loads((acts / "meta.json").read_text())
    n_layers = meta["n_layers"]
    y = np.load(acts / "y.npy")
    eids = np.load(acts / "example_ids.npy")

    _rows, train_eids, test_eids = te_mod.load_or_make_split(
        Path(args.dataset), Path(args.split)
    )
    tr = np.fromiter((int(e) in train_eids for e in eids), bool, len(eids))
    te = ~tr

    my_layers = [li for li in range(n_layers) if li % args.n_gpus == args.gpu_id]
    print(f"[train-all] gpu {args.gpu_id}/{args.n_gpus} handling {len(my_layers)} layers", file=sys.stderr)

    for li in my_layers:
        dst = out / f"layer_{li:02d}.json"
        if dst.exists():
            continue
        # Isolate per-layer failures: a single diverging layer (NaN weights,
        # non-finite activations) must not kill the worker and abort the rest.
        try:
            Xmm = np.load(acts / f"layer_{li:02d}.npy", mmap_mode="r")
            Xtr = np.asarray(Xmm[tr], dtype=np.float32)
            ytr, etr = y[tr], eids[tr]
            if len(np.unique(ytr)) < 2 or te.sum() == 0:
                dst.write_text(json.dumps({"layer": li, "skipped": "degenerate labels"}))
                continue
            if not np.isfinite(Xtr).all():
                dst.write_text(json.dumps({"layer": li, "error": "non-finite activations"}))
                print(f"[train-all] layer {li:02d} SKIP non-finite activations", file=sys.stderr)
                continue
            r = train_one_layer(Xtr, ytr, etr, epochs=args.epochs, device=device, verbose=False)
            w, b = np.asarray(r["w"], np.float32), float(r["b"])
            if not (np.isfinite(w).all() and np.isfinite(b)):
                dst.write_text(json.dumps({"layer": li, "error": "diverged (NaN probe weights)"}))
                print(f"[train-all] layer {li:02d} SKIP diverged", file=sys.stderr)
                continue

            Xte = np.asarray(Xmm[te], dtype=np.float32)
            tok_p = 1.0 / (1.0 + np.exp(-(Xte @ w + b)))
            tok_y, te_eids = y[te], eids[te]
            tok_auc = (float(roc_auc_score(tok_y, tok_p))
                       if len(np.unique(tok_y)) > 1 else float("nan"))
            # Honest contrast over the TEST tokens.
            honest = honest_token_aucs(
                tok_p, tok_y, te_eids, offsets_by_eid, dataset_rows_by_eid,
            )
            ex_ids, ex_p = te_mod.example_scores(tok_p, te_eids)
            ex_y = np.array([int(y[(eids == e)].max() > 0) for e in ex_ids])
            ex_auc = (float(roc_auc_score(ex_y, ex_p))
                      if len(np.unique(ex_y)) > 1 else float("nan"))
            rec = {"layer": li, "layer_frac": li / (n_layers - 1),
                   "test_ex_auc": ex_auc,
                   # `tokens_auc` is the new canonical name; `test_tok_auc` is
                   # kept as a backward-compat alias for the aggregator.
                   "tokens_auc": honest["tokens_auc"],
                   "test_tok_auc": tok_auc,
                   "tokens_code_auc": honest["tokens_code_auc"],
                   "dropped_fraction": honest["dropped_fraction"],
                   "n_pos_code": honest["n_pos_code"],
                   "n_total_code": honest["n_total_code"],
                   "val_ex_auc": float(r["ex_auc"]), "n_test_ex": int(len(ex_ids))}
            dst.write_text(json.dumps(rec))
            print(f"[train-all] layer {li:02d}  ex_auc={ex_auc:.3f} "
                  f"tokens_auc={honest['tokens_auc']:.3f} "
                  f"tokens_code_auc={honest['tokens_code_auc']:.3f} "
                  f"(dropped {honest['dropped_fraction']:.2f})", file=sys.stderr)
        except Exception as e:  # noqa: BLE001 — record + continue, never abort the sweep
            dst.write_text(json.dumps({"layer": li, "error": f"{type(e).__name__}: {str(e)[:200]}"}))
            print(f"[train-all] layer {li:02d} ERROR {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)


if __name__ == "__main__":
    main()
