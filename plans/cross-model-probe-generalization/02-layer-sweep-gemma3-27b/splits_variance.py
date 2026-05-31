# [ai-generated]
"""Repeated-split variance for the per-layer AUC curve.

Reuses the cached per-layer activation memmaps (they are split-independent) and
retrains the span-max probe under K group-clean SVEN splits (different seeds).
Records per-(layer, seed) example/token AUC so the AUC-vs-layer curve can be
drawn with mean +/- std error bands.

Design decisions (see EXPERIMENT.md "Variance" section):
  - Only the OUTER held-out group split is varied. That is exactly what the user
    asked about ("multiple splits of the dataset"). The internal 90/10
    epoch-selection split inside train_one_layer stays at its fixed seed=7 -- it
    is part of the training recipe, not a dataset split; varying it would
    conflate procedure noise with split noise.
  - Group-clean is preserved for every seed: the shuffle is at the pair-group
    level (pair_group_key), mirroring load_or_make_split's seeded branch exactly.
    seed=42 reproduces the canonical persisted split bit-for-bit, so its curve
    must equal the single-split layer sweep (built-in sanity tie-in).
  - frac_heldout=0.2 fixed (same as the canonical split).

Resumable + GPU-shardable: one layer_{NN}.json per layer holding the list of
per-seed results; skip any that exist. With --n-gpus G --gpu-id i a worker
handles layers where li % G == i. Baselines are NOT computed here (the
aggregator does them, once per seed).
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
from sklearn.metrics import roc_auc_score  # noqa: E402


def _load_train_eval():
    """Import the canonical train_eval module by path (no __init__ in its dir)."""
    p = REPO / "src" / "remotes" / "the cluster" / "train_eval.py"
    spec = importlib.util.spec_from_file_location("remote_train_eval", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_split_for_seed(eid_to_group: dict[int, str], seed: int, frac_heldout: float = 0.2):
    """Group-clean held-out split for an arbitrary seed.

    Mirrors load_or_make_split's seeded else-branch EXACTLY (same default_rng,
    same group-level shuffle, same n_held rounding) so seed=42 reproduces the
    persisted canonical split. Returns (train_eids:set, test_eids:set).
    """
    groups = sorted(set(eid_to_group.values()))
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n_held = max(1, int(round(frac_heldout * len(groups))))
    heldout = set(groups[:n_held])
    train_eids = {e for e, g in eid_to_group.items() if g not in heldout}
    test_eids = {e for e, g in eid_to_group.items() if g in heldout}
    return train_eids, test_eids


def _eval_split(Xfull, y, eids, tr, te, te_mod, epochs, device):
    """Train one probe on tr, eval ex/tok AUC on te. Returns dict or {error|skipped}."""
    ytr, etr = y[tr], eids[tr]
    if len(np.unique(ytr)) < 2 or te.sum() == 0:
        return {"skipped": "degenerate labels"}
    Xtr = Xfull[tr]
    r = train_one_layer(Xtr, ytr, etr, epochs=epochs, device=device, verbose=False)
    w, b = np.asarray(r["w"], np.float32), float(r["b"])
    if not (np.isfinite(w).all() and np.isfinite(b)):
        return {"error": "diverged (NaN probe weights)"}
    Xte = Xfull[te]
    tok_p = 1.0 / (1.0 + np.exp(-(Xte @ w + b)))
    tok_y, te_eids = y[te], eids[te]
    tok_auc = (float(roc_auc_score(tok_y, tok_p))
               if len(np.unique(tok_y)) > 1 else float("nan"))
    ex_ids, ex_p = te_mod.example_scores(tok_p, te_eids)
    ex_y = np.array([int(y[(eids == e)].max() > 0) for e in ex_ids])
    ex_auc = (float(roc_auc_score(ex_y, ex_p))
              if len(np.unique(ex_y)) > 1 else float("nan"))
    return {"test_ex_auc": ex_auc, "test_tok_auc": tok_auc,
            "val_ex_auc": float(r["ex_auc"]), "n_test_ex": int(len(ex_ids)),
            "n_train_ex": int(len(np.unique(etr)))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="42,43,44,45,46",
                    help="comma-separated outer-split seeds (42 == canonical)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--n-gpus", type=int, default=1)
    ap.add_argument("--gpu-id", type=int, default=0)
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    te_mod = _load_train_eval()
    acts = Path(args.acts_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    meta = json.loads((acts / "meta.json").read_text())
    n_layers = meta["n_layers"]
    y = np.load(acts / "y.npy")
    eids = np.load(acts / "example_ids.npy")

    # Group key per example id, from the dataset rows (eid == row index).
    rows = [json.loads(l) for l in Path(args.dataset).open()]
    eid_to_group = {i: te_mod.pair_group_key(r) for i, r in enumerate(rows)}

    # Pre-compute the train/test boolean masks once per seed (over token rows).
    masks = {}
    for seed in seeds:
        tr_eids, _te_eids = make_split_for_seed(eid_to_group, seed)
        tr = np.fromiter((int(e) in tr_eids for e in eids), bool, len(eids))
        masks[seed] = (tr, ~tr)

    my_layers = [li for li in range(n_layers) if li % args.n_gpus == args.gpu_id]
    print(f"[variance] gpu {args.gpu_id}/{args.n_gpus} handling {len(my_layers)} layers "
          f"x {len(seeds)} seeds", file=sys.stderr)

    for li in my_layers:
        dst = out / f"layer_{li:02d}.json"
        if dst.exists():
            continue
        try:
            # Load the whole layer into RAM once, then slice per seed (avoids
            # re-reading the memmap K times from disk).
            Xfull = np.asarray(np.load(acts / f"layer_{li:02d}.npy", mmap_mode="r"),
                               dtype=np.float32)
            if not np.isfinite(Xfull).all():
                dst.write_text(json.dumps({"layer": li, "error": "non-finite activations"}))
                print(f"[variance] layer {li:02d} SKIP non-finite", file=sys.stderr)
                continue
            per_seed = []
            for seed in seeds:
                tr, te = masks[seed]
                rec = _eval_split(Xfull, y, eids, tr, te, te_mod, args.epochs, device)
                rec["seed"] = seed
                per_seed.append(rec)
            del Xfull
            aucs = [s["test_ex_auc"] for s in per_seed if "test_ex_auc" in s]
            rec = {"layer": li, "layer_frac": li / (n_layers - 1), "seeds": per_seed}
            dst.write_text(json.dumps(rec))
            if aucs:
                print(f"[variance] layer {li:02d}  ex_auc mean={np.mean(aucs):.3f} "
                      f"std={np.std(aucs):.3f} (n={len(aucs)})", file=sys.stderr)
        except Exception as e:  # noqa: BLE001 -- record + continue, never abort the sweep
            dst.write_text(json.dumps({"layer": li, "error": f"{type(e).__name__}: {str(e)[:200]}"}))
            print(f"[variance] layer {li:02d} ERROR {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)


if __name__ == "__main__":
    main()
