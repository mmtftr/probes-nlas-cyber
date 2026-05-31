# [ai-generated]
"""Sweep the span-max loss over {variant x alpha x layer x seed}.

Reuses the cached per-layer activation memmaps from experiment 02
(runs/layersweep_<slug>/acts) and retrains the linear probe for every cell of:

    loss variant in {base, neg_incl}      # span_max_loss vs span_max_loss_neg_incl
    alpha        in --alphas              # in-span up-weight of the per-token BCE
    layer        in --layers              # the best layers found in exp 02
    seed         in --seeds               # group-clean split seeds (variance)

Per cell: held-out example- + token-AUC. One JSON per cell, resumable. The grid
is GPU-sharded by global cell index (--n-gpus/--gpu-id); each worker sorts its
cells by layer so the in-RAM layer cache (size 1) is reused across cells.

Only the OUTER test split is varied across seeds (same convention as exp 02's
splits_variance.py): the internal val seed inside train_one_layer stays at 7.
seed=42 reproduces the canonical split, so the (base, alpha=10, seed=42) cell
must equal exp-02's single-split value at that layer (built-in sanity tie-in).
"""
from __future__ import annotations
import argparse
import importlib.util
import itertools
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
    p = REPO / "src" / "remotes" / "the cluster" / "train_eval.py"
    spec = importlib.util.spec_from_file_location("remote_train_eval", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_split_for_seed(eid_to_group, seed, frac_heldout=0.2):
    """Group-clean held-out split for a seed (mirrors load_or_make_split exactly)."""
    groups = sorted(set(eid_to_group.values()))
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n_held = max(1, int(round(frac_heldout * len(groups))))
    heldout = set(groups[:n_held])
    train_eids = {e for e, g in eid_to_group.items() if g not in heldout}
    return train_eids, {e for e, g in eid_to_group.items() if g in heldout}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--offsets", default=None,
                    help="Per-row char offsets npz (offsets_row_NNNN keys) for the "
                         "honest live-code token AUC. Defaults to <acts-dir>/offsets.npz.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", required=True, help="comma-separated layer indices")
    ap.add_argument("--alphas", default="1,5,10,20,50")
    ap.add_argument("--losses", default="base,neg_incl")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--n-gpus", type=int, default=1)
    ap.add_argument("--gpu-id", type=int, default=0)
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    losses = [x.strip() for x in args.losses.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    te_mod = _load_train_eval()
    acts = Path(args.acts_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    y = np.load(acts / "y.npy")
    eids = np.load(acts / "example_ids.npy")

    # Load offsets + dataset rows once for the honest live-code token AUC.
    offsets_path = Path(args.offsets) if args.offsets else (acts / "offsets.npz")
    offsets_by_eid = load_offsets_npz(offsets_path)
    dataset_rows_by_eid = load_dataset_rows(Path(args.dataset))

    rows = [json.loads(l) for l in Path(args.dataset).open()]
    eid_to_group = {i: te_mod.pair_group_key(r) for i, r in enumerate(rows)}
    masks = {s: tuple(np.fromiter((int(e) in S for e in eids), bool, len(eids))
                      for S in (make_split_for_seed(eid_to_group, s)[0],))[0]
             for s in seeds}
    masks = {s: (m, ~m) for s, m in masks.items()}  # (train_mask, test_mask)

    # Full cell grid; shard by global index; sort my cells by layer for cache reuse.
    grid = list(itertools.product(layers, losses, alphas, seeds))
    mine = [c for i, c in enumerate(grid) if i % args.n_gpus == args.gpu_id]
    mine.sort(key=lambda c: c[0])  # group by layer
    print(f"[loss-sweep] gpu {args.gpu_id}/{args.n_gpus}: {len(mine)}/{len(grid)} cells",
          file=sys.stderr)

    cur_layer, Xfull = None, None
    for li, loss, alpha, seed in mine:
        tag = f"L{li:02d}_{loss}_a{alpha:g}_s{seed}"
        dst = out / f"cell_{tag}.json"
        if dst.exists():
            continue
        try:
            if li != cur_layer:
                Xfull = np.asarray(np.load(acts / f"layer_{li:02d}.npy", mmap_mode="r"),
                                   dtype=np.float32)
                cur_layer = li
                if not np.isfinite(Xfull).all():
                    Xfull = None  # mark bad; cells below will record error
            if Xfull is None:
                dst.write_text(json.dumps({**_meta(li, loss, alpha, seed), "error": "non-finite acts"}))
                continue
            tr, te = masks[seed]
            ytr, etr = y[tr], eids[tr]
            if len(np.unique(ytr)) < 2 or te.sum() == 0:
                dst.write_text(json.dumps({**_meta(li, loss, alpha, seed), "skipped": "degenerate"}))
                continue
            r = train_one_layer(Xfull[tr], ytr, etr, epochs=args.epochs, device=device,
                                verbose=False, alpha=alpha, neg_incl=(loss == "neg_incl"))
            w, b = np.asarray(r["w"], np.float32), float(r["b"])
            if not (np.isfinite(w).all() and np.isfinite(b)):
                dst.write_text(json.dumps({**_meta(li, loss, alpha, seed), "error": "diverged"}))
                continue
            Xte = Xfull[te]
            tok_p = 1.0 / (1.0 + np.exp(-(Xte @ w + b)))
            tok_y, te_eids = y[te], eids[te]
            tok_auc = (float(roc_auc_score(tok_y, tok_p)) if len(np.unique(tok_y)) > 1 else float("nan"))
            # Honest live-code-only token AUC alongside the inflated all-token AUC.
            honest = honest_token_aucs(
                tok_p, tok_y, te_eids, offsets_by_eid, dataset_rows_by_eid,
            )
            ex_ids, ex_p = te_mod.example_scores(tok_p, te_eids)
            ex_y = np.array([int(y[(eids == e)].max() > 0) for e in ex_ids])
            ex_auc = (float(roc_auc_score(ex_y, ex_p)) if len(np.unique(ex_y)) > 1 else float("nan"))
            rec = {**_meta(li, loss, alpha, seed), "test_ex_auc": ex_auc,
                   "test_tok_auc": tok_auc, "val_ex_auc": float(r["ex_auc"]),
                   "tokens_code_auc": honest["tokens_code_auc"],
                   "dropped_fraction": honest["dropped_fraction"]}
            dst.write_text(json.dumps(rec))
            print(f"[loss-sweep] {tag}  ex={ex_auc:.3f} tok={tok_auc:.3f}", file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            dst.write_text(json.dumps({**_meta(li, loss, alpha, seed),
                                       "error": f"{type(e).__name__}: {str(e)[:200]}"}))
            print(f"[loss-sweep] {tag} ERROR {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)


def _meta(li, loss, alpha, seed):
    return {"layer": li, "loss": loss, "alpha": alpha, "seed": seed}


if __name__ == "__main__":
    main()
