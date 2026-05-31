# [ai-generated]
"""Sweep richer probe heads over {feature-set x arch x seed}.

Tests whether a richer probe beats the current linear@single-layer probe on
held-out example-AUC. Two richness axes:

    feature-set : one or more layers' activations concatenated PER TOKEN
                  (single layer -> the exp-03 linear baseline; multiple ->
                  Openia-style layer-concat). e.g. "19" or "9,19,26,61".
    arch        : "linear" (LinearProbe, the exp-03 head) or
                  "mlp{H}" (MLPProbe with hidden=H, a non-linear head).

α=1.0 fixed (the exp-03 finding), neg_incl off, MAX-pool example scoring (the
canonical example_scores). Variance over 5 group-clean splits (seeds 42-46).

Reuses the cached per-layer activation memmaps from exp 02
(runs/layersweep_<slug>/acts) -- NO re-extraction, NO model load.

The grid is GPU-sharded by global cell index (--n-gpus/--gpu-id); each worker
sorts its cells by feature-set so the concatenated-feature build (and its
per-(feature_set,seed) cache) is reused across the archs of the same feature
set. seed=42 reproduces the canonical split, so the (single-best-layer, linear,
seed=42) cell must equal exp-03's (base, α=1, seed=42) at that layer (built-in
sanity tie-in).
"""
from __future__ import annotations
import argparse
import importlib.util
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from src.training.train_probe_spanmax import MLPProbe, train_one_layer  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402


def _load_train_eval():
    """Import the canonical train_eval module by path (no __init__ in its dir)."""
    p = REPO / "src" / "remotes" / "clariden" / "train_eval.py"
    spec = importlib.util.spec_from_file_location("clariden_train_eval", p)
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


def _parse_feature_sets(spec: str) -> list[list[int]]:
    """';'-separated groups; each group ','-separated layer ints. -> list of layer lists."""
    out = []
    for group in spec.split(";"):
        layers = [int(x) for x in group.split(",") if x.strip()]
        if layers:
            out.append(layers)
    return out


def _factory_for_arch(arch: str):
    """arch -> probe_factory (or None for linear). 'mlp256','mlp512',... -> MLPProbe(hidden=H)."""
    if arch == "linear":
        return None
    m = re.fullmatch(r"mlp(\d+)", arch)
    if not m:
        raise ValueError(f"unknown arch: {arch}")
    H = int(m.group(1))
    return lambda d, H=H: MLPProbe(d, hidden=H)


def _fslug(layers: list[int]) -> str:
    return "-".join(str(li) for li in layers)


def _meta(layers, arch, seed):
    return {"feature_set": ",".join(str(li) for li in layers), "layers": list(layers),
            "arch": arch, "seed": seed}


def _build_features(acts: Path, layers: list[int], tr: np.ndarray, te: np.ndarray):
    """Bounded-memory concat of per-token activations across `layers`.

    For each layer: mmap-load, slice tr / te rows, append to lists, drop the full
    array. Then hstack the parts (or pass the single array through). Returns
    (Xtr, Xte) float32, or (None, None) if any layer has non-finite activations.
    """
    tr_parts, te_parts = [], []
    for li in layers:
        full = np.load(acts / f"layer_{li:02d}.npy", mmap_mode="r")
        Xtr_l = np.asarray(full[tr], dtype=np.float32)
        Xte_l = np.asarray(full[te], dtype=np.float32)
        del full
        if not (np.isfinite(Xtr_l).all() and np.isfinite(Xte_l).all()):
            return None, None
        tr_parts.append(Xtr_l)
        te_parts.append(Xte_l)
    if len(tr_parts) == 1:
        return tr_parts[0], te_parts[0]
    return np.hstack(tr_parts), np.hstack(te_parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--feature-sets", required=True,
                    help="';'-separated layer groups, each ','-separated, e.g. '19;17,19,22;9,19,26,61'")
    ap.add_argument("--archs", default="linear,mlp256,mlp512")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--n-gpus", type=int, default=1)
    ap.add_argument("--gpu-id", type=int, default=0)
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    feature_sets = _parse_feature_sets(args.feature_sets)
    archs = [a.strip() for a in args.archs.split(",") if a.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    te_mod = _load_train_eval()
    acts = Path(args.acts_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    json.loads((acts / "meta.json").read_text())  # presence check (mirrors exp-02)
    y = np.load(acts / "y.npy")
    eids = np.load(acts / "example_ids.npy")

    # Group key per example id, from the dataset rows (eid == row index).
    rows = [json.loads(l) for l in Path(args.dataset).open()]
    eid_to_group = {i: te_mod.pair_group_key(r) for i, r in enumerate(rows)}

    # Pre-compute (train_mask, test_mask) over token rows once per seed.
    masks = {}
    for seed in seeds:
        tr_eids, _ = make_split_for_seed(eid_to_group, seed)
        tr = np.fromiter((int(e) in tr_eids for e in eids), bool, len(eids))
        masks[seed] = (tr, ~tr)

    # Full cell grid; shard by global index; sort my cells by feature-set so the
    # built concatenated features (cache size 1, keyed by feature_set+seed) is
    # reused across the archs of the same feature set.
    grid = list(itertools.product(feature_sets, archs, seeds))
    mine = [c for i, c in enumerate(grid) if i % args.n_gpus == args.gpu_id]
    mine.sort(key=lambda c: (_fslug(c[0]), c[2], c[1]))
    print(f"[richer] gpu {args.gpu_id}/{args.n_gpus}: {len(mine)}/{len(grid)} cells",
          file=sys.stderr)

    cache_key, Xtr_c, Xte_c = None, None, None  # (feature_set,seed) -> built features
    for layers, arch, seed in mine:
        fslug = _fslug(layers)
        tag = f"{fslug}_{arch}_s{seed}"
        dst = out / f"cell_{tag}.json"
        if dst.exists():
            continue
        try:
            key = (tuple(layers), seed)
            if key != cache_key:
                tr, te = masks[seed]
                Xtr_c, Xte_c = _build_features(acts, layers, tr, te)
                cache_key = key
            if Xtr_c is None:
                dst.write_text(json.dumps({**_meta(layers, arch, seed),
                                           "error": "non-finite acts"}))
                continue
            tr, te = masks[seed]
            ytr, etr = y[tr], eids[tr]
            if len(np.unique(ytr)) < 2 or te.sum() == 0:
                dst.write_text(json.dumps({**_meta(layers, arch, seed),
                                           "skipped": "degenerate"}))
                continue
            factory = _factory_for_arch(arch)
            r = train_one_layer(Xtr_c, ytr, etr, epochs=args.epochs, device=device,
                                verbose=False, alpha=args.alpha, neg_incl=False,
                                probe_factory=factory)
            with torch.no_grad():
                tok_logits = r["probe"](torch.from_numpy(Xte_c).float()).numpy()
            if not np.isfinite(tok_logits).all():
                dst.write_text(json.dumps({**_meta(layers, arch, seed), "error": "diverged"}))
                continue
            tok_p = 1.0 / (1.0 + np.exp(-tok_logits))
            tok_y, te_eids = y[te], eids[te]
            tok_auc = (float(roc_auc_score(tok_y, tok_p))
                       if len(np.unique(tok_y)) > 1 else float("nan"))
            ex_ids, ex_p = te_mod.example_scores(tok_p, te_eids)
            ex_y = np.array([int(y[(eids == e)].max() > 0) for e in ex_ids])
            ex_auc = (float(roc_auc_score(ex_y, ex_p))
                      if len(np.unique(ex_y)) > 1 else float("nan"))
            rec = {**_meta(layers, arch, seed), "in_dim": int(Xtr_c.shape[1]),
                   "test_ex_auc": ex_auc, "test_tok_auc": tok_auc,
                   "val_ex_auc": float(r["ex_auc"])}
            dst.write_text(json.dumps(rec))
            print(f"[richer] {tag}  ex={ex_auc:.3f} tok={tok_auc:.3f}", file=sys.stderr)
        except Exception as e:  # noqa: BLE001 -- record + continue, never abort the sweep
            dst.write_text(json.dumps({**_meta(layers, arch, seed),
                                       "error": f"{type(e).__name__}: {str(e)[:200]}"}))
            print(f"[richer] {tag} ERROR {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)


if __name__ == "__main__":
    main()
