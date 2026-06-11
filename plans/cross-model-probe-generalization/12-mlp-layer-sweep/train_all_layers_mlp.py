# [ai-generated]
"""Honest per-layer sweep with an MLP head + tokens_code-based layer selection.

Adapted from `06-honest-metric-sweeps/train_all_layers.py` (linear span-max
sweep). The ONLY substantive change is the probe head: instead of a single
linear direction, every layer is fit with an MLP head (default `mlp256`).

Motivation (Tier-4 #8 — the TRUE MLP ceiling): exp-09 measured the MLP head
ONLY at the LINEAR-selected best layer. But the MLP's own optimal layer may
differ. This runner sweeps the MLP head over ALL layers, selects by
`val_tokens_code_auc` (held-out, no test leakage), and reports the MLP's true
best-layer TEST `tokens_code` — the honest MLP ceiling, comparable to the
linear ceiling from exp-06.

Splits (identical to exp-06, all group-aware via pair_group_key):
  - test  = the persisted seeded 20% group hold-out (load_or_make_split).
  - val   = a further 15% of the TRAIN groups, held out for layer SELECTION
            only (seed=42, deterministic, disjoint from test).
  - fit   = the remaining train groups; the probe is trained on these.

Per-layer JSON fields (same as exp-06, plus `head`):
    head                                   probe head provenance (e.g. mlp256).
    val_tokens_code_auc / val_tokens_auc   selection-val honest + inflated AUC.
    tokens_code_auc / tokens_auc           TEST honest + inflated AUC (reported).
    test_tok_auc                           backward-compat alias of TEST tokens_auc.
    dropped_fraction                       1 - live_code_mask.mean() over TEST.
    val_ex_auc / test_ex_auc               example-level (ride along; not used
                                           for selection).

CRITICAL vs exp-06: the MLP head returns w=None, so scoring CANNOT use the
closed-form X@w+b that exp-06 uses. We score with the trained torch MODULE
forward (torch.sigmoid(probe(X))), copied from 09/train_head_baseline.py. The
np.isfinite(w) weight-divergence guard is dropped (no w); the np.isfinite(Xfit)
activation guard and the degenerate-split guard are kept.

Resumable + GPU-shardable: one layer_{NN}.json per layer; skips existing.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from src.training.train_probe_spanmax import MLPProbe, train_one_layer  # noqa: E402
from src.eval.honest_scoring import (  # noqa: E402
    honest_token_aucs,
    load_dataset_rows,
    load_offsets_npz,
)
from sklearn.metrics import roc_auc_score  # noqa: E402

# Fraction of TRAIN groups held out for layer selection (val). seed fixed for
# determinism + resumability across shards/re-runs.
VAL_FRAC = 0.15
VAL_SEED = 42


def _load_train_eval():
    """Import the canonical train_eval module by path (no __init__ in its dir)."""
    p = REPO / "src" / "remotes" / "train_eval.py"
    spec = importlib.util.spec_from_file_location("remote_train_eval", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _auc(yv, pv) -> float:
    return float(roc_auc_score(yv, pv)) if len(np.unique(yv)) > 1 else float("nan")


def _factory_for(head: str):
    """head -> probe_factory. 'mlp256','mlp512' -> MLPProbe(H). (No linear here:
    this sweep is the MLP ceiling; the linear ceiling lives in exp-06.)"""
    m = re.fullmatch(r"mlp(\d+)", head)
    if not m:
        raise ValueError(f"bad --head {head!r} (expected mlpH, e.g. mlp256|mlp512)")
    H = int(m.group(1))
    return lambda d, H=H: MLPProbe(d, hidden=H)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--head", default="mlp256", help="mlp256 | mlp512 | mlpH")
    ap.add_argument("--offsets", default=None,
                    help="Per-row char offsets npz (offsets_row_NNNN keys). "
                         "Defaults to <acts-dir>/offsets.npz.")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--n-gpus", type=int, default=1)
    ap.add_argument("--gpu-id", type=int, default=0)
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    factory = _factory_for(args.head)  # validate head early (raises on bad value)

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

    rows, train_eids, test_eids = te_mod.load_or_make_split(
        Path(args.dataset), Path(args.split)
    )
    # Carve a group-aware selection-val split from TRAIN (disjoint from test).
    tr_eid_to_group = {e: te_mod.pair_group_key(rows[e]) for e in train_eids}
    tr_groups = sorted(set(tr_eid_to_group.values()))
    rng = np.random.default_rng(VAL_SEED)
    rng.shuffle(tr_groups)
    n_val = max(1, int(round(VAL_FRAC * len(tr_groups))))
    val_groups = set(tr_groups[:n_val])
    val_eids = {e for e, g in tr_eid_to_group.items() if g in val_groups}
    fit_eids = train_eids - val_eids

    fit = np.fromiter((int(e) in fit_eids for e in eids), bool, len(eids))
    val = np.fromiter((int(e) in val_eids for e in eids), bool, len(eids))
    te = np.fromiter((int(e) in test_eids for e in eids), bool, len(eids))
    print(f"[train-all-mlp] gpu {args.gpu_id}/{args.n_gpus}  head={args.head}  "
          f"fit_tok={fit.sum()} val_tok={val.sum()} test_tok={te.sum()}", file=sys.stderr)

    my_layers = [li for li in range(n_layers) if li % args.n_gpus == args.gpu_id]

    for li in my_layers:
        dst = out / f"layer_{li:02d}.json"
        if dst.exists():
            continue
        # Isolate per-layer failures: one diverging layer must not abort the rest.
        try:
            Xmm = np.load(acts / f"layer_{li:02d}.npy", mmap_mode="r")
            Xfit = np.asarray(Xmm[fit], dtype=np.float32)
            yfit, efit = y[fit], eids[fit]
            if len(np.unique(yfit)) < 2 or te.sum() == 0 or val.sum() == 0:
                dst.write_text(json.dumps({"layer": li, "head": args.head,
                                           "skipped": "degenerate labels/splits"}))
                continue
            if not np.isfinite(Xfit).all():
                dst.write_text(json.dumps({"layer": li, "head": args.head,
                                           "error": "non-finite activations"}))
                print(f"[train-all-mlp] layer {li:02d} SKIP non-finite activations", file=sys.stderr)
                continue
            r = train_one_layer(Xfit, yfit, efit, epochs=args.epochs, device=device,
                                verbose=False, probe_factory=factory)
            # MLP head -> w=None, b=None; the trained module is always in r["probe"].
            # No closed-form weight-divergence guard (no w to inspect); score via
            # the module forward instead (see _score below).
            probe = r["probe"].to(device).eval()

            def _score(mask):
                Xs = np.asarray(Xmm[mask], dtype=np.float32)
                with torch.no_grad():
                    logits = probe(torch.from_numpy(Xs).to(device))
                    return torch.sigmoid(logits).detach().cpu().numpy()

            # selection-val honest AUC (the layer-selection signal)
            val_p = _score(val)
            val_h = honest_token_aucs(val_p, y[val], eids[val], offsets_by_eid, dataset_rows_by_eid)
            # test honest AUC (reported)
            tok_p = _score(te)
            tok_y, te_eids = y[te], eids[te]
            test_h = honest_token_aucs(tok_p, tok_y, te_eids, offsets_by_eid, dataset_rows_by_eid)
            ex_ids, ex_p = te_mod.example_scores(tok_p, te_eids)
            ex_y = np.array([int(y[(eids == e)].max() > 0) for e in ex_ids])

            rec = {"layer": li, "head": args.head, "layer_frac": li / (n_layers - 1),
                   # selection signal
                   "val_tokens_code_auc": val_h["tokens_code_auc"],
                   "val_tokens_auc": val_h["tokens_auc"],
                   # test (reported)
                   "tokens_code_auc": test_h["tokens_code_auc"],
                   "tokens_auc": test_h["tokens_auc"],
                   "test_tok_auc": _auc(tok_y, tok_p),  # backward-compat alias
                   "dropped_fraction": test_h["dropped_fraction"],
                   "n_pos_code": test_h["n_pos_code"],
                   "n_total_code": test_h["n_total_code"],
                   # example-level rides along (not used for selection)
                   "test_ex_auc": _auc(ex_y, ex_p),
                   "val_ex_auc": float(r["ex_auc"]),
                   "n_test_ex": int(len(ex_ids))}
            dst.write_text(json.dumps(rec))
            print(f"[train-all-mlp] layer {li:02d} {args.head}  "
                  f"val_tc={val_h['tokens_code_auc']:.3f}  "
                  f"test_tc={test_h['tokens_code_auc']:.3f}  "
                  f"test_tok={test_h['tokens_auc']:.3f} (dropped {test_h['dropped_fraction']:.2f})",
                  file=sys.stderr)
        except Exception as e:  # noqa: BLE001 — record + continue, never abort the sweep
            dst.write_text(json.dumps({"layer": li, "head": args.head,
                                       "error": f"{type(e).__name__}: {str(e)[:200]}"}))
            print(f"[train-all-mlp] layer {li:02d} ERROR {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)


if __name__ == "__main__":
    main()
