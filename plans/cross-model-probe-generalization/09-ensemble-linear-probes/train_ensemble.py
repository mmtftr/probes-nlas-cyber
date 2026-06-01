# [ai-generated]
"""Train ONE ensemble-of-K-linear-probes head at a model's best layer and report
the honest `tokens_code` AUC overall + per-language + per-CWE.

Model-agnostic. Runs on CACHED acts at a single layer (no re-extraction). Mirrors
`06/train_all_layers.py` EXACTLY for the splits (group-aware test hold-out +
15% group-aware VAL carve, VAL_SEED=42) and the honest eval (`honest_token_aucs`),
so results are apples-to-apples with the 0.788 single-linear baseline. Mirrors
`06/breakdown_lang_cwe.py` for the per-lang / per-CWE subsets.

Key difference from the linear sweep: the ensemble head is non-linear (max /
logsumexp / softmax_gate over K directions), so `train_one_layer` returns
`w=None` and we score test/val tokens by RUNNING THE TRAINED MODULE, not `X @ w`.

Outputs one JSON per (K, agg) cell:
    {layer, K, agg, tau, gate_mode,
     val_tokens_code_auc, val_tokens_auc,           # selection signal
     overall {tokens_code_auc, tokens_auc, n_pos_code, n_total_code, n_examples},
     by_lang {python|c|cpp: {...}},
     by_cwe  {CWE-XXX: {..., n_pos_examples}},
     test_ex_auc, n_test_ex,
     directions_path}                                 # .pt with K dirs + gate params
plus a sibling `<out>.dirs.pt` holding the K weight directions + gate params for
later inter-direction cosine-sim / per-CWE firing analysis.

Resumable: skips if <out> already exists.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.training.train_probe_spanmax import train_one_layer  # noqa: E402
from src.eval.honest_scoring import (  # noqa: E402
    honest_token_aucs, load_dataset_rows, load_offsets_npz,
)
from sklearn.metrics import roc_auc_score  # noqa: E402
from ensemble_probe import make_factory  # noqa: E402

# Same group-aware VAL carve as 06/train_all_layers.py (must match for honest
# layer/cell selection that is apples-to-apples with the linear baseline).
VAL_FRAC = 0.15
VAL_SEED = 42


def _load_train_eval():
    p = REPO / "src" / "remotes" / "the cluster" / "train_eval.py"
    spec = importlib.util.spec_from_file_location("remote_train_eval", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _auc(yv, pv) -> float:
    return float(roc_auc_score(yv, pv)) if len(np.unique(yv)) > 1 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--best-layer", type=int, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--agg", required=True, choices=("max", "logsumexp", "softmax_gate"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--offsets", default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--model", default="")
    # TODO(adhoc-decision): tau default = 1.0 (smooth max). The lead may want a
    # tau sweep {1,4,10} for agg=logsumexp; exposed as a CLI knob, not hard-coded.
    ap.add_argument("--tau", type=float, default=1.0)
    # TODO(adhoc-decision): gate granularity per_token (default) vs global.
    ap.add_argument("--gate-mode", default="per_token", choices=("per_token", "global"))
    ap.add_argument("--min-cwe-pos", type=int, default=10,
                    help="skip CWEs with fewer than this many positive test rows")
    ap.add_argument("--div-lambda", type=float, default=0.0,
                    help="weight on the EnsembleProbe direction-divergence penalty "
                         "(mean cos^2 between directions). 0 = off (default).")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        print(f"[ensemble] {out} exists, skipping", file=sys.stderr)
        return

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    te_mod = _load_train_eval()
    acts = Path(args.acts_dir)
    offsets_by_eid = load_offsets_npz(Path(args.offsets) if args.offsets else acts / "offsets.npz")
    rows_by_eid = load_dataset_rows(Path(args.dataset))

    y = np.load(acts / "y.npy")
    eids = np.load(acts / "example_ids.npy")
    rows, train_eids, test_eids = te_mod.load_or_make_split(Path(args.dataset), Path(args.split))

    # 15% group-aware VAL carve from TRAIN, identical recipe to 06/train_all_layers.
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
    print(f"[ensemble] {args.model} L{args.best_layer} K={args.K} agg={args.agg} "
          f"fit_tok={fit.sum()} val_tok={val.sum()} test_tok={te.sum()}", file=sys.stderr)

    Xmm = np.load(acts / f"layer_{args.best_layer:02d}.npy", mmap_mode="r")
    Xfit = np.asarray(Xmm[fit], dtype=np.float32)
    if len(np.unique(y[fit])) < 2 or te.sum() == 0 or val.sum() == 0:
        out.write_text(json.dumps({"layer": args.best_layer, "K": args.K, "agg": args.agg,
                                   "skipped": "degenerate labels/splits"}))
        return
    if not np.isfinite(Xfit).all():
        out.write_text(json.dumps({"layer": args.best_layer, "K": args.K, "agg": args.agg,
                                   "error": "non-finite activations"}))
        return

    factory = make_factory(args.K, args.agg, tau=args.tau, gate_mode=args.gate_mode)
    r = train_one_layer(Xfit, y[fit], eids[fit], epochs=args.epochs, device=device,
                        verbose=False, probe_factory=factory,
                        reg_penalty=(lambda m: m.divergence_penalty()) if args.div_lambda else None,
                        reg_weight=args.div_lambda)
    probe = r["probe"].to(device).eval()

    # Post-training direction diversity: mean off-diagonal |cosine| of the K
    # directions (1.0 = collapsed, ~0 = orthogonal). Lets us confirm the
    # divergence penalty actually orthogonalised the directions.
    import torch.nn.functional as _F
    _W = _F.normalize(probe.directions()["W"].float(), dim=1)
    _C = (_W @ _W.t())
    _K = _W.shape[0]
    post_cos = float((_C.abs().sum() - _K) / (_K * (_K - 1))) if _K > 1 else 0.0

    def score(mask) -> np.ndarray:
        """Run the trained ensemble module on a token mask -> per-token sigmoid prob."""
        Xs = np.asarray(Xmm[mask], dtype=np.float32)
        with torch.no_grad():
            logits = probe(torch.from_numpy(Xs).to(device))
            return torch.sigmoid(logits).detach().cpu().numpy()

    # selection-val honest AUC (the cell-selection signal, like the linear sweep)
    val_p = score(val)
    val_h = honest_token_aucs(val_p, y[val], eids[val], offsets_by_eid, rows_by_eid)

    # TEST tokens (scored once, then sliced for overall / per-lang / per-CWE).
    tok_p = score(te)
    tok_y, te_e = y[te], eids[te]
    test_list = sorted(test_eids)

    def subset(eid_set):
        if not eid_set:
            return None
        m = np.isin(te_e, np.fromiter(eid_set, dtype=te_e.dtype))
        if m.sum() == 0:
            return None
        h = honest_token_aucs(tok_p[m], tok_y[m], te_e[m], offsets_by_eid, rows_by_eid)
        return {"tokens_code_auc": h["tokens_code_auc"], "tokens_auc": h["tokens_auc"],
                "n_pos_code": h["n_pos_code"], "n_total_code": h["n_total_code"],
                "n_examples": int(len(eid_set))}

    overall = subset(set(test_list))
    by_lang = {}
    for lang in ("python", "c", "cpp"):
        es = {e for e in test_list if (rows[e].get("lang") or "").lower() == lang}
        s = subset(es)
        if s:
            by_lang[lang] = s

    neg_eids = {e for e in test_list if not rows[e].get("cwe")}
    cwe_counts = Counter(rows[e].get("cwe") for e in test_list if rows[e].get("cwe"))
    by_cwe = {}
    for cwe, n in cwe_counts.most_common():
        if n < args.min_cwe_pos:
            continue
        pos_eids = {e for e in test_list if rows[e].get("cwe") == cwe}
        s = subset(pos_eids | neg_eids)
        if s:
            s["n_pos_examples"] = len(pos_eids)
            by_cwe[cwe] = s

    ex_ids, ex_p = te_mod.example_scores(tok_p, te_e)
    ex_y = np.array([int(y[(eids == e)].max() > 0) for e in ex_ids])

    # Save the K directions + gate params for inspection (cosine sim / per-CWE firing).
    dirs_path = out.with_suffix(".dirs.pt")
    torch.save(probe.directions(), dirs_path)

    rec = {
        "model": args.model, "layer": args.best_layer,
        "K": args.K, "agg": args.agg, "tau": args.tau, "gate_mode": args.gate_mode,
        "div_lambda": args.div_lambda, "post_train_cos_abs_mean": post_cos,
        "val_tokens_code_auc": val_h["tokens_code_auc"],
        "val_tokens_auc": val_h["tokens_auc"],
        "overall": overall,
        "by_lang": by_lang,
        "by_cwe": by_cwe,
        "test_ex_auc": _auc(ex_y, ex_p),
        "n_test_ex": int(len(ex_ids)),
        "directions_path": str(dirs_path),
    }
    out.write_text(json.dumps(rec, indent=2))
    ov = overall["tokens_code_auc"] if overall else float("nan")
    print(f"[ensemble] {args.model} L{args.best_layer} K={args.K} {args.agg} "
          f"val_tc={val_h['tokens_code_auc']:.3f} test_tc={ov:.3f} "
          f"langs={list(by_lang)} cwes={list(by_cwe)}", file=sys.stderr)


if __name__ == "__main__":
    main()
