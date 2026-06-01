# [ai-generated]
"""Train ONE probe head (linear | mlpH) at a model's best layer and report honest
`tokens_code` AUC overall + per-language + per-CWE — the SAME pipeline as
`train_ensemble.py` (identical group-aware test split + 15% VAL carve VAL_SEED=42,
identical `honest_token_aucs` eval, identical fit set), differing ONLY in the probe
head. This makes the linear and MLP points directly comparable to the ensemble cells.

Purpose (per the lead): exp-09's real question is whether an ensemble of K
INTERPRETABLE linear directions can RECOVER the MLP's gain over a single linear
probe. This runner supplies the two reference points the ensemble is measured
against: `linear` (the floor) and `mlpH` (the target).

Heads:
  linear  -> LinearProbe            (probe_factory=None; == the K=1 ensemble cell)
  mlp256  -> MLPProbe(hidden=256)    (same arch family as exp-04 sweep-5)
  mlp512  -> MLPProbe(hidden=512)

Output JSON mirrors train_ensemble's cell, with `head` in place of K/agg. Resumable.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from src.training.train_probe_spanmax import MLPProbe, train_one_layer  # noqa: E402
from src.eval.honest_scoring import (  # noqa: E402
    honest_token_aucs, load_dataset_rows, load_offsets_npz,
)
from sklearn.metrics import roc_auc_score  # noqa: E402

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


def _factory_for(head: str):
    """head -> probe_factory (None == LinearProbe). 'mlp256','mlp512' -> MLPProbe(H)."""
    if head == "linear":
        return None
    m = re.fullmatch(r"mlp(\d+)", head)
    if not m:
        raise ValueError(f"bad --head {head!r} (expected linear|mlpH)")
    H = int(m.group(1))
    return lambda d, H=H: MLPProbe(d, hidden=H)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--best-layer", type=int, required=True)
    ap.add_argument("--head", required=True, help="linear | mlp256 | mlp512 | mlpH")
    ap.add_argument("--out", required=True)
    ap.add_argument("--offsets", default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--model", default="")
    ap.add_argument("--min-cwe-pos", type=int, default=10)
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        print(f"[head] {out} exists, skipping", file=sys.stderr)
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
    print(f"[head] {args.model} L{args.best_layer} head={args.head} "
          f"fit_tok={fit.sum()} val_tok={val.sum()} test_tok={te.sum()}", file=sys.stderr)

    Xmm = np.load(acts / f"layer_{args.best_layer:02d}.npy", mmap_mode="r")
    Xfit = np.asarray(Xmm[fit], dtype=np.float32)
    if len(np.unique(y[fit])) < 2 or te.sum() == 0 or val.sum() == 0:
        out.write_text(json.dumps({"layer": args.best_layer, "head": args.head,
                                   "skipped": "degenerate labels/splits"}))
        return
    if not np.isfinite(Xfit).all():
        out.write_text(json.dumps({"layer": args.best_layer, "head": args.head,
                                   "error": "non-finite activations"}))
        return

    factory = _factory_for(args.head)
    r = train_one_layer(Xfit, y[fit], eids[fit], epochs=args.epochs, device=device,
                        verbose=False, probe_factory=factory)
    probe = r["probe"].to(device).eval()

    def score(mask) -> np.ndarray:
        Xs = np.asarray(Xmm[mask], dtype=np.float32)
        with torch.no_grad():
            logits = probe(torch.from_numpy(Xs).to(device))
            return torch.sigmoid(logits).detach().cpu().numpy()

    val_p = score(val)
    val_h = honest_token_aucs(val_p, y[val], eids[val], offsets_by_eid, rows_by_eid)

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

    rec = {
        "model": args.model, "layer": args.best_layer, "head": args.head,
        "val_tokens_code_auc": val_h["tokens_code_auc"],
        "val_tokens_auc": val_h["tokens_auc"],
        "overall": overall, "by_lang": by_lang, "by_cwe": by_cwe,
        "test_ex_auc": _auc(ex_y, ex_p), "n_test_ex": int(len(ex_ids)),
    }
    out.write_text(json.dumps(rec, indent=2))
    ov = overall["tokens_code_auc"] if overall else float("nan")
    print(f"[head] {args.model} L{args.best_layer} {args.head} "
          f"val_tc={val_h['tokens_code_auc']:.3f} test_tc={ov:.3f}", file=sys.stderr)


if __name__ == "__main__":
    main()
