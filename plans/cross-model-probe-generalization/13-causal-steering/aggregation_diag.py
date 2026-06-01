# [ai-generated] DIAGNOSTIC (analysis-only; do NOT commit results).
"""Why does example-level max-pool AUC fall BELOW chance for the GENERAL
span-max probe on the MEMORY family, when the token-level (code-masked) AUC is
above chance?  And why does the FAMILY-pooled probe aggregate fine?

This reproduces, at ONE model's best layer, on the seed-42 group-clean split:
  - GENERAL probe: train_one_layer on ALL train tokens (belief-audit recipe).
  - FAMILY probe : train_one_layer on {memory train positives} U {neg train}.
Then, on the TEST set restricted to {memory test positives} U {negatives},
dumps per-token sigmoid and computes:

  (A) token-level AUC two ways:
        tokens_all  = AUC over every token   (tok_y per-token label)
        tokens_code = AUC over live-code tokens only (the exp-10 honest metric)
      -> reproduce ~0.519 (Qwen) / ~0.561 (gemma) tokens_code for general.

  (B) example-level AUCs under many aggregations (pos = memory, neg = cwe==null):
        max_all        : max over ALL tokens          (the belief-audit number)
        mean_all       : mean over ALL tokens
        topk5_all      : mean of top-5 token probs (all tokens)
        topk10_all     : mean of top-10
        max_code       : max over LIVE-CODE tokens only
        mean_code      : mean over live-code tokens only
        topk5_code     : mean top-5 over live-code tokens
        max_labelpos   : max over LABELED-positive tokens (tok_y==1); neg uses max_code
        mean_labelpos  : mean over labeled-positive tokens; neg uses mean_code
      -> reproduce ~0.32 (Qwen) / ~0.47 (gemma) for general max_all; see which
         aggregation is most faithful and whether ANY puts general above chance.

  (C) distributions: per-example max / mean / top5 / n_tok / n_code, pos vs neg.
        Is neg max systematically ABOVE pos max?  Is max driven by token COUNT
        (length)?  Correlation of per-example max-prob with n_tok / n_code.

  (D) the general-vs-family contrast on the SAME examples: family max_all AUC,
        plus the per-example score distributions, to isolate what differs in the
        general probe's token-score distribution on memory-vuln C.

Outputs a JSON blob to --out (under scratch; not committed).
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

import os
# Prefer the env REPO (set by the cluster env.sh); fall back to repo-tree layout.
_env_repo = os.environ.get("REPO")
REPO = Path(_env_repo) if _env_repo else Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from src.training.train_probe_spanmax import train_one_layer  # noqa: E402
from src.eval.honest_scoring import (  # noqa: E402
    load_offsets_npz, load_dataset_rows, build_code_mask,
)

FAMILY = {
    "CWE-089": "injection", "CWE-078": "injection", "CWE-022": "injection",
    "CWE-079": "injection", "CWE-190": "injection",
    "CWE-125": "memory", "CWE-476": "memory", "CWE-416": "memory",
    "CWE-787": "memory",
}


def _load_train_eval():
    p = REPO / "src" / "remotes" / "the cluster" / "train_eval.py"
    spec = importlib.util.spec_from_file_location("remote_train_eval", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_split_for_seed(eid_to_group, seed, frac_heldout=0.2):
    """Belief-audit split (compare_belief_audit.make_split_for_seed), VERBATIM."""
    groups = sorted(set(eid_to_group.values()))
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n_held = max(1, int(round(frac_heldout * len(groups))))
    heldout = set(groups[:n_held])
    train_eids = {e for e, g in eid_to_group.items() if g not in heldout}
    return train_eids, {e for e, g in eid_to_group.items() if g in heldout}


def _tok_probs(probe_result, X):
    w = np.asarray(probe_result["w"], np.float32)
    b = float(probe_result["b"])
    return 1.0 / (1.0 + np.exp(-(X @ w + b)))


def _auc(y, s):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=1.0)
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[diag] device={device} model={args.model} layer={args.layer}",
          file=sys.stderr)

    acts = Path(args.acts_dir)
    layer = args.layer
    te_mod = _load_train_eval()

    Xfull = np.asarray(np.load(acts / f"layer_{layer:02d}.npy", mmap_mode="r"),
                       dtype=np.float32)
    y = np.load(acts / "y.npy")
    eids = np.load(acts / "example_ids.npy")
    offsets_by_eid = load_offsets_npz(acts / "offsets.npz")
    rows_by_eid = load_dataset_rows(args.dataset)

    rows = [json.loads(l) for l in Path(args.dataset).open() if l.strip()]
    eid_to_group = {i: te_mod.pair_group_key(r) for i, r in enumerate(rows)}

    def cwe_of(e):
        return rows[int(e)].get("cwe")

    def family_of(e):
        return FAMILY.get(cwe_of(e))

    acts_eids = set(int(e) for e in np.unique(eids))

    tr_eids, te_eids = make_split_for_seed(eid_to_group, args.seed)
    tr = np.fromiter((int(e) in tr_eids for e in eids), bool, len(eids))
    te = ~tr
    ytr, etr = y[tr], eids[tr]

    # --- GENERAL probe (all train tokens) ---
    print("[diag] training GENERAL probe ...", file=sys.stderr)
    gen_res = train_one_layer(Xfull[tr], ytr, etr, epochs=args.epochs,
                              device=device, verbose=False, alpha=args.alpha,
                              neg_incl=False)

    # --- FAMILY (memory) probe: memory train pos U neg train ---
    neg_tr = {e for e in tr_eids if not cwe_of(e) and e in acts_eids}
    neg_te = {e for e in te_eids if not cwe_of(e) and e in acts_eids}
    pos_tr = {e for e in tr_eids if family_of(e) == "memory" and e in acts_eids}
    pos_te = {e for e in te_eids if family_of(e) == "memory" and e in acts_eids}

    spec_fit_eids = pos_tr | neg_tr
    spec_fit_mask = np.fromiter((int(e) in spec_fit_eids for e in eids),
                                bool, len(eids))
    print(f"[diag] training FAMILY(memory) probe on {spec_fit_mask.sum()} tokens"
          f" ({len(pos_tr)} pos ex, {len(neg_tr)} neg ex) ...", file=sys.stderr)
    spec_res = train_one_layer(
        np.asarray(Xfull[spec_fit_mask], np.float32),
        y[spec_fit_mask], eids[spec_fit_mask],
        epochs=args.epochs, device=device, verbose=False,
        alpha=args.alpha, neg_incl=False)

    # --- TEST eval set: memory positives U all negatives ---
    eval_eids = pos_te | neg_te
    eval_mask = np.fromiter((int(e) in eval_eids for e in eids), bool, len(eids))
    Xev = np.asarray(Xfull[eval_mask], np.float32)
    ev_tok_eids = eids[eval_mask]
    ev_tok_y = y[eval_mask]

    gen_tok_p = _tok_probs(gen_res, Xev)
    spec_tok_p = _tok_probs(spec_res, Xev)

    # live-code mask over the eval tokens
    code_mask = build_code_mask(ev_tok_eids, offsets_by_eid, rows_by_eid)
    print(f"[diag] eval tokens={len(ev_tok_eids)} code_frac={code_mask.mean():.3f}",
          file=sys.stderr)

    # ===== (A) token-level AUC reproduction =====
    # token-level positives = labeled vuln-line tokens (tok_y==1) vs rest.
    def tok_auc_all(p):
        return _auc(ev_tok_y, p)

    def tok_auc_code(p):
        return _auc(ev_tok_y[code_mask], p[code_mask])

    token_block = {
        "general": {"tokens_all": tok_auc_all(gen_tok_p),
                    "tokens_code": tok_auc_code(gen_tok_p)},
        "family": {"tokens_all": tok_auc_all(spec_tok_p),
                   "tokens_code": tok_auc_code(spec_tok_p)},
        "n_eval_tokens": int(len(ev_tok_eids)),
        "n_pos_tokens_all": int((ev_tok_y == 1).sum()),
        "n_pos_tokens_code": int((ev_tok_y[code_mask] == 1).sum()),
        "code_frac": float(code_mask.mean()),
    }

    # ===== per-example aggregation machinery =====
    uniq_ex = np.unique(ev_tok_eids)
    ex_label = np.array([int(family_of(e) == "memory") for e in uniq_ex])  # 1=memory-pos
    # sanity: negatives are cwe==null
    is_pos = ex_label == 1

    def per_example(reduce_fn, p, restrict=None, pos_only=False):
        """reduce_fn(arr)->scalar over the per-example token probs (optionally
        restricted to a boolean per-token mask `restrict`). pos_only: restrict to
        labeled-positive tokens (tok_y==1) for positive examples; for negative
        examples (no labeled-pos tokens) fall back to `restrict` (code) tokens."""
        out = np.full(len(uniq_ex), np.nan)
        for i, e in enumerate(uniq_ex):
            sel = ev_tok_eids == e
            pe = p[sel]
            ye = ev_tok_y[sel]
            ce = code_mask[sel] if restrict is not None else np.ones(len(pe), bool)
            if pos_only:
                lp = ye == 1
                if lp.any():
                    arr = pe[lp]
                else:  # negatives: no labeled pos -> fall back to code tokens
                    arr = pe[ce] if ce.any() else pe
            elif restrict is not None:
                arr = pe[ce] if ce.any() else pe
            else:
                arr = pe
            if len(arr) == 0:
                arr = pe
            out[i] = reduce_fn(arr)
        return out

    def topk_mean(k):
        return lambda a: float(np.sort(a)[-k:].mean())

    aggs = {
        "max_all": (np.max, None, False),
        "mean_all": (np.mean, None, False),
        "topk5_all": (topk_mean(5), None, False),
        "topk10_all": (topk_mean(10), None, False),
        "max_code": (np.max, "code", False),
        "mean_code": (np.mean, "code", False),
        "topk5_code": (topk_mean(5), "code", False),
        "max_labelpos": (np.max, "code", True),
        "mean_labelpos": (np.mean, "code", True),
    }

    # ===== (B) example-level AUC under each aggregation, general + family =====
    agg_block = {"general": {}, "family": {}}
    ex_scores = {"general": {}, "family": {}}
    for name, (fn, restrict, pos_only) in aggs.items():
        gs = per_example(fn, gen_tok_p, restrict, pos_only)
        fs = per_example(fn, spec_tok_p, restrict, pos_only)
        agg_block["general"][name] = _auc(ex_label, gs)
        agg_block["family"][name] = _auc(ex_label, fs)
        ex_scores["general"][name] = gs
        ex_scores["family"][name] = fs

    # ===== (C) distributions + length analysis =====
    n_tok = np.array([int((ev_tok_eids == e).sum()) for e in uniq_ex])
    n_code = np.array([int(code_mask[ev_tok_eids == e].sum()) for e in uniq_ex])

    def summ(a):
        a = np.asarray(a, float)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return None
        return {"mean": float(a.mean()), "median": float(np.median(a)),
                "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90)),
                "min": float(a.min()), "max": float(a.max()), "n": int(a.size)}

    def dist_block(p, scores):
        gmax = scores["max_all"]
        gmean = scores["mean_all"]
        gtop5 = scores["topk5_all"]
        gmax_code = scores["max_code"]
        out = {
            "max_all_pos": summ(gmax[is_pos]),
            "max_all_neg": summ(gmax[~is_pos]),
            "mean_all_pos": summ(gmean[is_pos]),
            "mean_all_neg": summ(gmean[~is_pos]),
            "top5_all_pos": summ(gtop5[is_pos]),
            "top5_all_neg": summ(gtop5[~is_pos]),
            "max_code_pos": summ(gmax_code[is_pos]),
            "max_code_neg": summ(gmax_code[~is_pos]),
            # fraction of negatives whose max EXCEEDS the median positive max
            "frac_neg_max_above_pos_median_max":
                float(np.mean(gmax[~is_pos] > np.median(gmax[is_pos]))),
            # correlation of per-example max with token count (length artifact?)
            "corr_max_all_vs_ntok": float(np.corrcoef(gmax, n_tok)[0, 1]),
            "corr_max_all_vs_ncode": float(np.corrcoef(gmax, n_code)[0, 1]),
            "corr_max_all_vs_ntok_neg":
                float(np.corrcoef(gmax[~is_pos], n_tok[~is_pos])[0, 1]),
            "corr_max_all_vs_ntok_pos":
                float(np.corrcoef(gmax[is_pos], n_tok[is_pos])[0, 1]),
        }
        return out

    dist_general = dist_block(gen_tok_p, ex_scores["general"])
    dist_family = dist_block(spec_tok_p, ex_scores["family"])

    length_block = {
        "n_tok_pos": summ(n_tok[is_pos]),
        "n_tok_neg": summ(n_tok[~is_pos]),
        "n_code_pos": summ(n_code[is_pos]),
        "n_code_neg": summ(n_code[~is_pos]),
        # AUC of pure token-count as the score (length baseline at example level)
        "length_ntok_auc": _auc(ex_label, n_tok.astype(float)),
        "length_ncode_auc": _auc(ex_label, n_code.astype(float)),
    }

    record = {
        "model": args.model,
        "layer": layer,
        "seed": args.seed,
        "epochs": args.epochs,
        "alpha": args.alpha,
        "n_eval_examples": int(len(uniq_ex)),
        "n_pos_examples": int(is_pos.sum()),
        "n_neg_examples": int((~is_pos).sum()),
        "token_auc": token_block,
        "example_agg_auc": agg_block,
        "dist_general": dist_general,
        "dist_family": dist_family,
        "length": length_block,
    }
    Path(args.out).write_text(json.dumps(record, indent=2))

    # --- human-readable summary to stderr ---
    print(f"\n[diag] === {args.model} L{layer} (seed {args.seed}) ===", file=sys.stderr)
    print(f"[diag] n_ex pos={is_pos.sum()} neg={(~is_pos).sum()}", file=sys.stderr)
    print(f"[diag] TOKEN general: all={token_block['general']['tokens_all']:.3f}"
          f" code={token_block['general']['tokens_code']:.3f}  |  "
          f"family: all={token_block['family']['tokens_all']:.3f}"
          f" code={token_block['family']['tokens_code']:.3f}", file=sys.stderr)
    print("[diag] EXAMPLE-AUC (agg: general / family):", file=sys.stderr)
    for name in aggs:
        g = agg_block["general"][name]
        f = agg_block["family"][name]
        print(f"[diag]   {name:14s} {g:.3f} / {f:.3f}", file=sys.stderr)
    dg = dist_general
    print(f"[diag] GENERAL max_all  pos.med={dg['max_all_pos']['median']:.3f}"
          f" neg.med={dg['max_all_neg']['median']:.3f}"
          f"  frac_neg>pos_med={dg['frac_neg_max_above_pos_median_max']:.3f}",
          file=sys.stderr)
    print(f"[diag] GENERAL corr(max_all, n_tok)={dg['corr_max_all_vs_ntok']:.3f}"
          f" (neg-only {dg['corr_max_all_vs_ntok_neg']:.3f})", file=sys.stderr)
    print(f"[diag] length: n_tok pos.med={length_block['n_tok_pos']['median']:.0f}"
          f" neg.med={length_block['n_tok_neg']['median']:.0f}"
          f"  length_ntok_auc={length_block['length_ntok_auc']:.3f}", file=sys.stderr)
    print(f"[diag] wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
