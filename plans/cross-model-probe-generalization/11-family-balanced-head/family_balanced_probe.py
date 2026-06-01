# [ai-generated]
"""Exp-11 (Tier-4 #7): can a SINGLE general probe with FAMILY-BALANCED sampling
hold BOTH vuln families (injection + memory) at once?

Background: the standard general probe (exp-09 `--sampler none`, the linear
floor) under-allocates capacity to the memory family — on honest `tokens_code`
it scores ~0.88 on injection but ~chance on memory. The fit set is dominated by
injection positives, so a single linear direction that maximizes overall token
AUC essentially ignores memory. This runner tests the cheapest fix: rebalance
the FIT set so memory-family positives are not drowned out, train ONE probe, and
re-measure per-family `tokens_code` AUC head-to-head against the unbalanced
general probe on the IDENTICAL test pools.

The lead expects at most a minor lift (or even a slight drop) — this is a
RULE-OUT. Correctness and an apples-to-apples comparison matter more than
winning. Everything except the fit-set sampler is held IDENTICAL to exp-09's
`train_head_baseline.py` (same group-aware test split, same 15% VAL carve at
VAL_SEED=42, same `honest_token_aucs` eval, same linear head): the overall /
by_lang / by_cwe blocks are computed exactly as in 09 so those cells stay
directly comparable, and we add a per-family block mirroring exp-10's pooling.

Sampler:
  none             -> plain general fit. MUST reproduce the 09 linear baseline
                      EXACTLY (Qwen2.5-Coder 0.788, Qwen3-32B 0.806,
                      Qwen3.6-27B 0.787, gemma-3-27b 0.770). Harness sanity check.
  family_balanced  -> oversample memory-family POSITIVE examples by an integer
                      factor k = min(round(n_inj / n_mem), 8) before fitting.

CWE -> family map is VERBATIM from 10-per-cwe-probes/per_cwe_probe.py (the
canonical source) so the head-to-head with exp-10 is exact. cwe != null
<=> label == 1 in this dataset; negatives = rows with cwe == null.

Multi-task per-family-head variant (a 2-logit head with a per-family loss) is
the PLANNED ESCALATION if family-balanced sampling fails to lift memory. It is
NOT a drop-in (needs a custom factory + custom loss) and is deliberately NOT
implemented here. See EXPERIMENT.md.
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

from src.training.train_probe_spanmax import train_one_layer  # noqa: E402
from src.eval.honest_scoring import (  # noqa: E402
    honest_token_aucs, load_dataset_rows, load_offsets_npz,
)
from sklearn.metrics import roc_auc_score  # noqa: E402

VAL_FRAC = 0.15
VAL_SEED = 42

# CWE -> family. VERBATIM from 10-per-cwe-probes/per_cwe_probe.py — kept IDENTICAL
# to exp-10 for a head-to-head comparison. TODO(adhoc-decision): CWE-190 ->
# injection is a judgement call carried over from exp-10 (integer-overflow is C
# and often a memory-safety precursor, but sweep-6 grouped the taint/data-flow
# CWEs as the injection set). Change in BOTH places if the lead re-decides.
FAMILY = {
    "CWE-089": "injection", "CWE-078": "injection", "CWE-022": "injection",
    "CWE-079": "injection", "CWE-190": "injection",
    "CWE-125": "memory", "CWE-476": "memory", "CWE-416": "memory",
    "CWE-787": "memory",
}

# Cap on the memory oversampling factor: even when injection out-numbers memory
# by >8x we never duplicate a memory example more than 8 times — beyond that the
# fit set is dominated by ~54 distinct memory examples copied verbatim, which
# overfits to those specific examples without adding real signal.
MAX_OVERSAMPLE_K = 8

# A per-family test AUC below this many positive EXAMPLES is flagged
# untrustworthy. Mirrors exp-10's MIN_TRUST_POS (== breakdown's --min-cwe-pos).
MIN_TRUST_POS = 10


def _load_train_eval():
    p = REPO / "src" / "remotes" / "the cluster" / "train_eval.py"
    spec = importlib.util.spec_from_file_location("remote_train_eval", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _auc(yv, pv) -> float:
    return float(roc_auc_score(yv, pv)) if len(np.unique(yv)) > 1 else float("nan")


def _factory_for(head: str):
    """head -> probe_factory (None == LinearProbe). exp-11 fixes head=linear, but
    keep the 09 shape so the call site is identical and a future mlp sweep is a
    one-line change."""
    if head == "linear":
        return None
    raise ValueError(f"exp-11 fixes --head linear, got {head!r}")


def _family_of(rows, eid):
    """injection | memory | None. None == negative (cwe == null) or a CWE not in
    the family map. Reads the dataset row straight (same as exp-10 cwe_of)."""
    return FAMILY.get(rows[eid].get("cwe"))


def family_balanced_resample(Xfit, yfit, efit, rows, *, max_k=MAX_OVERSAMPLE_K):
    """Oversample memory-family POSITIVE examples in the FIT set so memory is not
    drowned out by injection.

    Counts positive EXAMPLES per family by eid (NOT tokens), computes the integer
    oversample factor k = min(round(n_inj / n_mem), max_k), and duplicates each
    memory-positive eid's token-block (k - 1) extra times. Each duplicate copy
    gets a FRESH SYNTHETIC eid (orig_eid + C * copy_idx, C > max observed eid) so
    `_group_by_example` / `honest_token_aucs` treat copies as distinct examples
    and never collide with real eids. Duplicates only ever enter FIT.

    Immutability: inputs are never mutated; fresh arrays are built and returned.

    Returns (Xbal, ybal, ebal, info) where info records the family counts and k.
    """
    Xfit = np.asarray(Xfit)
    yfit = np.asarray(yfit)
    efit = np.asarray(efit)

    # Distinct fit eids and their family (positives only have a family).
    fit_eids_unique = np.unique(efit)
    inj_eids = [int(e) for e in fit_eids_unique
                if _family_of(rows, int(e)) == "injection"]
    mem_eids = [int(e) for e in fit_eids_unique
                if _family_of(rows, int(e)) == "memory"]
    n_inj, n_mem = len(inj_eids), len(mem_eids)

    info = {
        "n_inj_fit_examples": n_inj,
        "n_mem_fit_examples": n_mem,
        "max_oversample_k": int(max_k),
    }

    if n_mem == 0 or n_inj == 0:
        # No imbalance to correct (or no memory positives at all) — return fit
        # unchanged. k = 1 means "no oversampling applied".
        info["oversample_k"] = 1
        info["n_mem_eids_duplicated"] = 0
        return (np.array(Xfit, copy=True), np.array(yfit, copy=True),
                np.array(efit, copy=True), info)

    k = min(int(round(n_inj / n_mem)), int(max_k))
    k = max(k, 1)
    info["oversample_k"] = k

    if k <= 1:
        info["n_mem_eids_duplicated"] = 0
        return (np.array(Xfit, copy=True), np.array(yfit, copy=True),
                np.array(efit, copy=True), info)

    # C must exceed every real eid so synthetic eids can't collide with reals or
    # with each other across copy indices. efit holds only fit eids, but real
    # eids range over the whole dataset — base C on the global max + 1.
    max_eid = int(fit_eids_unique.max())
    C = max_eid + 1

    mem_set = set(mem_eids)
    # Boolean mask of fit tokens that belong to a memory-positive example.
    mem_tok_mask = np.fromiter((int(e) in mem_set for e in efit), bool, len(efit))

    X_dup_blocks = [Xfit]
    y_dup_blocks = [yfit]
    e_dup_blocks = [efit]
    # copy_idx 1..k-1 are the EXTRA copies (copy 0 is the original already in
    # efit). Each extra copy shifts memory eids by C * copy_idx -> a fresh,
    # non-colliding synthetic eid block.
    for copy_idx in range(1, k):
        X_dup_blocks.append(np.array(Xfit[mem_tok_mask], copy=True))
        y_dup_blocks.append(np.array(yfit[mem_tok_mask], copy=True))
        e_dup_blocks.append(efit[mem_tok_mask] + C * copy_idx)

    Xbal = np.concatenate(X_dup_blocks, axis=0)
    ybal = np.concatenate(y_dup_blocks, axis=0)
    ebal = np.concatenate(e_dup_blocks, axis=0)

    info["n_mem_eids_duplicated"] = n_mem
    info["n_synthetic_copies_per_eid"] = k - 1
    info["n_fit_tokens_before"] = int(len(efit))
    info["n_fit_tokens_after"] = int(len(ebal))
    info["synthetic_eid_offset_C"] = int(C)
    return Xbal, ybal, ebal, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--best-layer", type=int, required=True)
    ap.add_argument("--sampler", choices=("none", "family_balanced"),
                    default="family_balanced",
                    help="none = plain general fit (reproduces the 09 linear "
                         "baseline EXACTLY — harness sanity check). "
                         "family_balanced = oversample memory-family positive "
                         "examples by k=min(round(n_inj/n_mem),8). "
                         "TODO(adhoc-decision): oversample-positives vs full "
                         "class-balance vs token-weight — defaulting to "
                         "memory-positive oversampling as the least destructive "
                         "to injection. The lead settles this fork.")
    ap.add_argument("--head", default="linear", choices=("linear",),
                    help="exp-11 fixes the linear head (the floor we are trying "
                         "to lift on memory). Kept as a flag for parity with 09.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--offsets", default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--model", default="")
    ap.add_argument("--min-cwe-pos", type=int, default=10)
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        print(f"[fam11] {out} exists, skipping", file=sys.stderr)
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

    # --- group-aware 15% VAL carve of TRAIN (exact parity with 09) ---
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
    print(f"[fam11] {args.model} L{args.best_layer} sampler={args.sampler} "
          f"fit_tok={fit.sum()} val_tok={val.sum()} test_tok={te.sum()}",
          file=sys.stderr)

    Xmm = np.load(acts / f"layer_{args.best_layer:02d}.npy", mmap_mode="r")
    Xfit = np.asarray(Xmm[fit], dtype=np.float32)
    yfit = y[fit]
    efit = eids[fit]
    if len(np.unique(yfit)) < 2 or te.sum() == 0 or val.sum() == 0:
        out.write_text(json.dumps({"layer": args.best_layer, "sampler": args.sampler,
                                   "skipped": "degenerate labels/splits"}))
        return
    if not np.isfinite(Xfit).all():
        out.write_text(json.dumps({"layer": args.best_layer, "sampler": args.sampler,
                                   "error": "non-finite activations"}))
        return

    # --- the ONLY divergence from exp-09: rebalance the fit set ---
    if args.sampler == "family_balanced":
        Xfit_used, yfit_used, efit_used, sampler_info = family_balanced_resample(
            Xfit, yfit, efit, rows)
        oversample_k = sampler_info["oversample_k"]
    else:
        # `none` must be byte-for-byte the 09 fit -> reproduces the linear floor.
        Xfit_used, yfit_used, efit_used = Xfit, yfit, efit
        sampler_info = {"oversample_k": 1}
        oversample_k = 1

    factory = _factory_for(args.head)
    r = train_one_layer(Xfit_used, yfit_used, efit_used, epochs=args.epochs,
                        device=device, verbose=False, probe_factory=factory)
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

    # Overall / by_lang / by_cwe: IDENTICAL to exp-09 (apples-to-apples).
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

    # Per-family eval — pooling mirrors exp-10 EXACTLY: each family's pos pool is
    # {test positives whose CWE maps to that family} and the neg pool is ALL
    # cwe==null test negatives (same neg pool as the per-CWE breakdown), so the
    # number is the trustworthy ~200-injection / ~54-memory family AGGREGATE.
    by_family = {}
    for fam in ("memory", "injection"):
        pos_eids = {e for e in test_list if _family_of(rows, e) == fam}
        s = subset(pos_eids | neg_eids)
        if s is None:
            by_family[fam] = {"n_pos_examples": len(pos_eids),
                              "trust": len(pos_eids) >= MIN_TRUST_POS,
                              "error": "empty positive/negative pool"}
            continue
        s["n_pos_examples"] = len(pos_eids)
        s["trust"] = len(pos_eids) >= MIN_TRUST_POS
        by_family[fam] = s

    ex_ids, ex_p = te_mod.example_scores(tok_p, te_e)
    ex_y = np.array([int(y[(eids == e)].max() > 0) for e in ex_ids])

    rec = {
        "model": args.model, "layer": args.best_layer, "head": args.head,
        "sampler": args.sampler, "oversample_k": oversample_k,
        "sampler_info": sampler_info,
        "val_tokens_code_auc": val_h["tokens_code_auc"],
        "val_tokens_auc": val_h["tokens_auc"],
        "overall": overall, "by_lang": by_lang, "by_cwe": by_cwe,
        "by_family": by_family,
        "test_ex_auc": _auc(ex_y, ex_p), "n_test_ex": int(len(ex_ids)),
    }
    out.write_text(json.dumps(rec, indent=2))
    ov = overall["tokens_code_auc"] if overall else float("nan")
    mem = by_family.get("memory", {}).get("tokens_code_auc", float("nan"))
    inj = by_family.get("injection", {}).get("tokens_code_auc", float("nan"))
    print(f"[fam11] {args.model} L{args.best_layer} sampler={args.sampler} k={oversample_k} "
          f"val_tc={val_h['tokens_code_auc']:.3f} test_tc={ov:.3f} "
          f"mem={mem:.3f} inj={inj:.3f}", file=sys.stderr)


if __name__ == "__main__":
    main()
