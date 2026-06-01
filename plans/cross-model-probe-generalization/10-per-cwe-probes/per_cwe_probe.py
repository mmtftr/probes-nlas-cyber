# [ai-generated]
"""Exp-10: per-CWE specialized probes vs the GENERAL probe, per-CWE.

For one model at its single best layer (cached acts), this trains:
  - the GENERAL probe: span-max probe on ALL fit tokens (same recipe as
    06/breakdown_lang_cwe.py) — the existing ~0.788 / per-CWE breakdown probe.
  - a per-CWE SPECIALIZED probe for each requested CWE (or CWE-family): span-max
    probe on {CWE-X positives} ∪ {a negative pool} only.

Both are evaluated on the IDENTICAL per-CWE test subset
({CWE-X test positives} ∪ {negative test pool}) so the head-to-head AUC is
apples-to-apples with the general 06 breakdown numbers. The honest metric is
`tokens_code_auc` (live-code-only token AUC), via src/eval/honest_scoring.

Splits are the EXACT group-aware splits used by train_all_layers.py:
  - test = persisted seed-42 20% group hold-out (load_or_make_split).
  - a 15% group-aware VAL carve of TRAIN (VAL_SEED=42) is held out and NOT used
    to fit (kept for parity with 06 — selection-val tokens are excluded from
    every fit, general and specialized, so no probe sees them).
  - fit = the remaining train groups; per-CWE fit = fit ∩ (CWE-X pos ∪ neg pool).

Because cwe != null ⟺ label == 1 in this dataset (verified: 715 positives all
carry a CWE, 715 negatives all have cwe == null), "negatives" = all rows with
cwe == null. There is no per-CWE negative ambiguity at the row level.

CWE families (lead's sweep-6 grouping):
  injection-class (data-flow / taint): CWE-089 SQLi, CWE-078 cmd-inj,
      CWE-022 path-traversal, CWE-079 XSS, CWE-190 int-overflow*.
  memory-safety: CWE-125 OOB-read, CWE-476 NULL-deref, CWE-416 UAF,
      CWE-787 OOB-write.
  (*CWE-190 integer-overflow is C and often a memory-safety precursor; it is
   listed injection-class here only because sweep-6 grouped the taint/data-flow
   CWEs as the "detected" set. TODO(adhoc-decision) below: family assignment of
   CWE-190 is a judgement call — the lead should confirm.)

Output JSON (one per model) carries, per CWE / family:
  n_train_pos, n_test_pos, n_neg_test (scarcity), and the head-to-head
  {general tokens_code_auc} vs {specialized tokens_code_auc} on that subset,
  plus a `trust` flag (False when n_test_pos < MIN_TRUST_POS).
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

from src.training.train_probe_spanmax import train_one_layer, LinearProbe, MLPProbe  # noqa: E402
from src.eval.honest_scoring import (  # noqa: E402
    honest_token_aucs, load_dataset_rows, load_offsets_npz,
)

# Parity with train_all_layers.py: 15% group-aware VAL carve of TRAIN, seed 42.
# We don't *select* on it here (single best layer is fixed), but we exclude its
# tokens from every fit so general/specialized probes see the same fit pool the
# 06 layer-selection pipeline produced.
VAL_FRAC = 0.15
VAL_SEED = 42

# A per-CWE test AUC below this many positive examples is flagged untrustworthy.
# 5 pos (e.g. CWE-787 test=5, CWE-190 test=4) gives a CI half-width well over
# ±0.15 on AUC — report but do not draw conclusions. TODO(adhoc-decision): the
# threshold (10? 15?) is the lead's call; 10 mirrors breakdown's --min-cwe-pos.
MIN_TRUST_POS = 10

# CWE -> family. TODO(adhoc-decision): CWE-190 family is debatable (see module
# docstring). The lead decides; this map is the only place to change it.
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


def _probe_factory(head: str):
    """Return a probe_factory for train_one_layer, or None for the linear default."""
    if head == "linear":
        return None  # train_one_layer defaults to LinearProbe and returns (w, b)
    if head == "mlp":
        return lambda d: MLPProbe(d)
    raise ValueError(f"--head must be 'linear' or 'mlp', got {head!r}")


def _score_tokens(probe_result, Xte: np.ndarray) -> np.ndarray:
    """Per-token sigmoid prob from a train_one_layer result on test activations.

    Linear heads expose (w, b); non-linear heads expose the torch module under
    "probe" with w=None. We score uniformly via the module to support both,
    falling back to the explicit (w, b) for the linear case (cheap, no torch
    forward needed). Never mutates Xte."""
    import torch
    if probe_result.get("w") is not None:
        w = np.asarray(probe_result["w"], np.float32)
        b = float(probe_result["b"])
        return 1.0 / (1.0 + np.exp(-(Xte @ w + b)))
    probe = probe_result["probe"].eval()
    with torch.no_grad():
        logits = probe(torch.from_numpy(np.asarray(Xte, np.float32)))
        return torch.sigmoid(logits).cpu().numpy()


def _fit_probe(Xmm, fit_mask, y, eids, *, epochs, head, device):
    """Fit one span-max probe on the tokens selected by fit_mask. Returns the
    train_one_layer result dict (immutable inputs; builds float32 fit copies)."""
    Xfit = np.asarray(Xmm[fit_mask], np.float32)
    return train_one_layer(
        Xfit, y[fit_mask], eids[fit_mask],
        epochs=epochs, device=device, verbose=False,
        probe_factory=_probe_factory(head),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--best-layer", type=int, required=True)
    ap.add_argument("--cwe", default="ALL",
                    help="A single CWE id (CWE-125), a family "
                         "(injection|memory), or ALL (every CWE with >= "
                         "--min-train-pos train positives).")
    ap.add_argument("--head", choices=("linear", "mlp"), default="linear")
    ap.add_argument("--neg-pool", choices=("all", "same_lang", "same_family"),
                    default="all",
                    help="Which negatives a SPECIALIZED probe trains/tests "
                         "against. TODO(adhoc-decision): the lead picks this. "
                         "'all' = every cwe==null negative (default, matches the "
                         "06 general breakdown so head-to-head is exact). "
                         "'same_lang' = negatives whose row lang matches the "
                         "CWE's dominant language. 'same_family' = negatives "
                         "whose paired-positive family matches (NOTE: negatives "
                         "carry no CWE, so 'same_family' falls back to "
                         "'same_lang' here and is flagged in output).")
    ap.add_argument("--out", required=True)
    ap.add_argument("--offsets", default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--model", default="")
    ap.add_argument("--min-train-pos", type=int, default=10,
                    help="For --cwe ALL: skip CWEs with fewer train positives.")
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    te_mod = _load_train_eval()
    acts = Path(args.acts_dir)
    offsets_by_eid = load_offsets_npz(
        Path(args.offsets) if args.offsets else acts / "offsets.npz")
    rows_by_eid = load_dataset_rows(Path(args.dataset))

    y = np.load(acts / "y.npy")
    eids = np.load(acts / "example_ids.npy")
    rows, train_eids, test_eids = te_mod.load_or_make_split(
        Path(args.dataset), Path(args.split))

    # --- group-aware 15% VAL carve of TRAIN (exact parity with 06) ---
    tr_eid_to_group = {e: te_mod.pair_group_key(rows[e]) for e in train_eids}
    tr_groups = sorted(set(tr_eid_to_group.values()))
    rng = np.random.default_rng(VAL_SEED)
    rng.shuffle(tr_groups)
    n_val = max(1, int(round(VAL_FRAC * len(tr_groups))))
    val_groups = set(tr_groups[:n_val])
    val_eids = {e for e, g in tr_eid_to_group.items() if g in val_groups}
    fit_eids = train_eids - val_eids  # GENERAL fit pool (eids)

    # --- helpers reading CWE / lang straight from dataset rows (reuse 06 logic) ---
    def cwe_of(e):
        return rows[e].get("cwe")

    def lang_of(e):
        return (rows[e].get("lang") or "").lower()

    neg_fit = {e for e in fit_eids if not cwe_of(e)}      # all negative fit eids
    neg_test = {e for e in test_eids if not cwe_of(e)}    # all negative test eids

    # CWEs to process.
    train_pos_counts = Counter(cwe_of(e) for e in fit_eids if cwe_of(e))
    if args.cwe == "ALL":
        targets = [c for c, n in train_pos_counts.most_common()
                   if n >= args.min_train_pos]
    elif args.cwe in ("injection", "memory"):
        # ONE pooled family probe over ALL the family's positives (NOT a loop over
        # the individual member CWEs) — the trustworthy ~200-positive family test.
        targets = [args.cwe]
    else:
        targets = [args.cwe]

    Xmm = np.load(acts / f"layer_{args.best_layer:02d}.npy", mmap_mode="r")

    # --- GENERAL probe: trained ONCE on all fit tokens (the 06 baseline) ---
    fit_mask_general = np.fromiter((int(e) in fit_eids for e in eids), bool, len(eids))
    gen_res = _fit_probe(Xmm, fit_mask_general, y, eids,
                         epochs=args.epochs, head=args.head, device=device)

    def neg_pool_for(target_cwe_or_family, pos_eids_train, pos_eids_test):
        """Resolve the negative pool for a specialized probe under --neg-pool.

        Returns (neg_fit_eids, neg_test_eids, note). TODO(adhoc-decision): the
        negative-pool fork is the lead's to settle; 'all' is the default and the
        only one that keeps the per-CWE head-to-head exactly comparable to the
        06 general breakdown (which uses ALL negatives)."""
        if args.neg_pool == "all":
            return neg_fit, neg_test, None
        # Dominant language of the target's positives (train side).
        langs = Counter(lang_of(e) for e in pos_eids_train)
        dom = langs.most_common(1)[0][0] if langs else ""
        nf = {e for e in neg_fit if lang_of(e) == dom}
        nt = {e for e in neg_test if lang_of(e) == dom}
        note = f"neg_pool=same_lang(dom={dom})"
        if args.neg_pool == "same_family":
            # Negatives carry no CWE -> no family signal; fall back to same_lang.
            note = (f"neg_pool=same_family requested but negatives have no CWE; "
                    f"fell back to same_lang(dom={dom})")
        return nf, nt, note

    by_cwe = {}
    for cwe in targets:
        if cwe in ("injection", "memory"):
            # Pooled family probe: every positive whose CWE maps to this family.
            pos_fit = {e for e in fit_eids if FAMILY.get(cwe_of(e)) == cwe}
            pos_test = {e for e in test_eids if FAMILY.get(cwe_of(e)) == cwe}
            fam = cwe
        else:
            pos_fit = {e for e in fit_eids if cwe_of(e) == cwe}
            pos_test = {e for e in test_eids if cwe_of(e) == cwe}
            fam = FAMILY.get(cwe, "?")
        nf, nt, note = neg_pool_for(cwe, pos_fit, pos_test)
        spec_fit_eids = pos_fit | nf
        eval_eids = pos_test | nt  # IDENTICAL subset for general + specialized

        rec = {
            "family": fam,
            "n_train_pos": len(pos_fit),
            "n_test_pos": len(pos_test),
            "n_neg_fit": len(nf),
            "n_neg_test": len(nt),
            "trust": len(pos_test) >= MIN_TRUST_POS,
            "neg_pool_note": note,
        }
        if not pos_test or not nt or not pos_fit:
            rec["error"] = "empty positive or negative pool in fit/test"
            by_cwe[cwe] = rec
            continue

        # Eval masks over the flat token arrays.
        eval_mask = np.fromiter((int(e) in eval_eids for e in eids), bool, len(eids))
        Xte = np.asarray(Xmm[eval_mask], np.float32)
        te_e, te_y = eids[eval_mask], y[eval_mask]

        def honest_tc(token_probs):
            h = honest_token_aucs(token_probs, te_y, te_e, offsets_by_eid, rows_by_eid)
            return h["tokens_code_auc"], h["n_pos_code"], h["n_total_code"]

        # General probe on this subset.
        gen_p = _score_tokens(gen_res, Xte)
        g_tc, g_npos, g_ntot = honest_tc(gen_p)

        # Specialized probe: fit on CWE-X positives ∪ neg pool, then score subset.
        spec_fit_mask = np.fromiter((int(e) in spec_fit_eids for e in eids),
                                    bool, len(eids))
        spec_res = _fit_probe(Xmm, spec_fit_mask, y, eids,
                              epochs=args.epochs, head=args.head, device=device)
        spec_p = _score_tokens(spec_res, Xte)
        s_tc, s_npos, s_ntot = honest_tc(spec_p)

        rec.update({
            "general_tokens_code_auc": g_tc,
            "specialized_tokens_code_auc": s_tc,
            "delta_spec_minus_gen": (s_tc - g_tc)
            if (s_tc == s_tc and g_tc == g_tc) else float("nan"),
            "n_pos_code_eval": g_npos, "n_total_code_eval": g_ntot,
            "n_spec_fit_tokens": int(spec_fit_mask.sum()),
        })
        by_cwe[cwe] = rec
        tc_g = f"{g_tc:.3f}" if g_tc == g_tc else "nan"
        tc_s = f"{s_tc:.3f}" if s_tc == s_tc else "nan"
        print(f"[per-cwe] {args.model} {cwe} ({rec['family']}) "
              f"n_test_pos={rec['n_test_pos']} trust={rec['trust']} "
              f"gen={tc_g} spec={tc_s}", file=sys.stderr)

    out = {
        "model": args.model,
        "layer": args.best_layer,
        "head": args.head,
        "neg_pool": args.neg_pool,
        "cwe_arg": args.cwe,
        "min_trust_pos": MIN_TRUST_POS,
        "n_neg_fit_all": len(neg_fit),
        "n_neg_test_all": len(neg_test),
        "by_cwe": by_cwe,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[per-cwe] wrote {args.out} ({len(by_cwe)} CWEs)", file=sys.stderr)


if __name__ == "__main__":
    main()
