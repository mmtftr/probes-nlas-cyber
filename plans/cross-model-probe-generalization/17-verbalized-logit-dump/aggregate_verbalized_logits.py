# [ai-generated]
"""Merge the per-GPU verbalized-logit shards into one self-describing artifact.

Reads `verbalized_logits.gpu{0..3}.npz` (written by verbalized_logit_dump.py),
concatenates them in eid order, attaches `is_test` (from the SAME seed-42
group-clean split exp-16 used, loaded VERBATIM from load_or_make_split) plus the
per-row `cwe` / `lang`, computes the verbalized example-AUC (all 1430 examples and
test-only), and writes:

  - logits_verbalized.npz        — merged columns: eid, label, is_test, p_yes,
        yes_lp, no_lp, margin, yes_logits_raw[N,|yes|], no_logits_raw[N,|no|],
        topk_ids[N,K], topk_logits[N,K], plus yes_ids/no_ids metadata.
  - example_scores_verbalized.json — [{eid, p_yes, label, cwe, lang, is_test,
        yes_lp, no_lp, margin}]  (human-readable mirror of exp-16's example_scores)
  - metrics_verbalized_logits.json — {model, n_examples, n_test, verbalized_auc_all,
        verbalized_auc_test, yes/no token decodes, question, framing, hist_ref gate}

Correctness gate: for the two models exp-05 reported, `verbalized_auc_test` should
sit near the documented historical verbalized AUC (gemma-3-27b-it 0.554,
Qwen2.5-Coder-32B-Instruct 0.632). exp-05 averaged that over 5 seeds' test
intersections; this reports the single seed-42 test set, so small drift is
expected — a gross miss (>~0.05) signals a reproduction problem, not noise.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from sklearn.metrics import roc_auc_score  # noqa: E402
from src.remotes.the cluster.train_eval import load_or_make_split  # noqa: E402


# Documented exp-05 historical verbalized example-AUC (before/after rebuild,
# 2026-05-31; mean over 5 seeds' test intersections). Sanity reference only.
HIST_VERBALIZED_AUC = {
    "google/gemma-3-27b-it": 0.554,
    "Qwen/Qwen2.5-Coder-32B-Instruct": 0.632,
}

QUESTION = (
    "Does the code above contain a security vulnerability? "
    'Respond with ONLY one word — "yes" or "no" — and nothing else.'
)


def _auc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--shards", required=True, help="dir holding verbalized_logits.gpu*.npz")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True, help="output dir for merged artifacts")
    args = ap.parse_args()

    shards_dir = Path(args.shards)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    files = sorted(shards_dir.glob("verbalized_logits.gpu*.npz"))
    if not files:
        raise SystemExit(f"[aggregate] no verbalized_logits.gpu*.npz under {shards_dir}")

    cols = {k: [] for k in ("eid", "label", "p_yes", "yes_lp", "no_lp", "margin",
                            "yes_logits_raw", "no_logits_raw", "topk_ids", "topk_logits")}
    yes_ids = no_ids = None
    for f in files:
        npz = np.load(f)
        for k in cols:
            cols[k].append(npz[k])
        yi, ni = npz["yes_ids"], npz["no_ids"]
        if yes_ids is None:
            yes_ids, no_ids = yi, ni
        elif not (np.array_equal(yes_ids, yi) and np.array_equal(no_ids, ni)):
            raise SystemExit(f"[aggregate] yes/no id sets differ across shards ({f})")

    merged = {k: np.concatenate(v, axis=0) for k, v in cols.items()}
    order = np.argsort(merged["eid"], kind="stable")
    merged = {k: v[order] for k, v in merged.items()}
    eids = merged["eid"]

    # exp-16's split, loaded the same way, so is_test lines up token-for-token.
    rows, train_eids, test_eids = load_or_make_split(Path(args.dataset), Path(args.split))
    test_set = set(int(e) for e in test_eids)
    if len(eids) != len(rows):
        print(f"[aggregate] WARNING: {len(eids)} scored != {len(rows)} dataset rows "
              "(a shard gap?) — proceeding over the scored intersection", file=sys.stderr)

    is_test = np.fromiter((int(e) in test_set for e in eids), bool, len(eids))
    cwe = [rows[int(e)].get("cwe") for e in eids]
    lang = [rows[int(e)].get("lang") for e in eids]

    label, p_yes = merged["label"], merged["p_yes"]
    auc_all = _auc(label, p_yes)
    auc_test = _auc(label[is_test], p_yes[is_test])

    np.savez_compressed(
        out / "logits_verbalized.npz",
        eid=eids, label=merged["label"], is_test=is_test,
        p_yes=merged["p_yes"], yes_lp=merged["yes_lp"], no_lp=merged["no_lp"],
        margin=merged["margin"], yes_logits_raw=merged["yes_logits_raw"],
        no_logits_raw=merged["no_logits_raw"], topk_ids=merged["topk_ids"],
        topk_logits=merged["topk_logits"], yes_ids=yes_ids, no_ids=no_ids)

    ex_records = [
        {"eid": int(e), "p_yes": float(p), "label": int(l), "cwe": c, "lang": lg,
         "is_test": bool(t), "yes_lp": float(yl), "no_lp": float(nl), "margin": float(mg)}
        for e, p, l, c, lg, t, yl, nl, mg in zip(
            eids, p_yes, label, cwe, lang, is_test,
            merged["yes_lp"], merged["no_lp"], merged["margin"])]
    (out / "example_scores_verbalized.json").write_text(json.dumps(ex_records))

    hist = HIST_VERBALIZED_AUC.get(args.model)
    gate = None
    if hist is not None:
        gate = {"hist_verbalized_auc_5seed_testmean": hist,
                "this_verbalized_auc_test_seed42": auc_test,
                "delta": round(auc_test - hist, 4),
                "ok_within_0.05": bool(abs(auc_test - hist) <= 0.05)}

    summary = {
        "model": args.model,
        "n_examples": int(len(eids)),
        "n_test": int(is_test.sum()),
        "n_train": int((~is_test).sum()),
        "verbalized_auc_all": auc_all,
        "verbalized_auc_test": auc_test,
        "yes_ids": [int(i) for i in yes_ids],
        "no_ids": [int(i) for i in no_ids],
        "question": QUESTION,
        "framing": "code-before-question, neutral preamble (matches exp-05)",
        "hist_gate": gate,
    }
    (out / "metrics_verbalized_logits.json").write_text(json.dumps(summary, indent=2))
    msg = (f"[aggregate] {args.model}: n={len(eids)} (test={int(is_test.sum())}) "
           f"verbalized_auc_all={auc_all:.3f} verbalized_auc_test={auc_test:.3f}")
    if gate is not None:
        msg += f"  [gate hist={hist:.3f} Δ={gate['delta']:+.3f} ok={gate['ok_within_0.05']}]"
    print(msg, file=sys.stderr)


if __name__ == "__main__":
    main()
