# [ai-generated]
"""Reduce exp-16 per-token logit npz -> tiny per-example JSON (compute-near-data).

The exp-16 logits_layer{NN}.npz are ~7 MB each and don't transfer over the
cluster's file-transfer API. This reducer runs WHERE the npz live (the GPU host), reading
each model's operating-layer npz and emitting a few-KB `lasttok_reduced.json`
with, per example: the probe LOGIT at the final live-code token (`last`), the
max/mean code-only logit (`max`/`mean`), the positive-code-token count, and a
`has_code` flag — plus the scalar `tokens_code_auc` (numpy rank-AUC) that
score_last_token.py hard-gates against exp-16's stored value.

numpy-only (no sklearn/torch). Point --runs-dir at the dir holding the
logitdump_<slug>/ subdirs.
"""
from __future__ import annotations
import argparse
import json
import os

import numpy as np

# slug -> operating layer (blog headline layer)
JOBS = {
    "logitdump_Qwen_Qwen2.5-Coder-7B-Instruct": 16,
    "logitdump_google_gemma-3-4b-it": 7,
    "logitdump_google_gemma-3-12b-it": 15,
    "logitdump_google_gemma-3-12b-pt": 13,
    "logitdump_google_gemma-3-27b-it": 19,
}


def rank_auc(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float)
    n1 = (y == 1).sum(); n0 = (y == 0).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort"); sr = s[order]
    ranks = np.empty(len(s), float); i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sr[j + 1] == sr[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def reduce_one(npz_path):
    z = np.load(npz_path)
    logit, prob, y = z["logit"], z["prob"], z["y"]
    eids = z["example_id"]; is_code = z["is_code"].astype(bool); is_test = z["is_test"].astype(bool)
    m = is_test & is_code
    tok_code_auc = rank_auc(y[m], prob[m])
    per = {}
    for e in np.unique(eids):
        idx = np.where(eids == e)[0]
        assert idx[-1] - idx[0] + 1 == len(idx), f"eid {e} token rows not contiguous"
        cm = is_code[idx]; sel = idx[cm] if cm.any() else idx
        per[str(int(e))] = {"last": float(logit[sel[-1]]), "max": float(logit[sel].max()),
                            "mean": float(logit[sel].mean()), "npos": int((y[sel] > 0).sum()),
                            "hc": bool(cm.any())}
    return tok_code_auc, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True, help="dir holding logitdump_<slug>/ subdirs")
    args = ap.parse_args()
    for slug, layer in JOBS.items():
        p = os.path.join(args.runs_dir, slug, f"logits_layer{layer:02d}.npz")
        if not os.path.exists(p):
            print("MISSING", p, flush=True); continue
        auc, per = reduce_one(p)
        op = os.path.join(args.runs_dir, slug, "lasttok_reduced.json")
        with open(op, "w") as f:
            json.dump({"slug": slug, "layer": layer, "tokens_code_auc": auc, "per_example": per}, f)
        print("WROTE", op, "tok_code_auc=%.5f" % auc, "n_ex", len(per), flush=True)


if __name__ == "__main__":
    main()
