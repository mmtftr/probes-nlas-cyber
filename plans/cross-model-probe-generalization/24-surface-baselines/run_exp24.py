# [ai-generated]
"""exp-24 — run all four eval designs, surface baselines vs the Qwen-32B probe.

Headline metric = pooled `tokens_code_auc` (live-code tokens). Secondary =
per-example-mean AUC. Bootstrap-over-examples 95% CIs. n per cell reported;
cells with <10 test vuln examples flagged untrusted.

Substrate = exp-16 L25 dump (probe token axis); see substrate.py / EXPERIMENT.md.
Outputs: results/design{1,2,3,4}_*.json.
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from substrate import load_substrate, INJ
import features as F

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
MIN_TRUST = 10
SEED = 42


def log(*a):
    print("[exp24]", *a, file=sys.stderr, flush=True)


def auc(y, sc):
    return float(roc_auc_score(y, sc)) if len(np.unique(y)) > 1 else float("nan")


# ----- substrate + per-eid token index maps -----
S = load_substrate()
N = len(S.y)
CODE_BY_EID: dict[int, np.ndarray] = {}
_tmp = defaultdict(list)
for i in range(N):
    if S.is_code[i]:
        _tmp[int(S.eid[i])].append(i)
for e, ix in _tmp.items():
    CODE_BY_EID[e] = np.asarray(ix, np.int64)

TRAIN_EX = sorted({int(e) for e in S.eid[~S.is_test]})
TEST_EX = sorted({int(e) for e in S.eid[S.is_test]})
CLEAN_TRAIN = [e for e in TRAIN_EX if S.clean_ex[e]]
CLEAN_TEST = [e for e in TEST_EX if S.clean_ex[e]]


def code_toks(eids):
    if not eids:
        return np.array([], np.int64)
    return np.concatenate([CODE_BY_EID[e] for e in eids if e in CODE_BY_EID])


def lang_filter(idx, langs):
    return idx[np.isin(S.lang[idx], langs)]


# ----- generic pooled eval + bootstrap over examples -----
def pool_eval(score, pos_eids, neg_eids, n_boot=1000, rng=None, restrict_lang=None):
    """Eval pool = code tokens of (pos_eids ∪ neg_eids); per-token label = S.y.
    Returns pooled tokens_code_auc + per-example-mean AUC, each with bootstrap CI.
    `restrict_lang` (tuple) keeps only tokens whose file language is in the set."""
    if rng is None:
        rng = np.random.default_rng(SEED)
    eids = list(pos_eids) + list(neg_eids)
    e2t = {}
    for e in eids:
        t = CODE_BY_EID.get(e, np.array([], np.int64))
        if restrict_lang is not None and t.size:
            t = lang_filter(t, restrict_lang)
        e2t[e] = t
    allt = np.concatenate([e2t[e] for e in eids]) if eids else np.array([], np.int64)
    if allt.size == 0:
        return None
    y = S.y[allt]
    a = auc(y, score[allt])
    # per-example mean
    ex_sc, ex_y = [], []
    for e in eids:
        t = e2t[e]
        if t.size == 0:
            continue
        ex_sc.append(float(score[t].mean()))
        ex_y.append(int((S.y[t] == 1).any()))
    ex_auc = auc(np.array(ex_y), np.array(ex_sc)) if len(set(ex_y)) > 1 else float("nan")
    # bootstrap over examples
    ba, bex = [], []
    eids_arr = np.array(eids)
    for _ in range(n_boot):
        samp = eids_arr[rng.integers(0, len(eids_arr), len(eids_arr))]
        tt = np.concatenate([e2t[int(e)] for e in samp if e2t[int(e)].size])
        if tt.size == 0:
            continue
        yy = S.y[tt]
        if len(np.unique(yy)) > 1:
            ba.append(auc(yy, score[tt]))
        es, ey = [], []
        for e in samp:
            t = e2t[int(e)]
            if t.size:
                es.append(float(score[t].mean())); ey.append(int((S.y[t] == 1).any()))
        if len(set(ey)) > 1:
            bex.append(auc(np.array(ey), np.array(es)))
    ci = [float(np.percentile(ba, 2.5)), float(np.percentile(ba, 97.5))] if ba else [float("nan")] * 2
    ci_ex = [float(np.percentile(bex, 2.5)), float(np.percentile(bex, 97.5))] if bex else [float("nan")] * 2
    return {
        "tokens_code_auc": a, "ci": ci,
        # example-LEVEL AUC, mean-pooled score per example (glossary: "example
        # AUC (mean-pool)", docs/project-log.md §3). NOT exp-23's
        # per_example_mean_auc, which is within-example macro token AUC.
        "example_mean_auc": ex_auc, "ci_example": ci_ex,
        "n_tok": int(allt.size), "n_pos_tok": int((y == 1).sum()),
        "n_pos_ex": int(sum(ex_y)), "n_ex": len(ex_y),
    }


# ============================ build feature blocks ============================
log("building feature blocks…")
t0 = time.time()
TRAIN_CODE_IDX = code_toks(TRAIN_EX)            # vocab source = all train live-code tokens
BLOCKS = F.build_feature_blocks(S, TRAIN_CODE_IDX)
log(f"feature blocks built in {time.time()-t0:.1f}s; vocab={BLOCKS['vocab_size']}, "
    f"H={BLOCKS['H'].shape}")

E_KEYS = ("U", "H", "L")        # baseline (e) = a+b+d
COMBINED = F.hstack(BLOCKS, E_KEYS)


def keyword_untrained_score():
    return np.asarray(BLOCKS["K"].sum(axis=1)).ravel().astype(np.float32)


def lang_untrained_score():
    return np.asarray(BLOCKS["L"].todense()).ravel().astype(np.float32)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    import designs
    if which in ("1", "all"):
        designs.design1()
    if which in ("2", "all"):
        designs.design2()
    if which in ("3", "all"):
        designs.design3()
    if which in ("4", "all"):
        designs.design4()
    log("done", which)
