# [ai-generated]
"""Measure the gap between FIG-H's family bar (positive-token-weighted mean of
per-CWE b_probe_auc over trusted CWEs) and the TRUE pooled family-vs-rest AUC
(union of trusted-family positive tokens vs all other live-code test tokens),
for the models whose per-token npz is present locally.

Same token universe as rescore_language.py: base = is_test & is_code; positives
in the `line/code` regime (npz `y`); per-token CWE from data/dataset.jsonl.
True pooled is the exact design-(b) generalisation of b_probe_auc to a CWE set.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
EXP16 = REPO / "plans/cross-model-probe-generalization/16-token-logit-dump/results"
DS = [json.loads(l) for l in open(REPO / "data" / "dataset.jsonl")]

# FIG-H trusted families (190/787 dropped as untrusted, matching make_figs.py)
INJ_CWES = ["CWE-089", "CWE-078", "CWE-022", "CWE-079"]
MEM_CWES = ["CWE-125", "CWE-416", "CWE-476"]

# models with a local per-token npz (slug, operating layer)
LOCAL = {
    "Qwen2.5-Coder-32B-Instruct": ("logitdump_Qwen_Qwen2.5-Coder-32B-Instruct", 25),
    "gemma-3-1b-it": ("logitdump_google_gemma-3-1b-it", 25),
}


def auc(y, s):
    return float(roc_auc_score(y, s))


def weighted_mean(per_cwe, fam):
    rows = [(per_cwe[c]["b_probe_auc"], per_cwe[c]["n_pos_tokens"]) for c in fam]
    tot = sum(n for _, n in rows)
    return sum(a * n for a, n in rows) / tot


for model, (slug, L) in LOCAL.items():
    npz = np.load(EXP16 / slug / f"logits_layer{L:02d}.npz")
    prob = npz["prob"]
    y_line = npz["y"].astype(int)
    eid = npz["example_id"]
    base = npz["is_test"].astype(bool) & npz["is_code"].astype(bool)
    cwe_tok = np.array([DS[int(e)].get("cwe") for e in eid], dtype=object)
    label_tok = np.array([DS[int(e)]["label"] for e in eid])

    res = json.loads((HERE / "results" / f"{model}.json").read_text())
    per_cwe = res["per_cwe"]
    print(f"\n=== {model}  L{L}  gate={res['gate']['line_code_auc']:.4f} ===")
    for fam_name, fam in (("injection", INJ_CWES), ("memory", MEM_CWES)):
        wm = weighted_mean(per_cwe, fam)
        # true pooled design-(b), EXACT mirror of rescore_language.py's family_pooled
        keep = [c for c in fam if c in per_cwe and not per_cwe[c]["untrusted"]]
        in_fam = np.isin(cwe_tok.astype(str), keep)
        fam_pos = base & (label_tok == 1) & (y_line == 1) & in_fam
        pooled = auc(fam_pos[base].astype(int), prob[base])
        npos = int(fam_pos.sum())
        print(f"  {fam_name:9s}  weighted-mean={wm:.4f}  true-pooled={pooled:.4f}  "
              f"Δ={pooled - wm:+.4f}  (n_pos_tok={npos})")
