# [ai-generated]
"""CHEAP recompute: re-evaluate the EXISTING exp-16 probes under cleaned label
definitions, WITHOUT retraining (no cluster, no GPU). Uses the dumped per-token
logits + char offsets + is_code mask already on disk.

Label schemes compared (token-level AUC on the held-out test split):
  line/all     baseline: whole-line evidence (npz y), ALL test tokens
  line/code    baseline code-only (= historical tokens_code_auc anchor)
  line+isc/code  whole-line evidence AND is_code positives, code tokens
  token/code-X   tight-diff span AND is_code positives, CODE tokens only      (comments-as-ignored)
  token/all-Y    tight-diff span AND is_code positives, ALL tokens (non-code=neg) (comments-as-negative)

Tight spans are reconstructed locally: difflib(before, after) on each vuln
example vs its SAFE pair; changed/deleted char ranges in `before` = the tight
vulnerable region. Additive-only fixes (insert with no before range) -> no
positive token (rely on example-level positivity).

CAVEAT: the probe was TRAINED on line/all labels, so this measures how the
label-DEFINITION shifts the metric on the existing probe — it does NOT predict a
retrained probe. The X-vs-Y retrain is the cluster job.
"""
from __future__ import annotations
import difflib, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
REPO = HERE.parents[2]
DS = REPO / "data" / "dataset.jsonl"

OP_LAYER = {
    "logitdump_Qwen_Qwen2.5-Coder-32B-Instruct": 25,
    "logitdump_Qwen_Qwen2.5-Coder-7B-Instruct": 16,
    "logitdump_google_gemma-3-1b-it": 25,
    "logitdump_google_gemma-3-4b-it": 7,
    "logitdump_google_gemma-3-12b-it": 15,
    "logitdump_google_gemma-3-27b-it": 19,
    "logitdump_google_gemma-3-12b-pt": 13,
}


def auc(y, s):
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")


def tight_spans(before: str, after: str):
    sm = difflib.SequenceMatcher(a=before, b=after, autojunk=False)
    return [(i1, i2) for tag, i1, i2, j1, j2 in sm.get_opcodes()
            if tag in ("replace", "delete") and i2 > i1]


def main():
    ds = [json.loads(l) for l in open(DS)]
    # pair map vuln_eid -> safe_eid (same (file,func), zipped in order)
    grp = defaultdict(list)
    for eid, r in enumerate(ds):
        grp[(r.get("_file_name"), r.get("_func_name"))].append(eid)
    vuln_to_safe = {}
    for eids in grp.values():
        vs = [e for e in eids if ds[e]["label"] == 1]
        ss = [e for e in eids if ds[e]["label"] == 0]
        for i in range(min(len(vs), len(ss))):
            vuln_to_safe[vs[i]] = ss[i]
    # cache tight spans per vuln eid
    tspans = {v: tight_spans(ds[v]["code"], ds[s]["code"]) for v, s in vuln_to_safe.items()}

    print(f"{'model':40s} {'line/all':>9} {'line/code':>9} {'line+isc':>9} {'tok/code-X':>10} {'tok/all-Y':>10}  npos_line npos_tok zero_pos_ex")
    for d_name, L in OP_LAYER.items():
        npz = np.load(RESULTS / d_name / f"logits_layer{L:02d}.npz")
        logit, y = npz["logit"], npz["y"].astype(int)
        eid, cs, ce = npz["example_id"], npz["char_start"], npz["char_end"]
        te, isc = npz["is_test"], npz["is_code"]

        # tight per-token positives (is_code gated)
        y_tok = np.zeros(len(y), dtype=int)
        for v, spans in tspans.items():
            idx = np.where(eid == v)[0]
            if not len(idx) or not spans:
                continue
            s_, e_ = cs[idx], ce[idx]
            ov = np.zeros(len(idx), dtype=bool)
            for (i1, i2) in spans:
                ov |= (s_ < i2) & (e_ > i1)
            y_tok[idx] = (ov & isc[idx]).astype(int)

        y_line_isc = (y.astype(bool) & isc).astype(int)  # whole-line ∩ is_code
        code = isc

        res = {
            "line/all":      auc(y[te], logit[te]),
            "line/code":     auc(y[te & code], logit[te & code]),
            "line+isc/code": auc(y_line_isc[te & code], logit[te & code]),
            "tok/code-X":    auc(y_tok[te & code], logit[te & code]),
            "tok/all-Y":     auc(y_tok[te], logit[te]),
        }
        # how many TEST vuln examples lose all positive tokens under tight+isc
        test_vuln = [v for v in vuln_to_safe if (eid == v).any() and te[eid == v].any() and ds[v]["label"] == 1]
        zero_pos = sum(1 for v in test_vuln if y_tok[eid == v].sum() == 0)
        npos_line = int(y[te].sum())
        npos_tok = int(y_tok[te].sum())
        name = d_name.replace("logitdump_", "")
        print(f"{name:40s} {res['line/all']:9.3f} {res['line/code']:9.3f} {res['line+isc/code']:9.3f} "
              f"{res['tok/code-X']:10.3f} {res['tok/all-Y']:10.3f}  {npos_line:9d} {npos_tok:8d} "
              f"{zero_pos:3d}/{len(test_vuln)}")


if __name__ == "__main__":
    main()
