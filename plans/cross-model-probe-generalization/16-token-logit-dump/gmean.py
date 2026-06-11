# [ai-generated]
"""g-mean / g-mean^2 readout on the exp-16 per-token logit dumps (LOCAL, no GPU).

g-mean = sqrt(TPR * TNR), g-mean^2 = TPR * TNR. Unlike AUC (threshold-free
ranking) it scores a chosen OPERATING POINT and collapses to 0 if either class
is ignored -- the right lens for the heavily imbalanced token-level eval.

We report it at the threshold that MAXIMISES g-mean (the probe's best
class-balanced operating point), at two granularities, on the held-out test:

  TOKEN level  -- positive = vulnerable code token, negative = safe code token
                  (tight-diff span AND is_code; "tok/code-X" honest label).
                  Heavily imbalanced -> g-mean is the informative metric.
  EXAMPLE level -- score = max code-token prob over the function; positive =
                  vuln function, negative = safe function. Reported over ALL
                  test examples and over the SUBTRACTIVE-only test subset
                  (additive vulns have no positive token by construction).

CAVEAT: exp-16 probes were TRAINED under the old line/all regime, not the
exp-19 subtractive/token/X regime. exp-19 showed token~=line (delta<=0.01) on
AUC, so these g-mean^2 numbers are representative; the definitive canonical
number is a cluster re-run of the exp-19 grid with this metric added.
"""
from __future__ import annotations
import difflib, json
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
REPO = HERE.parents[2]
DS = REPO / "data" / "dataset.jsonl"
MEMBERSHIP = REPO / "plans/cross-model-probe-generalization/19-subtractive-regime/subtractive_membership.json"

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
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")


def max_gmean(y, s):
    """Threshold maximising g-mean. Returns (gmean2, gmean, tpr, tnr, thr)."""
    from sklearn.metrics import roc_curve
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return (float("nan"),) * 5
    fpr, tpr, thr = roc_curve(y, s)
    tnr = 1.0 - fpr
    g = np.sqrt(np.clip(tpr * tnr, 0.0, None))
    k = int(np.argmax(g))
    t = float(min(thr[k], np.max(s)))
    return float(tpr[k] * tnr[k]), float(g[k]), float(tpr[k]), float(tnr[k]), t


def tight_spans(before: str, after: str):
    sm = difflib.SequenceMatcher(a=before, b=after, autojunk=False)
    return [(i1, i2) for tag, i1, i2, j1, j2 in sm.get_opcodes()
            if tag in ("replace", "delete") and i2 > i1]


def main():
    ds = [json.loads(l) for l in open(DS)]
    grp = defaultdict(list)
    for eid, r in enumerate(ds):
        grp[(r.get("_file_name"), r.get("_func_name"))].append(eid)
    vuln_to_safe = {}
    for eids in grp.values():
        vs = [e for e in eids if ds[e]["label"] == 1]
        ss = [e for e in eids if ds[e]["label"] == 0]
        for i in range(min(len(vs), len(ss))):
            vuln_to_safe[vs[i]] = ss[i]
    tspans = {v: tight_spans(ds[v]["code"], ds[s]["code"]) for v, s in vuln_to_safe.items()}
    sub_set = set(json.loads(MEMBERSHIP.read_text())["kept_eids"]) if MEMBERSHIP.is_file() else set()

    rows = []
    hdr = (f"{'model':28s} {'L':>3} {'AUC_tok':>7} | "
           f"{'TOK g2':>6} {'gmean':>6} {'TPR':>5} {'TNR':>5} {'thr':>6} | "
           f"{'EX g2(all)':>10} {'EX g2(sub)':>10} {'n_ex':>5}")
    print(hdr); print("-" * len(hdr))
    for d_name, L in OP_LAYER.items():
        npz = np.load(RESULTS / d_name / f"logits_layer{L:02d}.npz")
        logit, y = npz["logit"], npz["y"].astype(int)
        eid, cs, ce = npz["example_id"], npz["char_start"], npz["char_end"]
        te, isc = npz["is_test"].astype(bool), npz["is_code"].astype(bool)

        # honest token positive: tight-diff span AND is_code
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

        # --- TOKEN level on test code tokens ---
        m = te & isc
        a = auc(y_tok[m], logit[m])
        g2, gm, tpr, tnr, thr = max_gmean(y_tok[m], logit[m])

        # --- EXAMPLE level: max code-token LOGIT per example, vuln vs safe ---
        # (rank on logit, not sigmoid(prob): large probe logits saturate prob to
        #  exactly 1.0 and create ties that artificially collapse g-mean.)
        ex_score, ex_label, ex_eid = [], [], []
        cm = te & isc
        for e in np.unique(eid[cm]):
            sel = cm & (eid == e)
            if not sel.any():
                continue
            ex_score.append(float(logit[sel].max()))
            ex_label.append(int(ds[int(e)]["label"]))
            ex_eid.append(int(e))
        ex_score = np.array(ex_score); ex_label = np.array(ex_label); ex_eid = np.array(ex_eid)
        g2_all = max_gmean(ex_label, ex_score)[0]
        # subtractive-only: keep safe examples + vuln examples in the subtractive subset
        keep = np.array([(lb == 0) or (e in sub_set) for e, lb in zip(ex_eid, ex_label)])
        g2_sub = max_gmean(ex_label[keep], ex_score[keep])[0] if keep.any() else float("nan")

        name = d_name.replace("logitdump_", "").replace("Qwen_Qwen2.5-Coder-", "Qwen-").replace("google_gemma-3-", "g3-")
        print(f"{name:28s} {L:3d} {a:7.3f} | {g2:6.3f} {gm:6.3f} {tpr:5.2f} {tnr:5.2f} {thr:6.2f} | "
              f"{g2_all:10.3f} {g2_sub:10.3f} {len(ex_label):5d}")
        rows.append({"model": name, "layer": L, "auc_token": a,
                     "token_gmean_sq": g2, "token_gmean": gm, "token_tpr": tpr,
                     "token_tnr": tnr, "token_threshold": thr,
                     "example_gmean_sq_all": g2_all, "example_gmean_sq_sub": g2_sub,
                     "n_examples": int(len(ex_label))})
    (HERE / "gmean_results.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {HERE / 'gmean_results.json'}")
    print("g2 = g-mean^2 = TPR*TNR at the g-mean-maximising threshold (test split).")


if __name__ == "__main__":
    main()
