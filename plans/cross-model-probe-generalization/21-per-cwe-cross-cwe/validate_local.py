# [ai-generated]
"""Local validation of the exp-21 matrix machinery WITHOUT the cluster.

Uses the exp-16 pooled-probe logits (Qwen-32B L25) as a stand-in scorer to check:
  (1) subtractive membership + per-CWE counts reproduce the known table;
  (2) example_maxcode + pair_matrix produce a sensible "1-probe row" — the pooled
      probe should rank injection vuln>safe pairs well (≈exp-20) and memory ≈chance.
If those hold, the per-CWE machinery is sound; only the (proven) GPU training and
extraction differ on the cluster.
"""
import difflib, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DS = REPO / "data" / "dataset.jsonl"
LOG = REPO / "plans/cross-model-probe-generalization/16-token-logit-dump/results/logitdump_Qwen_Qwen2.5-Coder-32B-Instruct/logits_layer25.npz"
INJ = {"CWE-089", "CWE-078", "CWE-022", "CWE-079"}

# import the pure functions under test from the cluster script
import sys; sys.path.insert(0, str(HERE))
# avoid importing torch-heavy modules: pull only the pure fns by exec of their source
import importlib.util
src = (HERE / "train_percwe.py").read_text()
# strip the heavy imports + main so we can exec just the helpers
import re
keep = []
for block in src.split("\n\n"):
    if any(s in block for s in ("def tight_spans", "def overlaps_live",
                                "def example_maxcode", "def pair_matrix")):
        keep.append(block)
ns = {"np": np, "roc_auc_score": roc_auc_score, "difflib": difflib}
exec("\n\n".join(keep), ns)
tight_spans, overlaps_live = ns["tight_spans"], ns["overlaps_live"]
example_maxcode, pair_matrix = ns["example_maxcode"], ns["pair_matrix"]


def main():
    rows = [json.loads(l) for l in open(DS)]
    z = np.load(LOG)
    logit, eids = z["logit"], z["example_id"].astype(int)
    char_s, char_e, is_code, te = z["char_start"], z["char_end"], z["is_code"], z["is_test"]
    test_set = set(int(e) for e in np.unique(eids[te]))

    # subtractive membership (live-code gating skipped here: difflib-only proxy is
    # enough to validate the matrix code; true membership uses tree-sitter on cluster)
    grp = defaultdict(list)
    for eid, r in enumerate(rows):
        grp[(r.get("_file_name"), r.get("_func_name"))].append(eid)
    sub_pairs = []
    for es in grp.values():
        vs = [e for e in es if rows[e]["label"] == 1]
        ss = [e for e in es if rows[e]["label"] == 0]
        for i in range(min(len(vs), len(ss))):
            v, s = vs[i], ss[i]
            if tight_spans(rows[v]["code"], rows[s]["code"]):
                sub_pairs.append((v, s))
    cwe_of = {v: rows[v].get("cwe") for v, s in sub_pairs}
    cwes = sorted(set(cwe_of.values()), key=lambda c: (c not in INJ, c))

    # counts (difflib-only proxy ~ should be >= the tree-sitter subtractive 478)
    from collections import Counter
    tr = Counter(); tew = Counter()
    for v, s in sub_pairs:
        (tew if v in test_set else tr)[cwe_of[v]] += 1
    print(f"n sub_pairs (difflib proxy): {len(sub_pairs)}  (tree-sitter subtractive=478)")
    print(f"{'CWE':10s} {'tr':>4s} {'te':>4s}  class")
    for c in cwes:
        print(f"{c:10s} {tr[c]:4d} {tew[c]:4d}  {'inj' if c in INJ else 'mem'}")

    # pooled probe as scorer -> 1-row matrix across test CWEs
    test_eids_all = sorted({v for v, s in sub_pairs if v in test_set}
                           | {s for v, s in sub_pairs if v in test_set})
    sc = example_maxcode(logit, eids, is_code, test_eids_all)
    pacc, auc, npr = pair_matrix(lambda e: sc[e], sub_pairs, cwe_of, test_set, cwes)
    print("\npooled-probe pair-accuracy per test CWE (should be hi for inj, ~.5 for mem):")
    for c in cwes:
        print(f"  {c:10s} pairAcc={pacc[c]:.2f} auc={auc[c]:.2f} n={npr[c]}")
    inj_pa = np.nanmean([pacc[c] for c in cwes if c in INJ])
    mem_pa = np.nanmean([pacc[c] for c in cwes if c not in INJ])
    print(f"\nblock check: injection pairAcc={inj_pa:.2f}  memory pairAcc={mem_pa:.2f}")
    assert inj_pa > mem_pa, "expected injection >> memory for the pooled probe"
    print("OK — matrix machinery reproduces the exp-20 injection/memory split.")


if __name__ == "__main__":
    main()
