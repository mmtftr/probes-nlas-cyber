# [ai-generated]
"""Size the proposed SVEN-subtractive subset: vuln examples that DO have a
localizable code-level change (>=1 tight code-positive token), i.e. drop the
additive/cosmetic-only fixes. Checks model-independence (does the membership
agree across all 7 models?) and reports train/test counts + surviving pairs.
"""
from __future__ import annotations
import difflib, json
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
REPO = HERE.parents[2]
ds = [json.loads(l) for l in open(REPO / "data" / "dataset.jsonl")]

OP = {
    "logitdump_Qwen_Qwen2.5-Coder-32B-Instruct": 25, "logitdump_Qwen_Qwen2.5-Coder-7B-Instruct": 16,
    "logitdump_google_gemma-3-1b-it": 25, "logitdump_google_gemma-3-4b-it": 7,
    "logitdump_google_gemma-3-12b-it": 15, "logitdump_google_gemma-3-27b-it": 19,
    "logitdump_google_gemma-3-12b-pt": 13,
}

# pair map + tight spans
grp = defaultdict(list)
for eid, r in enumerate(ds):
    grp[(r.get("_file_name"), r.get("_func_name"))].append(eid)
v2s = {}
for eids in grp.values():
    vs = [e for e in eids if ds[e]["label"] == 1]; ss = [e for e in eids if ds[e]["label"] == 0]
    for i in range(min(len(vs), len(ss))):
        v2s[vs[i]] = ss[i]

def tight(b, a):
    sm = difflib.SequenceMatcher(a=b, b=a, autojunk=False)
    return [(i1, i2) for t, i1, i2, j1, j2 in sm.get_opcodes() if t in ("replace", "delete") and i2 > i1]
tspans = {v: tight(ds[v]["code"], ds[s]["code"]) for v, s in v2s.items()}

membership = {}  # model -> set of subtractive vuln eids (token criterion)
for d_name, L in OP.items():
    z = np.load(RESULTS / d_name / f"logits_layer{L:02d}.npz")
    eid, cs, ce, isc = z["example_id"], z["char_start"], z["char_end"], z["is_code"]
    sub = set()
    for v, spans in tspans.items():
        idx = np.where(eid == v)[0]
        if not len(idx):
            continue
        ov = np.zeros(len(idx), bool)
        for (i1, i2) in spans:
            ov |= (cs[idx] < i2) & (ce[idx] > i1)
        if (ov & isc[idx]).sum() > 0:
            sub.add(int(v))
    membership[d_name] = sub

# cross-model agreement
sets = list(membership.values())
inter = set.intersection(*sets); union = set.union(*sets)
print(f"subtractive vuln (token criterion) per model: {[len(s) for s in sets]}")
print(f"  intersection {len(inter)}  union {len(union)}  (disagreement on {len(union-inter)} examples)")

ref = membership["logitdump_Qwen_Qwen2.5-Coder-32B-Instruct"]
# split
test_eids = {e for e, r in enumerate(ds)}  # derive is_test from npz
z = np.load(RESULTS / "logitdump_Qwen_Qwen2.5-Coder-32B-Instruct/logits_layer25.npz")
eid, te = z["example_id"], z["is_test"]
is_test = {int(e): bool(te[eid == e].any()) for e in np.unique(eid)}

vuln = [e for e in range(len(ds)) if ds[e]["label"] == 1]
n_v_tr = sum(1 for e in vuln if not is_test[e]); n_v_te = sum(1 for e in vuln if is_test[e])
sub_tr = sum(1 for e in ref if not is_test[e]); sub_te = sum(1 for e in ref if is_test[e])
print(f"\nvuln examples: {len(vuln)} (train {n_v_tr} / test {n_v_te})")
print(f"SUBTRACTIVE vuln kept: {len(ref)} (train {sub_tr} / test {sub_te})")
print(f"ADDITIVE/cosmetic dropped: {len(vuln)-len(ref)} (train {n_v_tr-sub_tr} / test {n_v_te-sub_te})")
# drop-pair => dataset = subtractive vuln + their safe pairs
print(f"\nIf we DROP the additive PAIRS entirely (vuln+safe):")
print(f"  dataset size = {len(ref)} vuln + {len(ref)} safe = {2*len(ref)}  (was {len(ds)})")
print(f"If we KEEP safe of dropped pairs (only drop additive vuln):")
print(f"  dataset size = {len(ref)} vuln + {len([e for e in range(len(ds)) if ds[e]['label']==0])} safe = {len(ref)+sum(1 for e in range(len(ds)) if ds[e]['label']==0)}")
