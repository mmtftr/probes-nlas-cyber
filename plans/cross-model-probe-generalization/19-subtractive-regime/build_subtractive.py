# [ai-generated]
"""Build the SVEN-subtractive subset.

A vuln example is SUBTRACTIVE iff its fix DELETES/REPLACES >=1 live-code
character in `before` — i.e. the vulnerable code can be localized to a real
token. ADDITIVE/cosmetic fixes (the fix only adds code, or only edits
comments/whitespace) have no such char and are dropped, together with their
safe pair ("drop-pair").

Membership is defined at the CHARACTER level via tree-sitter live-code ranges
(src.eval.code_mask.live_code_char_ranges) so it is model/tokenizer-independent.
Tight changed spans come from a local difflib of (before, after) over each
(file,func) pair.

Outputs (in this dir):
  subtractive_membership.json — {def, params, counts, subtractive_vuln,
      additive_vuln, kept_eids (vuln+safe, original ids), pairs:[[v,s],...]}
  dataset_subtractive.jsonl   — the kept base rows, each with an added
      "_orig_eid" field (line index into the base dataset.jsonl) for alignment.
"""
from __future__ import annotations
import difflib, json, sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
from src.eval.code_mask import live_code_char_ranges  # noqa: E402

DATA = REPO / "data" / "dataset.jsonl"
SPLIT = REPO / "data" / "sven_split_meta.json"


def tight_spans(before: str, after: str):
    sm = difflib.SequenceMatcher(a=before, b=after, autojunk=False)
    return [(i1, i2) for t, i1, i2, j1, j2 in sm.get_opcodes()
            if t in ("replace", "delete") and i2 > i1]


def overlaps_live(spans, live):
    if live is None:          # tree-sitter unavailable -> can't gate; treat as live
        return bool(spans)
    for (s, e) in spans:
        for (ls, le) in live:
            if s < le and e > ls:
                return True
    return False


def main():
    ds = [json.loads(l) for l in open(DATA)]
    split = json.loads(SPLIT.read_text())
    # group-clean held-out groups -> test eids
    heldout = set(split.get("heldout_groups") or [])

    def group_key(r):
        return f"func::{r.get('_file_name')}::{r.get('_func_name')}"
    test_eids = {i for i, r in enumerate(ds) if group_key(r) in heldout}

    # pair by (file, func)
    grp = defaultdict(list)
    for eid, r in enumerate(ds):
        grp[(r.get("_file_name"), r.get("_func_name"))].append(eid)

    pairs, sub_vuln, add_vuln = [], [], []
    for eids in grp.values():
        vs = [e for e in eids if ds[e]["label"] == 1]
        ss = [e for e in eids if ds[e]["label"] == 0]
        for i in range(min(len(vs), len(ss))):
            v, s = vs[i], ss[i]
            before, after = ds[v]["code"], ds[s]["code"]
            spans = tight_spans(before, after)
            live = live_code_char_ranges(before, ds[v].get("lang") or "")
            if overlaps_live(spans, live):
                sub_vuln.append(v)
                pairs.append([v, s])
            else:
                add_vuln.append(v)

    kept = sorted([v for v, s in pairs] + [s for v, s in pairs])
    n_test = lambda xs: sum(1 for e in xs if e in test_eids)
    out = {
        "def": "vuln example kept iff difflib(before,after) delete/replace span "
               "overlaps tree-sitter live-code in `before`; additive/cosmetic dropped "
               "with safe pair (drop-pair). Char-level => model-independent.",
        "n_base_examples": len(ds),
        "n_vuln": sum(r["label"] for r in ds),
        "counts": {
            "subtractive_vuln": len(sub_vuln),
            "additive_vuln": len(add_vuln),
            "kept_examples": len(kept),
            "subtractive_vuln_train": len(sub_vuln) - n_test(sub_vuln),
            "subtractive_vuln_test": n_test(sub_vuln),
            "additive_vuln_train": len(add_vuln) - n_test(add_vuln),
            "additive_vuln_test": n_test(add_vuln),
        },
        "subtractive_vuln": sorted(sub_vuln),
        "additive_vuln": sorted(add_vuln),
        "kept_eids": kept,
        "pairs": pairs,
    }
    (HERE / "subtractive_membership.json").write_text(json.dumps(out, indent=2))

    with (HERE / "dataset_subtractive.jsonl").open("w") as f:
        for e in kept:
            row = dict(ds[e]); row["_orig_eid"] = e
            f.write(json.dumps(row) + "\n")

    c = out["counts"]
    print(json.dumps(out["counts"], indent=2))
    print(f"\nSUBTRACTIVE subset: {c['kept_examples']} examples "
          f"({c['subtractive_vuln']} vuln + {c['subtractive_vuln']} safe), "
          f"from base {len(ds)}.")
    print(f"Dropped {c['additive_vuln']} additive/cosmetic pairs "
          f"({c['additive_vuln']*2} examples).")
    print(f"Test pairs: {c['subtractive_vuln_test']}  Train pairs: {c['subtractive_vuln_train']}")


if __name__ == "__main__":
    main()
