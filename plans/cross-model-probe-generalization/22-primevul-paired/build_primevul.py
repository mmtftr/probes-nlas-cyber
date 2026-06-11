# [ai-generated]
"""Build a PrimeVul-Paired dataset.jsonl in the SVEN schema, plus subtractive
membership, so the existing extractor + probe harness run on it unchanged.

Output rows match data/dataset.jsonl's schema exactly:
  code, label, lang, _file_name, _func_name, cwe, vuln_type, source,
  is_completion_vulnerable, label_confidence, token_labels{evidence,
  vulnerable_line, sink, source, sanitizer}, and a `_split` field
  (train/valid/test from PrimeVul's chronological split).

Pairing: PrimeVul-Paired rows alternate target=1 (vuln) / target=0 (fixed);
consecutive rows are one before/after pair. We assign each pair a UNIQUE
`_func_name` (pvpair<N>) so build_pairs() groups exactly one vuln + one safe.

token_labels for a vuln row = the tight difflib(before, after) delete/replace
char-spans in the vulnerable function (= the "evidence"/"vulnerable_line").
This is REQUIRED: the extractor aborts if no row carries positive token_labels.
(Downstream the honest label is recomputed from the same tight diff, so the
spans here only gate extraction + the optional line-granularity path.)

Subtractive membership = pairs whose tight span overlaps tree-sitter live code
in the vulnerable function (identical definition to exp-19 build_subtractive.py).

No GPU. Writes into this dir.
"""
from __future__ import annotations
import difflib, json, sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
from src.eval.code_mask import live_code_char_ranges  # noqa: E402

UNAMBIG = {".c": "c", ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp",
           ".hpp": "cpp", ".hh": "cpp", ".py": "python"}


def lang_of(fname: str, default: str = "c") -> str:
    f = (fname or "").lower()
    for ext, lang in UNAMBIG.items():
        if f.endswith(ext):
            return lang
    return default


def _line_spans(before: str, after: str, line_cap: int = 4000):
    """Fast line-level delete/replace spans mapped to `before` char ranges.
    For huge functions where char-level difflib (O(n*m)) is prohibitive."""
    a = before.splitlines(keepends=True)
    b = after.splitlines(keepends=True)
    if len(a) > line_cap or len(b) > line_cap:
        return [(0, len(before))] if before else []   # giant outlier: whole-func span
    starts, off = [], 0
    for ln in a:
        starts.append(off); off += len(ln)
    starts.append(off)
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    return [(starts[i1], starts[i2]) for t, i1, i2, j1, j2 in sm.get_opcodes()
            if t in ("replace", "delete") and i2 > i1]


def tight_spans(before: str, after: str, cap: int = 6000):
    """Char-level tight diff spans (delete/replace) in `before`. Length-guarded:
    funcs over `cap` chars use a fast line-level fallback (PrimeVul has funcs up
    to ~480 KB; char-level difflib would hang). cap=6000 keeps ALL SVEN funcs
    (max 5833 chars) char-level, matching exp-19. Same logic in train_grid.py."""
    if len(before) > cap or len(after) > cap:
        return _line_spans(before, after)
    sm = difflib.SequenceMatcher(a=before, b=after, autojunk=False)
    return [(i1, i2) for t, i1, i2, j1, j2 in sm.get_opcodes()
            if t in ("replace", "delete") and i2 > i1]


def overlaps_live(spans, live):
    if live is None:
        return bool(spans)
    for (s, e) in spans:
        for (ls, le) in live:
            if s < le and e > ls:
                return True
    return False


def load_split(split: str):
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("colin/PrimeVul", f"primevul_{split}_paired.jsonl",
                        repo_type="dataset")
    return [json.loads(l) for l in open(p)]


def strict_pairs(rows):
    if len(rows) % 2 != 0:
        raise SystemExit(f"odd row count {len(rows)} — not strictly paired")
    out = []
    for k in range(0, len(rows), 2):
        a, b = rows[k], rows[k + 1]
        if a.get("target") != 1 or b.get("target") != 0:
            raise SystemExit(f"pair {k//2}: expected (1,0), got ({a.get('target')},{b.get('target')})")
        out.append((a, b))
    return out


def main():
    out_rows = []          # SVEN-schema rows; eid = index into this list
    pairs = []             # [(vuln_eid, safe_eid), ...]
    sub_pairs, add_pairs = [], []
    gid = 0
    counts = {}
    for split in ("train", "valid", "test"):
        rows = load_split(split)
        ps = strict_pairs(rows)
        n_sub = 0
        for v, f in ps:
            before, after = v["func"], f["func"]
            lang = lang_of(v.get("file_name"))
            spans = tight_spans(before, after)            # changed chars in vuln func
            live = live_code_char_ranges(before, lang)
            is_sub = overlaps_live(spans, live)
            func_key = f"pvpair{gid}"; gid += 1
            ev = [[int(s), int(e)] for s, e in spans]
            v_eid = len(out_rows)
            out_rows.append(_row(v, lang, func_key, split, label=1, evidence=ev))
            s_eid = len(out_rows)
            out_rows.append(_row(f, lang, func_key, split, label=0, evidence=[]))
            pairs.append((v_eid, s_eid))
            (sub_pairs if is_sub else add_pairs).append((v_eid, s_eid))
            n_sub += int(is_sub)
        counts[split] = {"pairs": len(ps), "subtractive": n_sub, "additive": len(ps) - n_sub}

    ds_path = HERE / "primevul_dataset.jsonl"
    with ds_path.open("w") as fh:
        for r in out_rows:
            fh.write(json.dumps(r) + "\n")

    kept = sorted([e for p in sub_pairs for e in p])
    member = {
        "def": "PrimeVul vuln pair kept iff difflib(before,after) delete/replace span "
               "overlaps tree-sitter live-code in `before` (vuln func). Identical to "
               "exp-19 build_subtractive.py; char-level, model-independent.",
        "n_examples": len(out_rows), "n_pairs": len(pairs),
        "counts_by_split": counts,
        "subtractive_pairs": sub_pairs, "additive_pairs": add_pairs,
        "kept_eids": kept,
        "test_eids": [i for i, r in enumerate(out_rows) if r["_split"] == "test"],
    }
    (HERE / "primevul_membership.json").write_text(json.dumps(member, indent=2))

    tot_p = len(pairs); tot_sub = len(sub_pairs)
    print(f"wrote {ds_path}  ({len(out_rows)} rows / {tot_p} pairs)")
    print(f"subtractive pairs = {tot_sub} ({tot_sub/max(tot_p,1):.3f}); additive = {len(add_pairs)}")
    for s, c in counts.items():
        print(f"  {s:6s} pairs={c['pairs']:5d} sub={c['subtractive']:5d} add={c['additive']:5d}")
    print(f"langs: {Counter(r['lang'] for r in out_rows)}")
    print(f"wrote {HERE/'primevul_membership.json'}  (kept_eids={len(kept)}, test_eids={len(member['test_eids'])})")


def _row(src, lang, func_key, split, label, evidence):
    cwe = src.get("cwe") or []
    return {
        "code": src["func"],
        "label": int(label),
        "lang": lang,
        "_file_name": src.get("file_name") or "",
        "_func_name": func_key,
        "cwe": cwe,
        "vuln_type": (cwe[0] if cwe else ""),
        "source": "primevul",
        "is_completion_vulnerable": bool(label),
        "label_confidence": 1.0,
        "token_labels": {
            "evidence": evidence, "vulnerable_line": evidence,
            "sink": [], "source": [], "sanitizer": [],
        },
        "_split": split,
        "_pv_idx": src.get("idx"),
        "_commit_id": src.get("commit_id"),
    }


if __name__ == "__main__":
    main()
