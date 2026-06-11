# [ai-generated]
"""Characterize PrimeVul-Paired locally to inform the experiment briefing.

PrimeVul-Paired (HF colin/PrimeVul, config `paired`) is a before/after contrast
set just like our SVEN dataset: rows alternate target=1 (vulnerable `func`) then
target=0 (its patched `func`); consecutive rows are a pair from the same commit.

We report, per split: #pairs, language mix (by file extension), top CWEs, and
the SUBTRACTIVE FRACTION -- the share of pairs whose fix DELETES/REPLACES >=1
live-code char in the vulnerable function (reusing the exact exp-19 / ADR-0004
definition: difflib delete/replace span ∩ tree-sitter live-code). That fraction
decides how much of PrimeVul a token-localized probe can in-principle reach
(SVEN was ~67% subtractive; the additive third is undetectable).

No GPU. Downloads ~10 MB of JSONL from HF.
"""
from __future__ import annotations
import difflib, json, sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
from src.eval.code_mask import live_code_char_ranges  # noqa: E402

# UNAMBIGUOUS extensions only. `.h` and no-extension files are AMBIGUOUS in a
# C/C++ corpus (a .h header or extension-less file can be C or C++), so they
# follow the `default` lang and we report a C-vs-C++ sensitivity pass on them.
UNAMBIG = {".c": "c", ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp",
           ".hpp": "cpp", ".hh": "cpp", ".py": "python"}
AMBIG_SUFFIXES = (".h",)  # + no recognised extension


def tight_spans(before: str, after: str):
    sm = difflib.SequenceMatcher(a=before, b=after, autojunk=False)
    return [(i1, i2) for t, i1, i2, j1, j2 in sm.get_opcodes()
            if t in ("replace", "delete") and i2 > i1]


def overlaps_live(spans, live):
    if live is None:
        return bool(spans)          # no grammar -> can't gate; count any change
    for (s, e) in spans:
        for (ls, le) in live:
            if s < le and e > ls:
                return True
    return False


def is_ambiguous_lang(fname: str) -> bool:
    f = (fname or "").lower()
    if any(f.endswith(e) for e in UNAMBIG):
        return False
    return True  # .h or no recognised extension


def lang_of(fname: str, default: str = "c") -> str:
    f = (fname or "").lower()
    for ext, lang in UNAMBIG.items():
        if f.endswith(ext):
            return lang
    return default  # ambiguous (.h / no-ext) -> caller's default (C or C++)


def load_split(split: str):
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("colin/PrimeVul", f"primevul_{split}_paired.jsonl",
                        repo_type="dataset")
    return [json.loads(l) for l in open(p)]


def pairs_of(rows):
    """Consecutive (target=1, target=0) -> (vuln_row, fixed_row).

    The paired splits are STRICTLY alternating by construction; assert that
    rather than silently resync (a silent resync could drop/misalign pairs and
    quietly corrupt the subtractive fraction). Fail loud if the data drifts.
    """
    if len(rows) % 2 != 0:
        raise SystemExit(f"paired split has odd row count {len(rows)} — not strictly paired")
    out = []
    for k in range(0, len(rows), 2):
        a, b = rows[k], rows[k + 1]
        if a.get("target") != 1 or b.get("target") != 0:
            raise SystemExit(
                f"pair {k//2}: expected (target=1, target=0), got "
                f"({a.get('target')}, {b.get('target')}) at rows {k},{k+1}")
        out.append((a, b))
    return out


def analyse(split: str, ambig_default: str = "c", _rows=None):
    rows = _rows if _rows is not None else load_split(split)
    pairs = pairs_of(rows)
    langs, cwes = Counter(), Counter()
    sub = add = n_ambig = sub_ambig = 0
    for v, f in pairs:
        lang = lang_of(v.get("file_name"), default=ambig_default)
        langs[lang] += 1
        for c in (v.get("cwe") or []):
            cwes[c] += 1
        before, after = v["func"], f["func"]
        spans = tight_spans(before, after)
        live = live_code_char_ranges(before, lang)
        is_sub = overlaps_live(spans, live)
        sub += int(is_sub); add += int(not is_sub)
        if is_ambiguous_lang(v.get("file_name")):
            n_ambig += 1; sub_ambig += int(is_sub)
    return {
        "split": split, "rows": len(rows), "pairs": len(pairs),
        "ambig_default": ambig_default,
        "langs": dict(langs.most_common()),
        "top_cwe": dict(cwes.most_common(12)), "n_distinct_cwe": len(cwes),
        "subtractive": sub, "additive": add,
        "subtractive_frac": round(sub / max(len(pairs), 1), 3),
        "n_ambiguous": n_ambig, "ambiguous_sub_frac": round(sub_ambig / max(n_ambig, 1), 3),
    }


def main():
    splits = ("train", "valid", "test")
    raw = {s: load_split(s) for s in splits}            # download once
    res = [analyse(s, "c", _rows=raw[s]) for s in splits]   # primary: ambiguous -> C
    sens = [analyse(s, "cpp", _rows=raw[s]) for s in splits]  # sensitivity: ambiguous -> C++
    tot_p = sum(r["pairs"] for r in res)
    tot_sub = sum(r["subtractive"] for r in res)
    print(f"{'split':6s} {'pairs':>6} {'sub':>6} {'add':>6} {'sub_frac':>9} {'ambig':>6} {'amb_subf':>9}  langs")
    for r in res:
        print(f"{r['split']:6s} {r['pairs']:6d} {r['subtractive']:6d} {r['additive']:6d} "
              f"{r['subtractive_frac']:9.3f} {r['n_ambiguous']:6d} {r['ambiguous_sub_frac']:9.3f}  {r['langs']}")
    print(f"\nTOTAL pairs={tot_p}  subtractive={tot_sub} ({tot_sub/max(tot_p,1):.3f})")
    # lang sensitivity: how much does forcing ambiguous (.h/no-ext) files to C++
    # instead of C move the subtractive fraction?
    sp_c = sum(r["subtractive"] for r in res) / max(tot_p, 1)
    sp_cpp = sum(r["subtractive"] for r in sens) / max(tot_p, 1)
    print(f"LANG SENSITIVITY (ambiguous -> C vs C++): subfrac {sp_c:.3f} vs {sp_cpp:.3f} "
          f"(Δ={abs(sp_c-sp_cpp):.3f})")
    print(f"distinct CWEs (train) = {res[0]['n_distinct_cwe']}")
    print(f"top CWEs (train) = {res[0]['top_cwe']}")
    (HERE / "characterize_results.json").write_text(
        json.dumps({"primary_default_c": res, "sensitivity_default_cpp": sens}, indent=2))
    print(f"\nwrote {HERE / 'characterize_results.json'}")
    print("(SVEN reference: 1430 base examples -> 478 subtractive vuln pairs, ~67% of paired vulns.)")


if __name__ == "__main__":
    main()
