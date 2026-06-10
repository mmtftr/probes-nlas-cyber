# [ai-generated]
"""exp-23 — language-stratified rescore of exp-16's persisted general-probe logits.

LOCAL, CPU-only. No GPU, no retraining, no model download. Reads the per-token
logit dumps from exp-16 and re-evaluates the SAME general probe, stratified by
programming language and by vuln family, to separate a genuine family effect from
a C-vs-Python language confound.

Label regimes (both reported, primary = line/code = the published tokens_code_auc):
  line/code  : npz `y` (whole-line evidence) gated to is_code  -> historical headline
  tok/code-X : difflib tight-diff span (before vs its SAFE pair) AND is_code -> honest

All AUCs are pooled token-level ROC-AUC over held-out TEST live-code tokens unless
labelled otherwise. Bootstraps resample EXAMPLES (not tokens), seed 42, 1000 reps.

Run:  uv run python rescore_language.py
"""
from __future__ import annotations

import difflib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
EXP16 = REPO / "plans/cross-model-probe-generalization/16-token-logit-dump/results"
DS_PATH = REPO / "data" / "dataset.jsonl"
OUT = HERE / "results"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
N_BOOT = 1000

# operating layer + historical tokens_code_auc anchor (exp-16 RESULTS.md / exp-06)
MODELS = {
    "Qwen2.5-Coder-32B-Instruct": ("logitdump_Qwen_Qwen2.5-Coder-32B-Instruct", 25, 0.776),
    "Qwen2.5-Coder-7B-Instruct":  ("logitdump_Qwen_Qwen2.5-Coder-7B-Instruct", 16, 0.813),
    "gemma-3-1b-it":  ("logitdump_google_gemma-3-1b-it", 25, 0.744),
    "gemma-3-4b-it":  ("logitdump_google_gemma-3-4b-it", 7, 0.775),
    "gemma-3-12b-it": ("logitdump_google_gemma-3-12b-it", 15, 0.763),
    "gemma-3-27b-it": ("logitdump_google_gemma-3-27b-it", 19, 0.759),
    "gemma-3-12b-pt": ("logitdump_google_gemma-3-12b-pt", 13, 0.782),
}

INJ = {"CWE-022", "CWE-078", "CWE-079", "CWE-089"}
MEM = {"CWE-125", "CWE-190", "CWE-416", "CWE-476", "CWE-787"}


def family(cwe):
    if cwe in INJ:
        return "inj"
    if cwe in MEM:
        return "mem"
    return "none"


def lang_group(lang):
    return "py" if lang == "python" else "c"  # c == c+cpp


def auc(y, s):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def tight_spans(before, after):
    sm = difflib.SequenceMatcher(a=before, b=after, autojunk=False)
    return [(i1, i2) for tag, i1, i2, j1, j2 in sm.get_opcodes()
            if tag in ("replace", "delete") and i2 > i1]


def boot_auc_ci(y, s, ex, rng, n=N_BOOT):
    """1000x bootstrap over EXAMPLES. Resample example ids with replacement,
    pool their tokens, recompute pooled AUC. Returns (lo, hi) 95% percentile CI."""
    y = np.asarray(y); s = np.asarray(s); ex = np.asarray(ex)
    uniq = np.unique(ex)
    if len(uniq) < 2 or len(np.unique(y)) < 2:
        return (float("nan"), float("nan"))
    # index lists per example for fast gather
    idx_by_ex = {e: np.where(ex == e)[0] for e in uniq}
    out = []
    for _ in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by_ex[e] for e in pick])
        yy = y[rows]
        if len(np.unique(yy)) < 2:
            continue
        out.append(roc_auc_score(yy, s[rows]))
    if not out:
        return (float("nan"), float("nan"))
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


def per_example_mean_auc(y, s, ex):
    """Mean of within-example AUCs over examples that contain both classes
    (i.e. vuln examples with >=1 positive and >=1 negative code token)."""
    y = np.asarray(y); s = np.asarray(s); ex = np.asarray(ex)
    vals = []
    for e in np.unique(ex):
        m = ex == e
        if len(np.unique(y[m])) == 2:
            vals.append(roc_auc_score(y[m], s[m]))
    if not vals:
        return (float("nan"), 0)
    return (float(np.mean(vals)), len(vals))


def main():
    rng = np.random.default_rng(SEED)
    ds = [json.loads(l) for l in open(DS_PATH)]

    # pair map vuln_eid -> safe_eid (same file/func, zipped in order) for tight spans + pairAcc
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

    summary = {}
    for model, (slug, L, anchor) in MODELS.items():
        npz = np.load(EXP16 / slug / f"logits_layer{L:02d}.npz")
        logit = npz["logit"]; prob = npz["prob"]
        y_line = npz["y"].astype(int)
        eid = npz["example_id"]; cs = npz["char_start"]; ce = npz["char_end"]
        te = npz["is_test"].astype(bool); isc = npz["is_code"].astype(bool)

        # per-token attributes from the dataset
        lang_tok = np.array([lang_group(ds[int(e)]["lang"]) for e in eid])
        cwe_tok = np.array([ds[int(e)].get("cwe") for e in eid], dtype=object)
        fam_tok = np.array([family(c) for c in cwe_tok])
        label_tok = np.array([ds[int(e)]["label"] for e in eid])  # 1=vuln fn, 0=clean fn

        # honest tight-diff token positive (tight span AND is_code)
        y_tok = np.zeros(len(y_line), dtype=int)
        for v, spans in tspans.items():
            idx = np.where(eid == v)[0]
            if not len(idx) or not spans:
                continue
            s_, e_ = cs[idx], ce[idx]
            ov = np.zeros(len(idx), dtype=bool)
            for (i1, i2) in spans:
                ov |= (s_ < i2) & (e_ > i1)
            y_tok[idx] = (ov & isc[idx]).astype(int)

        # ---------- Task 1: format gate ----------
        base = te & isc
        gate = auc(y_line[base], prob[base])
        gate_tok = auc(y_tok[base], prob[base])
        gate_pass = abs(gate - anchor) <= 0.001

        # ---------- Task 2: within-language ----------
        within = {}
        for lg in ("py", "c"):
            for regime, ylab in (("line", y_line), ("tok", y_tok)):
                m = base & (lang_tok == lg)
                pooled = auc(ylab[m], prob[m])
                pe_mean, n_pe = per_example_mean_auc(ylab[m], prob[m], eid[m])
                lo, hi = boot_auc_ci(ylab[m], prob[m], eid[m], rng)
                within[f"{lg}_{regime}"] = {
                    "pooled_auc": pooled, "per_example_mean_auc": pe_mean,
                    "n_per_example_with_both_classes": n_pe,
                    "boot95_lo": lo, "boot95_hi": hi,
                    "n_tokens": int(m.sum()), "n_pos_tokens": int(ylab[m].sum()),
                    "n_examples": int(len(np.unique(eid[m]))),
                    "n_vuln_examples": int(len(np.unique(eid[m & (label_tok == 1)]))),
                }

        # ---------- Task 3: family x language cells (vs same-language negatives) ----------
        # cell = {vuln examples of (fam,lang)} U {clean examples of lang}; AUC over test code tokens.
        cells = {}
        for fam in ("inj", "mem"):
            for lg in ("py", "c"):
                pos_ex_mask = (label_tok == 1) & (fam_tok == fam) & (lang_tok == lg)
                neg_ex_mask = (label_tok == 0) & (lang_tok == lg)
                cell_tok = base & (pos_ex_mask | neg_ex_mask)
                n_pos_ex = int(len(np.unique(eid[base & pos_ex_mask])))
                n_neg_ex = int(len(np.unique(eid[base & neg_ex_mask])))
                for regime, ylab in (("line", y_line), ("tok", y_tok)):
                    a = auc(ylab[cell_tok], prob[cell_tok])
                    lo, hi = boot_auc_ci(ylab[cell_tok], prob[cell_tok], eid[cell_tok], rng)
                    cells[f"{fam}_{lg}_{regime}"] = {
                        "pooled_auc": a, "boot95_lo": lo, "boot95_hi": hi,
                        "n_pos_examples": n_pos_ex, "n_neg_examples": n_neg_ex,
                        "n_pos_tokens": int(ylab[cell_tok].sum()),
                        "n_tokens": int(cell_tok.sum()),
                        "untrusted": n_pos_ex < 10,
                    }

        # ---------- Task 4: language-indicator null table ----------
        # lang indicator score: c/cpp = 1, python = 0 (and reverse). y depends on design.
        lang_ind = (lang_tok == "c").astype(int)
        nulltab = {}

        # (a) general: y = vuln-span positive (both label regimes), all test code tokens
        for regime, ylab in (("line", y_line), ("tok", y_tok)):
            yy = ylab[base]
            nulltab[f"general_{regime}"] = {
                "lang_null_auc_cPos": auc(yy, lang_ind[base]),
                "lang_null_auc_pyPos": auc(yy, 1 - lang_ind[base]),
                "probe_auc": auc(yy, prob[base]),
                "n_pos_tokens": int(yy.sum()), "n_tokens": int(base.sum()),
            }

        # per-CWE designs (b) exp-06 (CWE pos vs ALL other code tokens) and
        # (c) exp-10/21 (CWE pos vs clean-row code tokens only). Positives = CWE-X
        # vuln-span code tokens (line regime). Report probe AUC + lang-null per CWE.
        clean_mask = base & (label_tok == 0)  # cwe==null clean rows, code, test
        percwe = {}
        for cwe in sorted(INJ | MEM):
            cwe_pos = base & (label_tok == 1) & (cwe_tok == cwe) & (y_line == 1)
            n_pos_ex = int(len(np.unique(eid[base & (label_tok == 1) & (cwe_tok == cwe)])))
            # design (c): positives vs clean-only
            sel_c = cwe_pos | clean_mask
            yc = cwe_pos[sel_c].astype(int)
            # design (b): positives vs ALL other test code tokens (everything not this CWE's pos)
            yb = cwe_pos[base].astype(int)
            cwe_lang = "c" if cwe in MEM else ("mixed")  # informational
            # dominant language of this CWE's positives
            if cwe_pos.any():
                dom = "c" if (lang_tok[cwe_pos] == "c").mean() >= 0.5 else "py"
                frac_c = float((lang_tok[cwe_pos] == "c").mean())
            else:
                dom, frac_c = "na", float("nan")
            percwe[cwe] = {
                "family": family(cwe), "dominant_lang": dom, "frac_pos_tokens_c": frac_c,
                "n_pos_examples": n_pos_ex, "n_pos_tokens": int(cwe_pos.sum()),
                # design (c) exp-10/21: vs clean only
                "c_probe_auc": auc(yc, prob[sel_c]),
                "c_lang_null_cPos": auc(yc, lang_ind[sel_c]),
                "c_n_tokens": int(sel_c.sum()),
                # design (b) exp-06: vs all other code tokens
                "b_probe_auc": auc(yb, prob[base]),
                "b_lang_null_cPos": auc(yb, lang_ind[base]),
                "b_n_tokens": int(base.sum()),
                "untrusted": n_pos_ex < 10,
            }

        # ---------- Task 6 (secondary): within-pair before/after max-logit pairAcc ----------
        # per family per language; pair = (vuln eid, safe eid). max code-token logit.
        def max_code_logit(e):
            m = (eid == e) & isc & te
            return float(logit[m].max()) if m.any() else None
        pairacc = {}
        buckets = defaultdict(lambda: [0, 0])  # key -> [correct, total]
        for v, s in vuln_to_safe.items():
            if not te[eid == v].any() or not te[eid == s].any():
                continue
            lv = max_code_logit(v); ls = max_code_logit(s)
            if lv is None or ls is None:
                continue
            key = f"{family(ds[v].get('cwe'))}_{lang_group(ds[v]['lang'])}"
            buckets[key][1] += 1
            if lv > ls:
                buckets[key][0] += 1
        for k, (c, t) in buckets.items():
            pairacc[k] = {"pair_acc": (c / t if t else float("nan")), "n_pairs": t,
                          "untrusted": t < 10}

        result = {
            "model": model, "layer": L, "anchor_tokens_code_auc": anchor,
            "gate": {"line_code_auc": gate, "tok_code_auc": gate_tok,
                     "delta_vs_anchor": gate - anchor, "pass": bool(gate_pass)},
            "within_language": within,
            "family_x_language": cells,
            "language_null": nulltab,
            "per_cwe": percwe,
            "pair_acc_secondary": pairacc,
        }
        (OUT / f"{model}.json").write_text(json.dumps(result, indent=2))
        summary[model] = {
            "gate_pass": bool(gate_pass), "gate_line_code_auc": gate,
            "within_py_line": within["py_line"]["pooled_auc"],
            "within_c_line": within["c_line"]["pooled_auc"],
            "general_lang_null_cPos_line": nulltab["general_line"]["lang_null_auc_cPos"],
        }
        print(f"{model:28s} L{L:>2} gate={gate:.4f}(Δ{gate-anchor:+.4f},{'OK' if gate_pass else 'FAIL'}) "
              f"py={within['py_line']['pooled_auc']:.3f} c={within['c_line']['pooled_auc']:.3f} "
              f"langnull={nulltab['general_line']['lang_null_auc_cPos']:.3f}")

    (OUT / "_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote per-model json + _summary.json to {OUT}")


if __name__ == "__main__":
    main()
