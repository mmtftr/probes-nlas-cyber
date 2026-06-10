# [ai-generated]
"""exp-24 eval designs 1-4. Imported by run_exp24 after its module state is built."""
from __future__ import annotations

import json
import time
from collections import Counter

import numpy as np

import run_exp24 as R
import features as F
from substrate import INJ

S = R.S
log = R.log


def _vuln_train_eids(c=None):
    return [e for e in R.TRAIN_EX if (S.cwe_ex[e] == c if c else not S.clean_ex[e])]


def _vuln_test_eids(c):
    return [e for e in R.TEST_EX if S.cwe_ex[e] == c]


def _pos_tokens(eids):
    """y==1 live-code tokens within the given examples."""
    t = R.code_toks(eids)
    return t[S.y[t] == 1]


def _fit(feat, pos_idx, neg_idx, rng):
    return F.fit_lr_score(feat, pos_idx, neg_idx, rng)


# ----- per-design baseline score builders (general training pool) -----
def _general_scores(rng):
    pos = R.TRAIN_CODE_IDX[S.y[R.TRAIN_CODE_IDX] == 1]
    neg = R.TRAIN_CODE_IDX[S.y[R.TRAIN_CODE_IDX] == 0]
    log(f"design1 train pos={len(pos)} neg={len(neg)}")
    return {
        "probe_general": S.prob,
        "token_unigram_lr": _fit(R.BLOCKS["U"], pos, neg, rng),
        "char_ngram_lr": _fit(R.BLOCKS["H"], pos, neg, rng),
        "keyword_lr": _fit(R.BLOCKS["K"], pos, neg, rng),
        "keyword_untrained": R.keyword_untrained_score(),
        "lang_indicator": R.lang_untrained_score(),
        "combined_abd_lr": _fit(R.COMBINED, pos, neg, rng),
    }


# ============================== DESIGN 1 ==============================
def design1():
    rng = np.random.default_rng(R.SEED)
    scores = _general_scores(rng)
    pos_eids = [e for e in R.TEST_EX if not S.clean_ex[e]]
    out = {"design": "1_general",
           "note": "all live-code test tokens; pos=annotated vuln-span tokens, "
                   "neg=other live-code tokens. Compare probe vs surface.",
           "probe_ref_headline": 0.776, "baselines": {}}
    for name, sc in scores.items():
        log(f"  design1 eval {name}")
        out["baselines"][name] = R.pool_eval(sc, pos_eids, R.CLEAN_TEST,
                                             n_boot=1000, rng=rng)
    best = max((v["tokens_code_auc"] for k, v in out["baselines"].items()
                if k != "probe_general"))
    out["probe_minus_best_surface"] = out["baselines"]["probe_general"]["tokens_code_auc"] - best
    (R.OUT / "design1_general.json").write_text(json.dumps(out, indent=2))
    log(f"design1 probe={out['baselines']['probe_general']['tokens_code_auc']:.4f} "
        f"best_surface={best:.4f} Δ={out['probe_minus_best_surface']:+.4f}")


# ============================== DESIGN 2 ==============================
def design2():
    """Per-CWE all-clean diagonal: train CWE-X span vs clean; eval CWE-X vuln vs
    clean. Compare against exp-21 transfer_allclean.json diagonal."""
    rng = np.random.default_rng(R.SEED)
    exp21 = json.load(open(R.OUT.parent.parent / "21-per-cwe-cross-cwe/results/qwen32b/transfer_allclean.json"))
    cwes = exp21["cwes"]
    neg_train = R.code_toks(R.CLEAN_TRAIN)
    out = {"design": "2_percwe_allclean_diagonal",
           "note": "train CWE-X span-tokens vs clean-train; eval CWE-X test vuln "
                   "vs clean-test (exp-21 eval pool, bit-identical). Probe column = "
                   "exp-21 specialized per-CWE probe (transfer_allclean diagonal).",
           "cwes": cwes, "inj": list(INJ), "rows": {}}
    for c in cwes:
        pos_train = _pos_tokens(_vuln_train_eids(c))
        n_tr = len(pos_train)
        pos_eids = _vuln_test_eids(c)
        n_te = len(pos_eids)
        trust = n_te >= R.MIN_TRUST
        row = {"n_train_pos_tok": int(n_tr), "n_test_vuln_ex": n_te, "trust": trust,
               "fam": "inj" if c in INJ else "mem",
               "exp21_probe_diag": exp21["diagonal"][c]["auc"],
               "exp21_probe_ci": exp21["diagonal"][c]["ci"]}
        if n_tr < 5:
            row["skipped"] = "n_train_pos<5"
            out["rows"][c] = row
            continue
        sc_e = _fit(R.COMBINED, pos_train, neg_train, rng)
        sc_u = _fit(R.BLOCKS["U"], pos_train, neg_train, rng)
        sc_h = _fit(R.BLOCKS["H"], pos_train, neg_train, rng)
        sc_k = _fit(R.BLOCKS["K"], pos_train, neg_train, rng)
        cols = {
            "combined_abd_lr": sc_e, "token_unigram_lr": sc_u, "char_ngram_lr": sc_h,
            "keyword_lr": sc_k, "keyword_untrained": R.keyword_untrained_score(),
            "lang_indicator": R.lang_untrained_score(), "probe_general": S.prob,
        }
        for nm, sc in cols.items():
            row[nm] = R.pool_eval(sc, pos_eids, R.CLEAN_TEST, n_boot=500, rng=rng)
        out["rows"][c] = row
        log(f"design2 {c}: lang={row['lang_indicator']['tokens_code_auc']:.3f} "
            f"comb={row['combined_abd_lr']['tokens_code_auc']:.3f} "
            f"exp21probe={row['exp21_probe_diag']:.3f} trust={trust}")
    (R.OUT / "design2_percwe_diag.json").write_text(json.dumps(out, indent=2))


# ============================== DESIGN 3 ==============================
def design3():
    """Within-language deconfound. (3a) design-1 within python / within c+cpp.
    (3b) design-2 diagonal within the CWE's own language (memory=C, inj=python)."""
    rng = np.random.default_rng(R.SEED)
    out = {"design": "3_within_language", "a_general": {}, "b_percwe": {}}

    # ---- 3a: general within-language ----
    for tag, langs in (("python", ("python",)), ("c_cpp", ("c", "cpp"))):
        tr = R.lang_filter(R.TRAIN_CODE_IDX, langs)
        pos = tr[S.y[tr] == 1]; neg = tr[S.y[tr] == 0]
        scores = {
            "probe_general": S.prob,
            "token_unigram_lr": _fit(R.BLOCKS["U"], pos, neg, rng),
            "char_ngram_lr": _fit(R.BLOCKS["H"], pos, neg, rng),
            "combined_abd_lr": _fit(R.COMBINED, pos, neg, rng),
            "lang_indicator": R.lang_untrained_score(),
            "keyword_untrained": R.keyword_untrained_score(),
        }
        pos_eids = [e for e in R.TEST_EX if not S.clean_ex[e] and S.lang_ex[e] in langs]
        neg_eids = [e for e in R.CLEAN_TEST if S.lang_ex[e] in langs]
        d = {}
        for nm, sc in scores.items():
            d[nm] = R.pool_eval(sc, pos_eids, neg_eids, n_boot=1000, rng=rng,
                                restrict_lang=langs)
        out["a_general"][tag] = d
        log(f"design3a {tag}: probe={d['probe_general']['tokens_code_auc']:.3f} "
            f"comb={d['combined_abd_lr']['tokens_code_auc']:.3f} "
            f"lang={d['lang_indicator']['tokens_code_auc']:.3f}")

    # ---- 3b: per-CWE diagonal within the CWE's own language ----
    # memory CWEs are ~100% C; injection ~Python. Use C clean negs for memory,
    # Python clean negs for injection -> language signal removed.
    cwe_lang = {"CWE-089": ("python",), "CWE-078": ("python",), "CWE-079": ("python",),
                "CWE-022": ("python",), "CWE-125": ("c", "cpp"), "CWE-416": ("c", "cpp"),
                "CWE-476": ("c", "cpp"), "CWE-787": ("c", "cpp"), "CWE-190": ("c", "cpp")}
    for c, langs in cwe_lang.items():
        pos_train = _pos_tokens([e for e in _vuln_train_eids(c) if S.lang_ex[e] in langs])
        if len(pos_train) < 5:
            out["b_percwe"][c] = {"skipped": "n_train_pos<5", "langs": list(langs)}
            continue
        neg_train = R.lang_filter(R.code_toks([e for e in R.CLEAN_TRAIN if S.lang_ex[e] in langs]), langs)
        sc_e = _fit(R.COMBINED, pos_train, neg_train, rng)
        sc_h = _fit(R.BLOCKS["H"], pos_train, neg_train, rng)
        pos_eids = [e for e in _vuln_test_eids(c) if S.lang_ex[e] in langs]
        neg_eids = [e for e in R.CLEAN_TEST if S.lang_ex[e] in langs]
        n_te = len(pos_eids)
        row = {"langs": list(langs), "fam": "inj" if c in INJ else "mem",
               "n_test_vuln_ex": n_te, "trust": n_te >= R.MIN_TRUST,
               "combined_abd_lr": R.pool_eval(sc_e, pos_eids, neg_eids, 500, rng, langs),
               "char_ngram_lr": R.pool_eval(sc_h, pos_eids, neg_eids, 500, rng, langs),
               "lang_indicator": R.pool_eval(R.lang_untrained_score(), pos_eids, neg_eids, 500, rng, langs),
               "probe_general": R.pool_eval(S.prob, pos_eids, neg_eids, 500, rng, langs)}
        out["b_percwe"][c] = row
        log(f"design3b {c} ({'/'.join(langs)}): comb={row['combined_abd_lr']['tokens_code_auc']:.3f} "
            f"lang={row['lang_indicator']['tokens_code_auc']:.3f} "
            f"probe={row['probe_general']['tokens_code_auc']:.3f} n_te={n_te}")
    (R.OUT / "design3_within_language.json").write_text(json.dumps(out, indent=2))


# ============================== DESIGN 4 ==============================
def design4():
    """9x9 surface (e)-LR transfer matrix; block means vs exp-21 family structure."""
    rng = np.random.default_rng(R.SEED)
    exp21 = json.load(open(R.OUT.parent.parent / "21-per-cwe-cross-cwe/results/qwen32b/transfer_allclean.json"))
    cwes = exp21["cwes"]
    neg_train = R.code_toks(R.CLEAN_TRAIN)
    neg_test = R.code_toks(R.CLEAN_TEST)

    # train one combined-(e) probe per CWE-X
    probe_x = {}
    n_train_pos = {}
    for c in cwes:
        pos_train = _pos_tokens(_vuln_train_eids(c))
        n_train_pos[c] = len(pos_train)
        if len(pos_train) < 5:
            continue
        probe_x[c] = _fit(R.COMBINED, pos_train, neg_train, rng)
        log(f"design4 trained surface-e {c} (n_pos={len(pos_train)})")

    # eval pools per CWE-Y (code tokens of vuln-Y test ∪ clean test), label=y
    posY = {c: R.code_toks(_vuln_test_eids(c)) for c in cwes}
    npos_tok = {c: int((S.y[posY[c]] == 1).sum()) for c in cwes}
    n_test_pos = {c: len(_vuln_test_eids(c)) for c in cwes}

    def cell(scX, cy):
        ev = np.concatenate([posY[cy], neg_test])
        return R.auc(S.y[ev], scX[ev])

    M = {cx: {cy: (cell(probe_x[cx], cy) if cx in probe_x else float("nan"))
              for cy in cwes} for cx in cwes}
    # lang-indicator reference matrix (row-invariant by construction)
    lang_sc = R.lang_untrained_score()
    lang_row = {cy: cell(lang_sc, cy) for cy in cwes}

    trusted = [c for c in cwes if n_test_pos[c] >= R.MIN_TRUST]
    inj_t = [c for c in trusted if c in INJ]; mem_t = [c for c in trusted if c not in INJ]

    def block4(getauc, weight):
        g = {"inj": inj_t, "mem": mem_t}; o = {}
        for gtr, ctr in g.items():
            for gte, cte in g.items():
                for tag, skip in (("", False), ("_offdiag", True)):
                    num = den = 0.0
                    for cx in ctr:
                        for cy in cte:
                            if skip and cx == cy:
                                continue
                            a = getauc(cx, cy); w = weight[cy]
                            if a == a and w:
                                num += a * w; den += w
                    o[f"{gtr}->{gte}{tag}"] = (num / den) if den else float("nan")
        return o

    blk = block4(lambda cx, cy: M[cx][cy], npos_tok)
    # bootstrap block CIs: resample clean-test + each trusted CWE's test pos examples
    boot = {k: [] for k in blk}
    clean_test = R.CLEAN_TEST
    for _ in range(500):
        cs = [clean_test[j] for j in rng.integers(0, len(clean_test), len(clean_test))]
        negb = R.code_toks(cs)
        colt, colw = {}, {}
        for cy in trusted:
            pe = _vuln_test_eids(cy)
            ps = [pe[j] for j in rng.integers(0, len(pe), len(pe))]
            colt[cy] = R.code_toks(ps); colw[cy] = int((S.y[colt[cy]] == 1).sum())

        def bcell(cx, cy):
            ev = np.concatenate([colt[cy], negb]); return R.auc(S.y[ev], probe_x[cx][ev])
        bb = block4(bcell, colw)
        for k, v in bb.items():
            if v == v:
                boot[k].append(v)
    blk_ci = {k: ([float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), len(v)]
                  if v else [float("nan")] * 3) for k, v in boot.items()}

    out = {"design": "4_surface_transfer_matrix", "feature": "combined_abd_lr (a+b+d)",
           "cwes": cwes, "inj": list(INJ), "trusted_cwes": trusted,
           "n_train_pos": n_train_pos, "n_test_pos": n_test_pos, "n_pos_tokens": npos_tok,
           "auc": M, "lang_indicator_row": lang_row,
           "block_auc_trusted": blk, "block_auc_trusted_ci": blk_ci,
           "exp21_block_auc_trusted": exp21["block_auc_trusted"],
           "exp21_block_auc_trusted_ci": exp21["block_auc_trusted_ci"]}
    (R.OUT / "design4_transfer_matrix.json").write_text(json.dumps(out, indent=2,
                                                       default=lambda x: None if x != x else x))
    log(f"design4 surface blocks: { {k: round(v,3) for k,v in blk.items() if 'offdiag' not in k} }")
    log(f"design4 exp21  blocks: { {k: round(v,3) for k,v in exp21['block_auc_trusted'].items() if 'offdiag' not in k} }")
