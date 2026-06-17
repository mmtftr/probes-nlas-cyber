# [ai-generated]
"""exp-27 — exp-24 surface baselines under exp-25's negative regimes.

Stage A (qwen axis): verbatim re-run of exp-24 design-2 (same rng stream,
including its pool_eval calls) capturing each CWE's fitted per-token score
vectors; every all-clean cell must bit-match design2_percwe_diag.json (gate 1).
Stage B: exp-25 negative pools (allclean / conly / pyonly / matchedpatch via
(_file_name,_func_name) i-th-vuln <-> i-th-safe pairing); per-cell token counts
must equal deconfound_{slug}.json exactly (gate 2).
Stage C: eval every captured score under every regime — exp-25 eval_cell
point estimates + lang_null (gate 3: must match exp-25's lang_null) +
exp-25 diag_ci-style 1000-boot CIs (paired safe-half resampling for
matchedpatch) for the headline columns. Secondary: conly-trained surface refit.

Then the same on the gemma-1b token axis (gates 2/3 only; fresh rng seed 42).

CPU only. Writes results/exp27_{qwen32b,gemma1b}_axis.json.
"""
from __future__ import annotations

import gc
import json
import sys
import time
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
PLANS = HERE.parent
EXP24 = PLANS / "24-surface-baselines"
EXP25 = PLANS / "25-allclean-language-matched"
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(EXP24))

import features as F            # noqa: E402  (exp-24, reused)
import substrate as SUB         # noqa: E402  (exp-24, reused)

GEMMA_DUMP = (PLANS / "16-token-logit-dump/results/"
              "logitdump_google_gemma-3-1b-it/logits_layer25.npz")
INJ = set(SUB.INJ)
SEED = 42
N_BOOT_CI = 1000
MIN_TRUST = 10
REGIMES = ("allclean", "conly", "pyonly", "matchedpatch")
# every surface variant gets a CI (review fix: a CI-less variant can't carry a
# "CI includes 0.5" claim); lang_indicator is constant within matched pools.
CI_COLS = ("char_ngram_lr", "combined_abd_lr", "token_unigram_lr",
           "keyword_lr", "keyword_untrained", "probe_general")
# paired probeG-minus-surface Δ bootstrap (same resamples both scores):
# memory CWEs, language/function-matched regimes only.
PAIR_REGIMES = ("conly", "matchedpatch")
PAIR_CWES = ("CWE-125", "CWE-416", "CWE-476")
PAIR_COLS = ("char_ngram_lr", "combined_abd_lr", "token_unigram_lr",
             "conlytrained_char_ngram_lr", "conlytrained_combined_abd_lr")
GATE_TOL = 1e-9


def stable_seed(*parts) -> int:
    return zlib.crc32("/".join(map(str, parts)).encode())


def log(*a):
    print("[exp27]", *a, file=sys.stderr, flush=True)


def auc(y, sc):
    return float(roc_auc_score(y, sc)) if len(np.unique(y)) > 1 else float("nan")


def fast_auc(y, sc):
    """Mann-Whitney AUC with average ranks == roc_auc_score, ~5x faster."""
    r = rankdata(sc)
    npos = int(y.sum())
    nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


# --------------------------------------------------------------------------
# axis-agnostic context (per token axis = per dump)
# --------------------------------------------------------------------------
class Axis:
    def __init__(self, s: SUB.Substrate, slug: str):
        self.S = s
        self.slug = slug
        self.code_by_eid: dict[int, np.ndarray] = {}
        tmp = defaultdict(list)
        for i in range(len(s.y)):
            if s.is_code[i]:
                tmp[int(s.eid[i])].append(i)
        for e, ix in tmp.items():
            self.code_by_eid[e] = np.asarray(ix, np.int64)
        self.train_ex = sorted({int(e) for e in s.eid[~s.is_test]})
        self.test_ex = sorted({int(e) for e in s.eid[s.is_test]})
        self.clean_train = [e for e in self.train_ex if s.clean_ex[e]]
        self.clean_test = [e for e in self.test_ex if s.clean_ex[e]]
        self.is_c_tok = np.isin(s.lang, ("c", "cpp"))

    def code_toks(self, eids):
        if not eids:
            return np.array([], np.int64)
        return np.concatenate([self.code_by_eid[e] for e in eids
                               if e in self.code_by_eid])

    def vuln_train_eids(self, c):
        return [e for e in self.train_ex if self.S.cwe_ex[e] == c]

    def vuln_test_eids(self, c):
        return [e for e in self.test_ex if self.S.cwe_ex[e] == c]

    def pos_tokens(self, eids):
        t = self.code_toks(eids)
        return t[self.S.y[t] == 1]


def build_matchedpatch(rows, test_set):
    """exp-25 deconfound.py pairing, verbatim: (_file_name,_func_name) groups,
    i-th vuln <-> i-th safe. Safe half of a test vuln must itself be held out."""
    grp = defaultdict(list)
    for e, r in enumerate(rows):
        grp[(r.get("_file_name"), r.get("_func_name"))].append(e)
    safe_of_vuln = {}
    for es in grp.values():
        vs = [e for e in es if rows[e].get("label") == 1]
        sf = [e for e in es if rows[e].get("label") == 0]
        for i in range(min(len(vs), len(sf))):
            safe_of_vuln[vs[i]] = sf[i]
    for v, s in safe_of_vuln.items():
        if v in test_set:
            assert s in test_set, f"matched-patch safe {s} of test vuln {v} not held-out"
    return safe_of_vuln


# --------------------------------------------------------------------------
# stage C eval — exp-25 eval_cell + diag_ci mirrors (axis-agnostic)
# --------------------------------------------------------------------------
def eval_cell(ax: Axis, score, pos_eids, neg_eids):
    pt = ax.code_toks(pos_eids)
    ng = ax.code_toks(neg_eids)
    if pt.size == 0 or ng.size == 0:
        return dict(auc=float("nan"), lang_null=float("nan"), n_pos_tok=0, n_neg_tok=0)
    ev = np.concatenate([pt, ng])
    lab = ax.S.y[ev]
    if len(np.unique(lab)) < 2:
        return dict(auc=float("nan"), lang_null=float("nan"),
                    n_pos_tok=int((ax.S.y[pt] == 1).sum()), n_neg_tok=int(ng.size))
    return dict(auc=auc(lab, score[ev]),
                lang_null=auc(lab, ax.is_c_tok[ev].astype(np.float32)),
                n_pos_tok=int((ax.S.y[pt] == 1).sum()), n_neg_tok=int(ng.size))


def ci_cell(ax: Axis, score, pos_eids, regime, pools, safe_of_vuln, seed):
    """exp-25 diag_ci mirror: resample positives w/ replacement; matchedpatch
    negatives follow the resampled positives (paired); other regimes resample
    the clean pool independently. 1000 boots, percentile 95% CI."""
    rng = np.random.default_rng(seed)
    pe = list(pos_eids)
    if not pe:
        return [float("nan"), float("nan"), 0]
    boots = []
    y, isc = ax.S.y, None
    for _ in range(N_BOOT_CI):
        ps = [pe[j] for j in rng.choice(len(pe), len(pe), replace=True)]
        pt = ax.code_toks(ps)
        if regime == "matchedpatch":
            sh = [safe_of_vuln[v] for v in ps if v in safe_of_vuln]
            ng = ax.code_toks(sh)
        else:
            pool = pools[regime]
            if not pool:
                return [float("nan"), float("nan"), 0]
            cs = [pool[j] for j in rng.choice(len(pool), len(pool), replace=True)]
            ng = ax.code_toks(cs)
        if pt.size == 0 or ng.size == 0:
            continue
        ev = np.concatenate([pt, ng])
        lab = y[ev]
        if len(np.unique(lab)) > 1:
            boots.append(fast_auc(lab, score[ev]))
    if not boots:
        return [float("nan"), float("nan"), 0]
    return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)),
            len(boots)]


def ci_pair(ax: Axis, score_a, score_b, pos_eids, regime, pools, safe_of_vuln,
            seed):
    """Paired Δ = AUC(score_a) − AUC(score_b) under the SAME example resamples
    (exp-25 diag_ci resampling). Returns [lo, hi, n_boots, frac_delta_pos]."""
    rng = np.random.default_rng(seed)
    pe = list(pos_eids)
    if not pe:
        return [float("nan")] * 4
    y = ax.S.y
    deltas = []
    for _ in range(N_BOOT_CI):
        ps = [pe[j] for j in rng.choice(len(pe), len(pe), replace=True)]
        pt = ax.code_toks(ps)
        if regime == "matchedpatch":
            sh = [safe_of_vuln[v] for v in ps if v in safe_of_vuln]
            ng = ax.code_toks(sh)
        else:
            pool = pools[regime]
            if not pool:
                return [float("nan")] * 4
            cs = [pool[j] for j in rng.choice(len(pool), len(pool), replace=True)]
            ng = ax.code_toks(cs)
        if pt.size == 0 or ng.size == 0:
            continue
        ev = np.concatenate([pt, ng])
        lab = y[ev]
        if len(np.unique(lab)) > 1:
            deltas.append(fast_auc(lab, score_a[ev]) - fast_auc(lab, score_b[ev]))
    if not deltas:
        return [float("nan")] * 4
    d = np.asarray(deltas)
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)),
            len(d), float((d > 0).mean())]


def run_axis(ax: Axis, blocks, combined, captured, exp25_json, design2_json,
             rows, out_path, conly_captured=None):
    """Stage B (pools + gates) and stage C (regime evals + CIs) for one axis."""
    s = ax.S
    exp25 = json.load(open(exp25_json))
    per25 = exp25["allclean_trained"]["per_cwe"]
    ci25 = exp25["diag_ci_allclean_trained"]

    def is_c_ex(e):
        return s.lang_ex[e] in ("c", "cpp")

    pools = {
        "allclean": ax.clean_test,
        "conly": [e for e in ax.clean_test if is_c_ex(e)],
        "pyonly": [e for e in ax.clean_test if not is_c_ex(e)],
    }
    safe_of_vuln = build_matchedpatch(rows, set(ax.test_ex))

    def neg_eids(c, regime):
        if regime == "matchedpatch":
            return [safe_of_vuln[v] for v in ax.vuln_test_eids(c) if v in safe_of_vuln]
        return pools[regime]

    # ---------- gate 2: exact token-count match vs exp-25 ----------
    gate2 = {}
    for c in captured:
        if c not in per25:
            continue
        for regime in REGIMES:
            pt = ax.code_toks(ax.vuln_test_eids(c))
            ng = ax.code_toks(neg_eids(c, regime))
            npos = int((s.y[pt] == 1).sum())
            nneg = int(ng.size)
            ref = per25[c][regime]
            ok = (npos == ref["n_pos_tok"]) and (nneg == ref["n_neg_tok"])
            gate2[f"{c}/{regime}"] = dict(n_pos_tok=npos, n_neg_tok=nneg,
                                          ref_pos=ref["n_pos_tok"],
                                          ref_neg=ref["n_neg_tok"], ok=ok)
            if not ok:
                raise SystemExit(f"GATE2 FAIL {ax.slug} {c}/{regime}: "
                                 f"pos {npos} vs {ref['n_pos_tok']}, "
                                 f"neg {nneg} vs {ref['n_neg_tok']}")
    log(f"{ax.slug} GATE2 PASS ({len(gate2)} cells, counts == exp-25 exactly)")

    # ---------- stage C ----------
    res_rows = {}
    t0 = time.time()
    for c, cols in captured.items():
        in25 = c in per25
        row = {"family": "inj" if c in INJ else "mem",
               "n_test_vuln_ex": len(ax.vuln_test_eids(c)),
               "trust": len(ax.vuln_test_eids(c)) >= MIN_TRUST,
               "exp25_probe": ({r: {"auc": per25[c][r]["auc"],
                                    "lang_null": per25[c][r]["lang_null"],
                                    "ci": ci25[c][r][:2]} for r in REGIMES}
                               if in25 else None),
               "regimes": {}}
        pe = ax.vuln_test_eids(c)
        for regime in REGIMES:
            ne = neg_eids(c, regime)
            cell = {}
            for nm, sc in cols.items():
                r = eval_cell(ax, sc, pe, ne)
                if nm in CI_COLS:
                    r["ci"] = ci_cell(ax, sc, pe, regime, pools, safe_of_vuln,
                                      stable_seed(ax.slug, c, regime, nm))
                cell[nm] = r
            # gate 3: lang_null identical to exp-25 (same pools, same indicator)
            if in25:
                ln, ref = cell["char_ngram_lr"]["lang_null"], per25[c][regime]["lang_null"]
                if not (np.isnan(ln) and np.isnan(ref)) and abs(ln - ref) > GATE_TOL:
                    raise SystemExit(f"GATE3 FAIL {ax.slug} {c}/{regime}: "
                                     f"lang_null {ln} vs exp-25 {ref}")
            # secondary: conly-trained surface refits (with CIs — the strongest
            # surface variant per cell is load-bearing for any ceiling claim)
            if conly_captured and c in conly_captured:
                for nm, sc in conly_captured[c].items():
                    r = eval_cell(ax, sc, pe, ne)
                    r["ci"] = ci_cell(ax, sc, pe, regime, pools, safe_of_vuln,
                                      stable_seed(ax.slug, c, regime, "conlytr", nm))
                    cell[f"conlytrained_{nm}"] = r
            # paired probeG−surface Δ (same resamples) — the locally computable
            # probe-vs-surface contrast; the specialized probe's per-token
            # scores are cluster-side, so its contrast stays unpaired.
            if c in PAIR_CWES and regime in PAIR_REGIMES:
                all_sc = dict(cols)
                if conly_captured and c in conly_captured:
                    for nm, sc in conly_captured[c].items():
                        all_sc[f"conlytrained_{nm}"] = sc
                cell["probeG_minus_surface_delta"] = {
                    nm: ci_pair(ax, cols["probe_general"], all_sc[nm], pe,
                                regime, pools, safe_of_vuln,
                                stable_seed(ax.slug, c, regime, "pair", nm))
                    for nm in PAIR_COLS if nm in all_sc}
            row["regimes"][regime] = cell
        res_rows[c] = row
        log(f"{ax.slug} {c} done ({time.time()-t0:.0f}s): mp char="
            f"{row['regimes']['matchedpatch']['char_ngram_lr']['auc']:.3f} "
            f"vs probe25={per25[c]['matchedpatch']['auc']:.3f}" if in25 else
            f"{ax.slug} {c} done (not in exp-25)")
    log(f"{ax.slug} GATE3 PASS (lang_null == exp-25 to {GATE_TOL})")

    out = {
        "axis": ax.slug, "metric": "tokens_code_auc", "seed": SEED,
        "n_boot_ci": N_BOOT_CI, "ci_cols": list(CI_COLS),
        "train_recipe": "exp-24 design-2: pos = y==1&is_code TRAIN tokens of "
                        "CWE-X vuln; neg = all-clean TRAIN live-code tokens "
                        "(NEG_CAP subsample); held FIXED across eval regimes.",
        "ci_recipe": "exp-25 diag_ci mirror: 1000 boots over examples; "
                     "matchedpatch negatives = paired safe halves of the "
                     "resampled positives; other regimes resample clean pool "
                     "independently.",
        "gate2_counts": gate2,
        "design2_gate": design2_json is not None,
        "exp25_source": str(exp25_json.name),
        "rows": res_rows,
    }
    out_path.write_text(json.dumps(out, indent=2,
                                   default=lambda x: None if x != x else x))
    log(f"{ax.slug} written -> {out_path}")
    return out


# --------------------------------------------------------------------------
# stage A — qwen axis: verbatim exp-24 design-2 re-run (bit-repro gate 1)
# --------------------------------------------------------------------------
def stage_a_qwen():
    """Replicates designs.design2 exactly (same rng stream incl. pool_eval)
    but captures the fitted score vectors and asserts vs the saved JSON."""
    import run_exp24 as R   # heavy import: builds qwen substrate + blocks

    ref = json.load(open(EXP24 / "results/design2_percwe_diag.json"))
    exp21 = json.load(open(PLANS / "21-per-cwe-cross-cwe/results/qwen32b/"
                                   "transfer_allclean.json"))
    cwes = exp21["cwes"]
    rng = np.random.default_rng(R.SEED)
    neg_train = R.code_toks(R.CLEAN_TRAIN)
    captured = {}
    S = R.S

    def _vuln_train_eids(c):
        return [e for e in R.TRAIN_EX if S.cwe_ex[e] == c]

    def _pos_tokens(eids):
        t = R.code_toks(eids)
        return t[S.y[t] == 1]

    max_dev = 0.0
    for c in cwes:
        pos_train = _pos_tokens(_vuln_train_eids(c))
        pos_eids = [e for e in R.TEST_EX if S.cwe_ex[e] == c]
        if len(pos_train) < 5:
            continue
        sc_e = F.fit_lr_score(R.COMBINED, pos_train, neg_train, rng)
        sc_u = F.fit_lr_score(R.BLOCKS["U"], pos_train, neg_train, rng)
        sc_h = F.fit_lr_score(R.BLOCKS["H"], pos_train, neg_train, rng)
        sc_k = F.fit_lr_score(R.BLOCKS["K"], pos_train, neg_train, rng)
        cols = {
            "combined_abd_lr": sc_e, "token_unigram_lr": sc_u,
            "char_ngram_lr": sc_h, "keyword_lr": sc_k,
            "keyword_untrained": R.keyword_untrained_score(),
            "lang_indicator": R.lang_untrained_score(), "probe_general": S.prob,
        }
        for nm, sc in cols.items():
            got = R.pool_eval(sc, pos_eids, R.CLEAN_TEST, n_boot=500, rng=rng)
            want = ref["rows"][c][nm]
            dev = abs(got["tokens_code_auc"] - want["tokens_code_auc"])
            max_dev = max(max_dev, dev)
            if dev > GATE_TOL:
                raise SystemExit(f"GATE1 FAIL {c}/{nm}: "
                                 f"{got['tokens_code_auc']} vs design2 "
                                 f"{want['tokens_code_auc']} (dev {dev:.2e})")
        captured[c] = cols
        log(f"qwen stage A {c}: design-2 repro OK")
    log(f"qwen GATE1 PASS (design-2 bit-repro, max dev {max_dev:.2e})")

    # secondary: conly-trained refits (char + combined), rng continues
    clean_train_c = [e for e in R.CLEAN_TRAIN if S.lang_ex[e] in ("c", "cpp")]
    neg_train_c = R.code_toks(clean_train_c)
    conly_captured = {}
    for c in cwes:
        pos_train = _pos_tokens(_vuln_train_eids(c))
        if len(pos_train) < 5:
            continue
        conly_captured[c] = {
            "combined_abd_lr": F.fit_lr_score(R.COMBINED, pos_train, neg_train_c, rng),
            "char_ngram_lr": F.fit_lr_score(R.BLOCKS["H"], pos_train, neg_train_c, rng),
        }
    log("qwen conly-trained refits done")
    return R, captured, conly_captured


# --------------------------------------------------------------------------
# gemma axis — same recipe, fresh rng, gates 2/3 only
# --------------------------------------------------------------------------
def train_axis_fresh(ax: Axis, blocks, combined):
    rng = np.random.default_rng(SEED)
    neg_train = ax.code_toks(ax.clean_train)
    s = ax.S
    cwes = sorted({s.cwe_ex[e] for e in ax.train_ex if s.cwe_ex[e]})
    captured, conly_captured = {}, {}
    for c in cwes:
        pos_train = ax.pos_tokens(ax.vuln_train_eids(c))
        if len(pos_train) < 5:
            continue
        captured[c] = {
            "combined_abd_lr": F.fit_lr_score(combined, pos_train, neg_train, rng),
            "token_unigram_lr": F.fit_lr_score(blocks["U"], pos_train, neg_train, rng),
            "char_ngram_lr": F.fit_lr_score(blocks["H"], pos_train, neg_train, rng),
            "keyword_lr": F.fit_lr_score(blocks["K"], pos_train, neg_train, rng),
            "keyword_untrained": np.asarray(blocks["K"].sum(axis=1)).ravel().astype(np.float32),
            "lang_indicator": np.asarray(blocks["L"].todense()).ravel().astype(np.float32),
            "probe_general": s.prob,
        }
        log(f"gemma trained {c} (n_pos_tok={len(pos_train)})")
    neg_train_c = ax.code_toks([e for e in ax.clean_train
                                if s.lang_ex[e] in ("c", "cpp")])
    for c in list(captured):
        pos_train = ax.pos_tokens(ax.vuln_train_eids(c))
        conly_captured[c] = {
            "combined_abd_lr": F.fit_lr_score(combined, pos_train, neg_train_c, rng),
            "char_ngram_lr": F.fit_lr_score(blocks["H"], pos_train, neg_train_c, rng),
        }
    return captured, conly_captured


def main():
    rows = [json.loads(l) for l in (SUB.REPO / "data/dataset.jsonl").open()]

    # ---------------- qwen32b axis ----------------
    R, captured, conly_captured = stage_a_qwen()
    ax_q = Axis(R.S, "qwen32b")
    # sanity: Axis helpers must agree with run_exp24 globals
    assert ax_q.train_ex == R.TRAIN_EX and ax_q.test_ex == R.TEST_EX
    assert ax_q.clean_test == R.CLEAN_TEST
    run_axis(ax_q, R.BLOCKS, R.COMBINED, captured,
             EXP25 / "results/deconfound_qwen32b.json",
             EXP24 / "results/design2_percwe_diag.json",
             rows, OUT / "exp27_qwen32b_axis.json", conly_captured)
    del R, captured, conly_captured, ax_q
    gc.collect()

    # ---------------- gemma1b axis ----------------
    SUB.DUMP = GEMMA_DUMP
    s2 = SUB.load_substrate()
    ax_g = Axis(s2, "gemma1b")
    log("gemma substrate loaded; building feature blocks…")
    t0 = time.time()
    train_code_idx = ax_g.code_toks(ax_g.train_ex)
    blocks_g = F.build_feature_blocks(s2, train_code_idx)
    combined_g = F.hstack(blocks_g, ("U", "H", "L"))
    log(f"gemma blocks built in {time.time()-t0:.0f}s (vocab={blocks_g['vocab_size']})")
    cap_g, conly_g = train_axis_fresh(ax_g, blocks_g, combined_g)
    run_axis(ax_g, blocks_g, combined_g, cap_g,
             EXP25 / "results/deconfound_gemma1b.json", None,
             rows, OUT / "exp27_gemma1b_axis.json", conly_g)
    log("ALL DONE")


if __name__ == "__main__":
    main()
