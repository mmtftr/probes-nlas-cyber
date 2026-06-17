# [ai-generated]
"""Example-level TPR @ 1% FPR, sliced by language and CWE (exp-33).

The operating-point view of the example-level story (blog NXT4 / open thread). For
the two flagship models (Qwen-32B, gemma-27b) and the lexical ceiling, report TPR
at a frozen FPR=0.01 — the regime a deployed monitor lives in — not just AUC.

Three example-level scorers, all reused (no re-extraction):
  probe        commit-position linear probe (exp-30 kept hidden states, refit at the
               deployable layer/C from introspection_probe.json). The Fig-7 hero.
  char-n-gram  exp-31's strongest char config on raw text; model-independent.
  verbalized   model's own P(yes) at the commit position (npz p_yes).

Threshold convention (see EXPERIMENT.md §5): ROC-interpolated TPR at FPR=0.01,
computed PER SLICE on that slice's matched positives+negatives. Negatives are
matched to CWE via the pair group (user-directed; clean 1:1). At these n the
discrete operating point allows 0-1 false positives, so the discrete TPR is stored
alongside the interpolated one for auditability.

Hard gates (a number is untrusted until these pass): probe refit AUC == exp-30
deployable test_auc (<=2e-3); char AUC == exp-31 stored (<=5e-3); verbalized
npz-p_yes AUC == exp-30 stored (<=5e-3).

CPU only. Run from repo root:  uv run python plans/.../33-operating-point-tpr/compute_tpr.py
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
E30 = REPO / "plans/cross-model-probe-generalization/30-last-token-introspection"
E31 = REPO / "plans/cross-model-probe-generalization/31-neutral-prompt-and-surface"
DATASET = REPO / "data/dataset.jsonl"
SPLIT = REPO / "data/sven_split_meta.json"

# Two flagship models: (display name, hidden-npz slug)
MODELS = {
    "Qwen2.5-Coder-32B-Instruct": "Qwen_Qwen2.5-Coder-32B-Instruct",
    "gemma-3-27b-it": "google_gemma-3-27b-it",
}
# strongest char config from exp-31 (the conservative lexical ceiling)
CHAR_KW = dict(analyzer="char", ngram_range=(3, 5), min_df=2, max_features=100000)
C_GRID_SURF = [1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0, 100.0]
SEED, N_BOOT, TARGET_FPR = 42, 2000, 0.01
PROBE_TOL, CHAR_TOL, VERB_TOL = 2e-3, 5e-3, 5e-3
CWE_MIN_POS = 10  # CWEs with >= this many held-out positives get their own slice


def pair_group_key(r):
    if r.get("_origin_repo"):
        return f"repo::{r['_origin_repo']}"
    fn, fu = r.get("_file_name") or "", r.get("_func_name") or ""
    return f"func::{fn}::{fu}" if (fn or fu) else f"row::{hashlib.sha1((r.get('code') or '').encode()).hexdigest()[:12]}"


def load_split():
    """Identical group-clean split to exp-29/30/31; returns eid lists in the SAME
    sorted test order the probe/surface scorers align to."""
    rows = [json.loads(l) for l in DATASET.open()]
    g = {i: pair_group_key(r) for i, r in enumerate(rows)}
    held = set(json.loads(SPLIT.read_text())["heldout_groups"])
    test = sorted(e for e in g if g[e] in held)
    train = [e for e in g if g[e] not in held]
    groups = sorted({g[e] for e in train})
    rng = np.random.default_rng(SEED); rng.shuffle(groups)
    valg = set(groups[:max(1, int(round(0.15 * len(groups))))])
    tr = [e for e in train if g[e] not in valg]
    val = [e for e in train if g[e] in valg]
    return rows, g, tr, val, test


def _auc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")


def surface_scores(texts, y, tr, val, te, veckw):
    """char-n-gram: vectorizer fit TRAIN-ONLY for C-select; refit vec+LR on
    train+val; test transform-only. (Verbatim from exp-31 surface_baseline.)"""
    t = lambda ids: [texts[i] for i in ids]
    v_tr = TfidfVectorizer(**veckw).fit(t(tr))
    Xtr, Xval = v_tr.transform(t(tr)), v_tr.transform(t(val))
    bestC = max(C_GRID_SURF, key=lambda C: _auc(y[val], LogisticRegression(C=C, max_iter=1000).fit(Xtr, y[tr]).decision_function(Xval)))
    v_f = TfidfVectorizer(**veckw).fit(t(tr + val))
    lr = LogisticRegression(C=bestC, max_iter=1000).fit(v_f.transform(t(tr + val)), y[tr + val])
    return lr.decision_function(v_f.transform(t(te)))


def probe_scores(npz_path, layer, C, y, tr, val, te):
    """commit-position probe refit at (layer, C); decision_function on test.
    (Verbatim from exp-31 surface_baseline.probe_scores.)"""
    z = np.load(npz_path, allow_pickle=True); H = z["H"].astype(np.float32)
    ri = {int(e): i for i, e in enumerate(z["eid"])}
    rows_for = lambda es: np.array([ri[e] for e in es]); col = layer + 1
    trv = rows_for(tr + val)
    sc = StandardScaler().fit(H[trv, col])
    lr = LogisticRegression(C=C, max_iter=400).fit(sc.transform(H[trv, col]), y[tr + val])
    return lr.decision_function(sc.transform(H[rows_for(te), col]))


def verbalized_scores(npz_path, te):
    """model's own P(yes) at the commit position, aligned to the test order."""
    z = np.load(npz_path, allow_pickle=True)
    ri = {int(e): i for i, e in enumerate(z["eid"])}
    p = z["p_yes"]
    return np.array([p[ri[e]] for e in te], dtype=float)


def _roc_maxtpr(y, s):
    """roc_curve, with duplicate FPR values collapsed to their MAX TPR (the top of
    each vertical ROC segment). drop_intermediate=False is REQUIRED: the default
    drops collinear vertices, after which np.interp would extrapolate across a
    dropped flat segment to a far vertex and INFLATE the low-FPR TPR (e.g. 0.7525
    vs the correct 0.75). Keeping every vertex gives the correct interpolated ROC
    value. NB the interpolated value is the convexified (randomized-threshold) ROC
    point, not necessarily reachable by a single deterministic threshold — that is
    `tpr_discrete`. (reviewer-mandated robustness)."""
    fpr, tpr, _ = roc_curve(y, s, drop_intermediate=False)
    uf = np.unique(fpr)
    ut = np.array([tpr[fpr == f].max() for f in uf])
    return uf, ut


def tpr_at_fpr(y, s, target=TARGET_FPR):
    """ROC-interpolated TPR at FPR=target, plus the discrete operating point
    (largest TPR reachable with FPR<=target) and the integer #FP that allows.

    NB at these n (n_neg<=83 in every per-slice cell) the 1% budget allows 0 false
    positives, so `tpr_discrete` is TPR at exactly 0 FP and the interpolated `tpr`
    can sit slightly above it (interpolating toward the 1-FP point). Both are stored
    so the interp-vs-deterministic gap is auditable."""
    y = np.asarray(y); s = np.asarray(s)
    if len(np.unique(y)) < 2:
        return dict(tpr=float("nan"), tpr_discrete=float("nan"), fp_allowed=None, auc=float("nan"))
    uf, ut = _roc_maxtpr(y, s)
    tpr_interp = float(np.interp(target, uf, ut))
    n_neg = int((y == 0).sum())
    fp_allowed = int(np.floor(target * n_neg))           # discrete budget at this n
    ok = uf <= (fp_allowed / n_neg) + 1e-12              # thresholds within budget
    tpr_discrete = float(ut[ok].max()) if ok.any() else 0.0
    return dict(tpr=tpr_interp, tpr_discrete=tpr_discrete, fp_allowed=fp_allowed, auc=_auc(y, s))


def _grp_resample(groups, rng):
    ug = np.unique(groups)
    drawn = rng.choice(ug, len(ug), replace=True)
    return np.concatenate([np.where(groups == g)[0] for g in drawn])


def _tpr_interp(y, s, target):
    uf, ut = _roc_maxtpr(y, s)
    return float(np.interp(target, uf, ut))


def tpr_ci(y, s, groups, target=TARGET_FPR, n_boot=N_BOOT):
    """Group-clustered bootstrap CI for TPR@target — resample the 141 SPLIT groups
    (the honest unit: rows in a group share text/structure; one collision group is
    12 rows). High variance at small n by construction; reported so the coarseness
    is visible, not hidden."""
    y = np.asarray(y); s = np.asarray(s); groups = np.asarray(groups)
    rng = np.random.default_rng(SEED); v = []
    for _ in range(n_boot):
        b = _grp_resample(groups, rng)
        if len(np.unique(y[b])) > 1:
            v.append(_tpr_interp(y[b], s[b], target))
    if not v:
        return [float("nan"), float("nan")]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def paired_delta_tpr(y, sa, sb, groups, target=TARGET_FPR, n_boot=N_BOOT):
    """Group-clustered PAIRED bootstrap of TPR_a - TPR_b at FPR=target (same
    resample applied to both scorers). The honest way to ask 'is probe > char/verb
    at the operating point?' — a CI excluding 0 is the only support for such a
    claim. At these n it is wide by construction; that is the point."""
    y = np.asarray(y); sa = np.asarray(sa); sb = np.asarray(sb); groups = np.asarray(groups)
    rng = np.random.default_rng(SEED); d = []
    for _ in range(n_boot):
        b = _grp_resample(groups, rng)
        if len(np.unique(y[b])) > 1:
            d.append(_tpr_interp(y[b], sa[b], target) - _tpr_interp(y[b], sb[b], target))
    if not d:
        return {"mean": float("nan"), "ci": [float("nan"), float("nan")]}
    return {"mean": float(np.mean(d)),
            "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "ci_excludes_0": bool(np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0)}


def _global_thr(s, yte, target):
    """smallest threshold with FPR<=target on the FULL negative pool (strict '>')."""
    neg = np.sort(s[yte == 0])[::-1]
    fp = int(np.floor(target * len(neg)))
    return float(neg[fp]) if fp < len(neg) else float("-inf")


def global_paired_delta(sa, sb, yte, groups, slices, target=TARGET_FPR, n_boot=N_BOOT):
    """Group-clustered paired bootstrap of (TPR_a - TPR_b) per slice at the GLOBAL
    threshold — each resample recomputes BOTH scorers' single full-pool threshold,
    then measures the per-slice recall gap. The honest uncertainty for the deployable
    figure (the per-slice interp Δ-CIs must NOT be reused for the global view)."""
    yte = np.asarray(yte); sa = np.asarray(sa); sb = np.asarray(sb); groups = np.asarray(groups)
    rng = np.random.default_rng(SEED)
    acc = {nm: [] for nm in slices}
    for _ in range(n_boot):
        b = _grp_resample(groups, rng)
        yb = yte[b]
        if (yb == 0).sum() < 2 or (yb == 1).sum() < 1:
            continue
        ta = _global_thr(sa[b], yb, target); tb = _global_thr(sb[b], yb, target)
        for nm, m in slices.items():
            posm = m[b] & (yb == 1)
            if posm.sum() == 0:
                continue
            acc[nm].append(float((sa[b][posm] > ta).mean() - (sb[b][posm] > tb).mean()))
    out = {}
    for nm, v in acc.items():
        if not v:
            out[nm] = {"mean": float("nan"), "ci": [float("nan"), float("nan")]}
            continue
        lo, hi = float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
        out[nm] = {"mean": float(np.mean(v)), "ci": [lo, hi], "ci_excludes_0": bool(lo > 0 or hi < 0)}
    return out


def main():
    rows, g, tr, val, te = load_split()
    y = np.array([int(rows[i]["label"]) for i in range(len(rows))])
    texts = [rows[i].get("code") or "" for i in range(len(rows))]
    yte = y[te]

    # per-test-position metadata (aligned to `te` order)
    lang_te = np.array([(rows[e].get("lang") or "").lower() for e in te])
    grp_te = np.array([g[e] for e in te])
    # match negatives to CWE via the pair group's positive CWE. NB groups are the
    # 141 SPLIT groups; rows in a group are balanced (146/146) but not strictly 1
    # pair each (one collision group `func::dbhelper.py::add_input` holds 6+6, all
    # CWE-089). Invariant below enforces a SINGLE positive CWE per group, so the
    # match is well-defined even though it is not literally pair-by-pair.
    grp_pos_cwe = {}
    for e in te:
        if y[e] == 1 and rows[e].get("cwe"):
            c = rows[e]["cwe"]
            if grp_pos_cwe.get(g[e], c) != c:
                raise RuntimeError(f"group {g[e]} has >1 positive CWE: {grp_pos_cwe[g[e]]} vs {c}")
            grp_pos_cwe[g[e]] = c
    cwe_te = np.array([rows[e]["cwe"] if (y[e] == 1 and rows[e].get("cwe"))
                       else grp_pos_cwe.get(g[e], "NONE") for e in te])
    if (cwe_te == "NONE").any():
        raise RuntimeError(f"{int((cwe_te=='NONE').sum())} test rows unmatched to a CWE")

    # slice definitions: name -> boolean mask over test positions
    slices = {"overall": np.ones(len(te), bool),
              "python": lang_te == "python",
              "c_cpp": np.isin(lang_te, ["c", "cpp", "c++"])}
    if int(slices["python"].sum() + slices["c_cpp"].sum()) != len(te):
        raise RuntimeError("language slices do not partition the test set")
    pos_per_cwe = {c: int(((cwe_te == c) & (yte == 1)).sum()) for c in set(cwe_te) if c != "NONE"}
    cwe_kept = sorted([c for c, n in pos_per_cwe.items() if n >= CWE_MIN_POS],
                      key=lambda c: (-pos_per_cwe[c], c))   # deterministic tie-break
    cwe_dropped = {c: pos_per_cwe[c] for c in pos_per_cwe if pos_per_cwe[c] < CWE_MIN_POS}
    for c in cwe_kept:
        slices[c] = cwe_te == c
        yy = yte[slices[c]]
        if (yy == 1).sum() == 0 or (yy == 0).sum() == 0:
            raise RuntimeError(f"CWE slice {c} missing a class: {int((yy==1).sum())}+/{int((yy==0).sum())}-")

    def slice_block(s_all):
        out = {}
        for nm, mask in slices.items():
            yy, ss, gg = yte[mask], s_all[mask], grp_te[mask]
            r = tpr_at_fpr(yy, ss)
            r.update(n_pos=int((yy == 1).sum()), n_neg=int((yy == 0).sum()),
                     tpr_ci=tpr_ci(yy, ss, gg))
            out[nm] = r
        return out

    def paired_block(sa, sb):
        """probe-minus-other paired Δ at FPR=0.01, per slice (group-clustered)."""
        return {nm: paired_delta_tpr(yte[m], sa[m], sb[m], grp_te[m]) for nm, m in slices.items()}

    def global_block(s_all):
        """Companion 'deployed monitor' view: ONE threshold (smallest with FPR<=
        target on the FULL test negative pool), then TPR within each slice at that
        single frozen threshold — comparable across slices, unlike the per-slice
        primary. Strict '>' against the threshold."""
        neg = np.sort(s_all[yte == 0])[::-1]            # negatives, descending
        fp = int(np.floor(TARGET_FPR * len(neg)))       # allowed FP on the full pool
        thr = float(neg[fp]) if fp < len(neg) else float("-inf")
        out = {"threshold": thr, "fp_allowed": fp, "global_fpr": float((s_all[yte == 0] > thr).mean())}
        for nm, m in slices.items():
            pos = s_all[m][yte[m] == 1]
            out[nm] = {"tpr": float((pos > thr).mean()) if len(pos) else float("nan"),
                       "n_pos": int(len(pos))}
        return out

    # ---- char-n-gram (model-independent lexical CEILING/control), gated to exp-31 ----
    char_s = surface_scores(texts, y, tr, val, te, CHAR_KW)
    char_auc = _auc(yte, char_s)
    e31 = json.loads((E31 / "results/surface_vs_probe.json").read_text())
    char_stored = e31["surface"]["char_3_5_100k"]["auc"]
    if abs(char_auc - char_stored) > CHAR_TOL:
        raise RuntimeError(f"char gate: {char_auc:.4f} != exp-31 {char_stored:.4f}")
    char_block = slice_block(char_s)

    e30 = json.loads((E30 / "results/introspection_probe.json").read_text())["models"]
    out = {"target_fpr": TARGET_FPR, "n_test": len(te), "n_groups": int(len(np.unique(grp_te))),
           "n_boot": N_BOOT, "seed": SEED,
           "threshold_method_primary": "per-slice ROC-interpolated TPR at FPR=0.01 "
                                       "(each slice controls FPR on its OWN matched negatives; "
                                       "NOT comparable across slices, NOT a single deployed threshold)",
           "threshold_method_companion": "global: one threshold per scorer at FPR=0.01 on the FULL "
                                          "test negative pool, TPR by slice (comparable; deployable view)",
           "tpr_metric_note": "FIGURE HEADLINES `global_threshold` (deployable, comparable across slices). "
                              "Per-slice `tpr` (interp) is convexified TPR@1%FPR but at all language/CWE slices "
                              "(n_neg<=83) the 1% budget allows 0 FP, so interp is dominated by the single 1-FP "
                              "point (e.g. python char interp 0.37 vs `tpr_discrete` 0.02). Use interp/discrete "
                              "as a secondary matched-slice table only, labelled; do not headline either.",
           "negatives": "matched to CWE via split group (balanced rows, one unique positive CWE/group)",
           "char_config": "char_3_5_100k (exp-31 strongest char = conservative lexical CEILING)",
           "char_auc": char_auc, "char_auc_stored": char_stored,
           "cwe_kept": cwe_kept, "cwe_dropped_lt10pos": cwe_dropped,
           "slices": list(slices.keys()), "models": {}}

    for name, slug in MODELS.items():
        npz = E30 / "hidden" / f"lasttoken_hidden_{slug}.npz"
        L, C = e30[name]["deployable"]["layer"], e30[name]["deployable"]["C"]
        # probe (primed), gated to exp-30 deployable test AUC
        ps = probe_scores(npz, L, C, y, tr, val, te)
        p_auc = _auc(yte, ps)
        if abs(p_auc - e30[name]["deployable"]["test_auc"]) > PROBE_TOL:
            raise RuntimeError(f"{name} probe gate: {p_auc:.4f} != exp-30 {e30[name]['deployable']['test_auc']:.4f}")
        # verbalized, gated to exp-30 stored verbalized test AUC
        vs = verbalized_scores(npz, te)
        v_auc = _auc(yte, vs)
        if abs(v_auc - e30[name]["verbalized"]["test_auc"]) > VERB_TOL:
            raise RuntimeError(f"{name} verbalized gate: {v_auc:.4f} != exp-30 {e30[name]['verbalized']['test_auc']:.4f}")
        out["models"][name] = {
            "deployable_layer": L, "deployable_C": C,
            "gates": {"probe_auc": p_auc, "probe_auc_stored": e30[name]["deployable"]["test_auc"],
                      "verbalized_auc": v_auc, "verbalized_auc_stored": e30[name]["verbalized"]["test_auc"]},
            "methods": {"probe": slice_block(ps), "char_ngram": char_block, "verbalized": slice_block(vs)},
            "paired_delta": {"probe_minus_char": paired_block(ps, char_s),
                             "probe_minus_verb": paired_block(ps, vs),
                             "char_minus_verb": paired_block(char_s, vs)},
            "global_threshold": {"probe": global_block(ps), "char_ngram": global_block(char_s),
                                 "verbalized": global_block(vs)},
            "global_paired_delta": {"probe_minus_char": global_paired_delta(ps, char_s, yte, grp_te, slices),
                                    "probe_minus_verb": global_paired_delta(ps, vs, yte, grp_te, slices)}}
        # console sanity — GLOBAL-threshold view (the figure's basis); '*' = global
        # probe-minus-char paired CI excludes 0.
        gt = out["models"][name]["global_threshold"]
        gpmc = out["models"][name]["global_paired_delta"]["probe_minus_char"]
        print(f"\n=== {name} (probe L{L} C={C}, AUC {p_auc:.3f}; char {char_auc:.3f}; verb {v_auc:.3f}) ===")
        print(f"  GLOBAL-threshold TPR (one frozen threshold, FPR~{gt['probe']['global_fpr']:.3f} on full pool)")
        print(f"{'slice':<10} {'n+':>4}   probe  char   verb    Δ(probe-char)[CI]")
        for nm in slices:
            np_ = out["models"][name]["methods"]["probe"][nm]["n_pos"]
            d = gpmc[nm]; sep = "*" if d.get("ci_excludes_0") else " "
            print(f"{nm:<10} {np_:>4}   {gt['probe'][nm]['tpr']:.2f}  {gt['char_ngram'][nm]['tpr']:.2f}  "
                  f"{gt['verbalized'][nm]['tpr']:.2f}   {d['mean']:+.2f}[{d['ci'][0]:+.2f},{d['ci'][1]:+.2f}]{sep}")
        n_sep = sum(1 for nm in slices if gpmc[nm].get("ci_excludes_0"))
        print(f"  global probe>char CI-separated slices: {n_sep}/{len(slices)}")

    outp = HERE / "results/operating_point.json"
    outp.write_text(json.dumps(out, indent=2))
    print(f"\n[done] -> {outp}")
    print(f"CWE slices kept (>={CWE_MIN_POS} pos): {cwe_kept}")
    print(f"CWE dropped (<{CWE_MIN_POS} pos): {cwe_dropped}")


def _selftest():
    """Tiny known-answer checks for the metric (reviewer-mandated)."""
    # 4 pos, 4 neg. scores: 3 pos clearly above all neg; 1 pos buried below all neg.
    y = np.array([1, 1, 1, 0, 0, 0, 0, 1])
    s = np.array([0.9, 0.8, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1])
    r = tpr_at_fpr(y, s, target=0.01)
    # n_neg=4 -> fp_allowed=floor(0.04)=0 -> discrete TPR at 0 FP = 3/4
    assert r["fp_allowed"] == 0 and abs(r["tpr_discrete"] - 0.75) < 1e-9, r
    # at FPR=0 the top of the vertical run is 0.75; interp at 0.01 stays 0.75 here
    assert abs(r["tpr"] - 0.75) < 1e-9, r
    # perfect separation -> TPR 1.0; flipped scores -> TPR 0.0 (orientation sanity)
    yp = np.array([1, 1, 0, 0]); sp = np.array([0.9, 0.8, 0.2, 0.1])
    assert abs(tpr_at_fpr(yp, sp, 0.01)["tpr_discrete"] - 1.0) < 1e-9
    assert abs(tpr_at_fpr(yp, -sp, 0.01)["tpr_discrete"] - 0.0) < 1e-9
    print("[selftest] tpr_at_fpr OK")


if __name__ == "__main__":
    _selftest()
    main()
