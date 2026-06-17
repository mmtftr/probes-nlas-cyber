# [ai-generated]
"""exp-28 addendum — token-unigram LR under PV matched-patch.

Triggered by exp-27's split verdict (char-ngram chance-consistent under
matched-patch on SVEN, but TOKEN-UNIGRAM survives on CWE-125). Adds the
unigram column to stage C using the SAME cached surface scores
(assets/surface_scores.npz, U rows) and the SAME mp_eval/mp_ci code paths
already audited by both reviewers. Writes results/pv_matchedpair_unigram.json.
"""
import json
import numpy as np
from pathlib import Path
import pv_deepdive as D

pv = D.pv_state()
feas, test_pos = D.feasible_cwes(pv)
test_union, Hs, Us = D.build_surface_scores(pv, feas)

pos_test_of = {c: [e for e in pv["test_eids"] if pv["cwe_of"][e] == c] for c in feas}


def to_global(s):
    g = np.full(len(pv["eids"]), np.nan, np.float32)
    g[test_union] = s
    return g


def mp_eval(score, pos_eids):
    fixes = [pv["fix_of"][v] for v in pos_eids]
    pt, nt = D.ctoks(pv, pos_eids), D.ctoks(pv, fixes)
    ev = np.concatenate([pt, nt])
    return D.auc_or_nan(pv["y_tok"][ev], score[ev])


def mp_ci(score, pos_eids, n_boot=D.N_BOOT_DIAG):
    def rep(r_):
        r = np.random.default_rng(D.SEED + 9000 + r_)
        ps = [pos_eids[j] for j in r.integers(0, len(pos_eids), len(pos_eids))]
        return mp_eval(score, ps)
    b = np.array(D._pmap(rep, list(range(n_boot)), "mpU-ci"))
    b = b[~np.isnan(b)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), int(b.size)]


out = {"analysis": "pv_matchedpair_unigram_addendum", "metric": "tokens_code_auc",
       "note": "token-unigram LR (exp-24 U block, train-vocab) under the SAME "
               "matched-patch pools/bootstrap as results/pv_matchedpair.json; "
               "added after exp-27's split verdict (unigram survives mp on 125 "
               "on SVEN). Same seeds as stage C -> CIs comparable.",
       "table": {}}
for c in feas:
    g = to_global(Us[c])
    pe = pos_test_of[c]
    a = mp_eval(g, pe)
    ci = mp_ci(g, pe)
    out["table"][c] = {"family": D.fam_of(c), "n_test_pairs": len(pe),
                       "unigram_mp_auc": a, "ci": ci}
    print(f"[mpU] {c} ({D.fam_of(c)}, n={len(pe)}): {a:.3f} CI [{ci[0]:.3f},{ci[1]:.3f}]")

(D.RES / "pv_matchedpair_unigram.json").write_text(
    json.dumps(out, indent=2, default=lambda x: None if x != x else x))
print("wrote results/pv_matchedpair_unigram.json")
