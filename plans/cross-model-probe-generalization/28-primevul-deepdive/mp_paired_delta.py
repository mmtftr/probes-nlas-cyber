# [ai-generated]
"""exp-28 addendum 2 — PAIRED probe−lexical Δ-bootstrap under PV matched-patch.

The decisive contrast exp-27 named: per bootstrap rep, resample test PAIRS once
and score probe AUC − lexical AUC on the SAME resample → percentile CI on Δ.
Removes shared resampling variance; a Δ-CI excluding 0 is a CI-separated
probe-over-lexical margin (the project has none so far).

Probe score per CWE = SVEN-trained probe where available (the §C verdict
cells: 125/416/476/787/190), else the PV-trained probe. Lexical = char-ngram
and token-unigram (cached stage-A scores). Same pools as pv_matchedpair.json.
Writes results/pv_matchedpair_delta.json.
"""
import json
import numpy as np
from pathlib import Path
import pv_deepdive as D

pv = D.pv_state()
feas, _ = D.feasible_cwes(pv)
test_union, Hs, Us = D.build_surface_scores(pv, feas)
Lg_pv = D.load_logits("logits_pv.npz", len(pv["eids"]))
Lg_sv = D.load_logits("logits_pv_svenprobes.npz", len(pv["eids"]))

pos_test_of = {c: [e for e in pv["test_eids"] if pv["cwe_of"][e] == c] for c in feas}


def to_global(s):
    g = np.full(len(pv["eids"]), np.nan, np.float32)
    g[test_union] = s
    return g


def pool(pos_eids):
    fixes = [pv["fix_of"][v] for v in pos_eids]
    pt, nt = D.ctoks(pv, pos_eids), D.ctoks(pv, fixes)
    ev = np.concatenate([pt, nt])
    return ev, pv["y_tok"][ev]


out = {"analysis": "pv_matchedpair_paired_delta", "metric": "tokens_code_auc",
       "note": "paired bootstrap over PAIRS (1000 reps, same resample scores all "
               "models per rep); delta = probe_auc - lexical_auc; CI excluding 0 "
               "= CI-separated margin. probe = SVEN-trained for shared CWEs "
               "(125/416/476/787/190), PV-trained otherwise.", "table": {}}

for c in feas:
    pe = pos_test_of[c]
    probe = Lg_sv[c] if c in D.SHARED_MEM else Lg_pv[c]
    probe_src = "sven" if c in D.SHARED_MEM else "pv"
    scores = {"probe": probe, "char": to_global(Hs[c]), "uni": to_global(Us[c])}

    ev0, y0 = pool(pe)
    pt_full = {k: D.auc_or_nan(y0, s[ev0]) for k, s in scores.items()}

    def rep(r_):
        r = np.random.default_rng(D.SEED + 9000 + r_)   # same seeds as stage C
        ps = [pe[j] for j in r.integers(0, len(pe), len(pe))]
        ev, y = pool(ps)
        if len(np.unique(y)) < 2:
            return None
        return {k: D.fast_auc_batch(s[ev][None, :], y)[0] for k, s in scores.items()}

    reps = [x for x in D._pmap(rep, list(range(D.N_BOOT_DIAG)), f"delta-{c}") if x]
    row = {"family": D.fam_of(c), "n_test_pairs": len(pe), "probe_source": probe_src,
           "point": {k: pt_full[k] for k in scores}, "n_reps": len(reps)}
    for lex in ("char", "uni"):
        ds = np.array([x["probe"] - x[lex] for x in reps])
        row[f"delta_probe_minus_{lex}"] = {
            "point": pt_full["probe"] - pt_full[lex],
            "ci": [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))],
            "ci_excludes_0": bool(np.percentile(ds, 2.5) > 0 or np.percentile(ds, 97.5) < 0)}
    out["table"][c] = row
    print(f"[delta] {c} ({row['family']}, probe={probe_src}, n={len(pe)}): "
          f"probe {pt_full['probe']:.3f} | d_char {row['delta_probe_minus_char']['point']:+.3f} "
          f"CI [{row['delta_probe_minus_char']['ci'][0]:+.3f},{row['delta_probe_minus_char']['ci'][1]:+.3f}]"
          f"{'*' if row['delta_probe_minus_char']['ci_excludes_0'] else ''} "
          f"| d_uni {row['delta_probe_minus_uni']['point']:+.3f} "
          f"CI [{row['delta_probe_minus_uni']['ci'][0]:+.3f},{row['delta_probe_minus_uni']['ci'][1]:+.3f}]"
          f"{'*' if row['delta_probe_minus_uni']['ci_excludes_0'] else ''}")

(D.RES / "pv_matchedpair_delta.json").write_text(
    json.dumps(out, indent=2, default=lambda x: None if x != x else x))
print("wrote results/pv_matchedpair_delta.json")
