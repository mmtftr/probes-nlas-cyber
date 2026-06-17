# [ai-generated]
"""Example-level surface baseline vs the commit-position probe (the lexical control).

The exp-24 control lifted to the example level: char-n-gram + token-unigram
classifiers on the RAW FUNCTION TEXT, same group-clean split, vectorizer fit
TRAIN-ONLY (C selected on val, refit on train+val, test transform-only), scored on
the held-out 292 — compared to the commit-position probe (refit at its deployable
layer/C on the kept hidden states). All CIs and paired-Δ use a PAIR-CLUSTERED
bootstrap (resample the 141 vuln/fix groups, not the 292 rows — rows within a pair
share surface n-grams, so the group is the honest unit; reviewer-mandated).

Several char configs are reported; the STRONGEST (max test AUC) is the conservative
ceiling the probe must clear. probe > strongest-char (clustered Δ-CI excludes 0) ⇒
above the lexical ceiling; else folds into the lexical story (exp-20/24/29).

CPU; reuses kept hidden npz + dataset text.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
E30 = REPO / "plans/cross-model-probe-generalization/30-last-token-introspection"
E17 = REPO / "plans/cross-model-probe-generalization/17-verbalized-logit-dump/results"
DATASET = REPO / "data/dataset.jsonl"
SPLIT = REPO / "data/sven_split_meta.json"

MODELS = {
    "Qwen2.5-Coder-32B-Instruct": "Qwen_Qwen2.5-Coder-32B-Instruct",
    "Qwen2.5-Coder-7B-Instruct": "Qwen_Qwen2.5-Coder-7B-Instruct",
    "gemma-3-1b-it": "google_gemma-3-1b-it", "gemma-3-4b-it": "google_gemma-3-4b-it",
    "gemma-3-12b-it": "google_gemma-3-12b-it", "gemma-3-27b-it": "google_gemma-3-27b-it",
}
# char configs: exp-31 default + two STRONGER (codex/Opus sensitivity); strongest = ceiling
CHAR_CONFIGS = [
    ("char_wb_3_5_50k", dict(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=50000)),
    ("char_3_5_100k", dict(analyzer="char", ngram_range=(3, 5), min_df=2, max_features=100000)),
    ("char_3_7_200k", dict(analyzer="char", ngram_range=(3, 7), min_df=2, max_features=200000)),
]
UNI_CONFIG = dict(analyzer="word", ngram_range=(1, 1), min_df=2, max_features=50000)
C_GRID = [1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0, 100.0]
SEED, N_BOOT, PROBE_TOL = 42, 1000, 2e-3


def pair_group_key(r):
    if r.get("_origin_repo"):
        return f"repo::{r['_origin_repo']}"
    fn, fu = r.get("_file_name") or "", r.get("_func_name") or ""
    return f"func::{fn}::{fu}" if (fn or fu) else f"row::{hashlib.sha1((r.get('code') or '').encode()).hexdigest()[:12]}"


def load_split():
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
    te_groups = np.array([g[e] for e in test])
    return rows, tr, val, test, te_groups


def _auc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")


def _grp_indices(te_groups, rng):
    ug = np.unique(te_groups)
    drawn = rng.choice(ug, len(ug), replace=True)
    return np.concatenate([np.where(te_groups == g)[0] for g in drawn])


def _ci(y, s, tg):
    rng = np.random.default_rng(SEED); v = []
    for _ in range(N_BOOT):
        b = _grp_indices(tg, rng)
        if len(np.unique(y[b])) > 1:
            v.append(roc_auc_score(y[b], s[b]))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def _paired(y, sa, sb, tg):
    rng = np.random.default_rng(SEED); d = []
    for _ in range(N_BOOT):
        b = _grp_indices(tg, rng)
        if len(np.unique(y[b])) > 1:
            d.append(roc_auc_score(y[b], sa[b]) - roc_auc_score(y[b], sb[b]))
    return {"mean": float(np.mean(d)), "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


def surface_scores(texts, y, tr, val, te, veckw):
    """vectorizer fit TRAIN-ONLY for C-select; refit vec+LR on train+val; test transform-only."""
    t = lambda ids: [texts[i] for i in ids]
    v_tr = TfidfVectorizer(**veckw).fit(t(tr))
    Xtr, Xval = v_tr.transform(t(tr)), v_tr.transform(t(val))
    bestC = max(C_GRID, key=lambda C: _auc(y[val], LogisticRegression(C=C, max_iter=1000).fit(Xtr, y[tr]).decision_function(Xval)))
    v_f = TfidfVectorizer(**veckw).fit(t(tr + val))
    lr = LogisticRegression(C=bestC, max_iter=1000).fit(v_f.transform(t(tr + val)), y[tr + val])
    return lr.decision_function(v_f.transform(t(te))), bestC


def probe_scores(npz_path, layer, C, y, tr, val, te):
    z = np.load(npz_path, allow_pickle=True); H = z["H"].astype(np.float32)
    ri = {int(e): i for i, e in enumerate(z["eid"])}
    rows_for = lambda es: np.array([ri[e] for e in es]); col = layer + 1
    trv = rows_for(tr + val)
    sc = StandardScaler().fit(H[trv, col])
    lr = LogisticRegression(C=C, max_iter=400).fit(sc.transform(H[trv, col]), y[tr + val])
    return lr.decision_function(sc.transform(H[rows_for(te), col]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primed-json", default=str(E30 / "results/introspection_probe.json"))
    ap.add_argument("--neutral-json", default=str(HERE / "results/introspection_probe_neutral.json"))
    ap.add_argument("--hidden-dir", default=str(E30 / "hidden"))
    ap.add_argument("--out", default=str(HERE / "results/surface_vs_probe.json"))
    args = ap.parse_args()

    rows, tr, val, te, tg = load_split()
    y = np.array([int(rows[i]["label"]) for i in range(len(rows))])
    texts = [rows[i].get("code") or "" for i in range(len(rows))]
    yte = y[te]
    primed = json.loads(Path(args.primed_json).read_text())["models"]
    neutral = (json.loads(Path(args.neutral_json).read_text())["models"]
               if Path(args.neutral_json).exists() else {})

    # surface (model-independent) — all char configs + unigram
    surf = {}
    for nm, kw in CHAR_CONFIGS + [("token_unigram", UNI_CONFIG)]:
        s, C = surface_scores(texts, y, tr, val, te, kw)
        surf[nm] = {"scores": s, "auc": _auc(yte, s), "C": C, "C_at_edge": C in (min(C_GRID), max(C_GRID)),
                    "ci": _ci(yte, s, tg)}
        print(f"[surface] {nm}: AUC={surf[nm]['auc']:.3f} (C={C}{' EDGE' if surf[nm]['C_at_edge'] else ''}) CI{surf[nm]['ci']}")
    char_best = max(CHAR_CONFIGS, key=lambda c: surf[c[0]]["auc"])[0]   # strongest char = conservative ceiling
    print(f"[surface] strongest char = {char_best} ({surf[char_best]['auc']:.3f})")

    out = {"n_test": len(te), "n_groups": int(len(np.unique(tg))), "n_boot": N_BOOT,
           "bootstrap": "pair-clustered (resample vuln/fix groups)",
           "surface": {k: {kk: v[kk] for kk in ("auc", "C", "C_at_edge", "ci")} for k, v in surf.items()},
           "strongest_char": char_best, "models": {}}
    for name, slug in MODELS.items():
        rec = {}
        for tag, src, glob in [("primed", primed, "lasttoken_hidden_{}.npz"),
                               ("neutral", neutral, "lasttoken_hidden_neutral_{}.npz")]:
            if name not in src:
                continue
            L, C = src[name]["deployable"]["layer"], src[name]["deployable"]["C"]
            npz = Path(args.hidden_dir) / glob.format(slug)
            if not npz.exists():
                print(f"[skip] {name} {tag}: missing {npz}"); continue
            sp = probe_scores(npz, L, C, y, tr, val, te)
            pa = _auc(yte, sp)
            gate = abs(pa - src[name]["deployable"]["test_auc"])
            if gate > PROBE_TOL:
                raise SystemExit(f"[gate] {name} {tag}: refit AUC {pa:.4f} != exp-30 {src[name]['deployable']['test_auc']:.4f}")
            rec[tag] = {"layer": L, "C": C, "probe_auc": pa, "probe_ci": _ci(yte, sp, tg),
                        "refit_gate_diff": gate,
                        "delta_vs_strongest_char": _paired(yte, sp, surf[char_best]["scores"], tg),
                        "delta_vs_char_default": _paired(yte, sp, surf["char_wb_3_5_50k"]["scores"], tg),
                        "delta_vs_unigram": _paired(yte, sp, surf["token_unigram"]["scores"], tg)}
            d = rec[tag]["delta_vs_strongest_char"]
            sep = "*" if (d["ci"][0] > 0 or d["ci"][1] < 0) else " "
            print(f"[{name}/{tag}] probe={pa:.3f} vs strongest-char({char_best})={surf[char_best]['auc']:.3f} "
                  f"Δ={d['mean']:+.3f}[{d['ci'][0]:+.3f},{d['ci'][1]:+.3f}]{sep}")
        out["models"][name] = rec
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
