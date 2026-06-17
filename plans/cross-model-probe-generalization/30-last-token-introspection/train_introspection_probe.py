# [ai-generated]
"""Last-token introspection probe + honest nulls + confound baselines.

Linear L2 logistic regression on the `Assistant:`-boundary hidden state (one
vector/example), example-level label. d (hidden) >> n (train ~1138), so the design
is built to NOT fool itself. Headline = held-out test AUC at the val-selected
(layer, C); a positive claim must clear BOTH nulls AND the confounds:

  LABEL-PERMUTATION NULL  — rerun the full select-layer pipeline on shuffled
      train/val labels (test labels intact), N_PERM times; report p=(1+#≥)/(N+1).
      Catches selection / d>>n overfit inflation.
  RANDOM-DIRECTION NULL   — at the deployable layer, AUC of random unit directions
      in standardized space (untrained); percentile of the real probe. Catches
      hidden-geometry artifacts.
  CONFOUNDS               — language indicator (python=1) and code-length AUC, plus
      WITHIN-LANGUAGE probe AUC (py-only / C-only test). A probe that only beats
      chance because it reads language is not "introspective" (cf exp-23: ~64% of
      the headline AUC margin is language).

All example-level (SECONDARY metric); the token-level headline 0.75-0.82 is
untouched. Compared like-with-like to verbalized P(yes) at the SAME position
(HARD gate vs exp-17) and the exp-29 code-token reads.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


# inlined from train_eval (model-independent, no torch)
def pair_group_key(row: dict) -> str:
    repo = row.get("_origin_repo")
    if repo:
        return f"repo::{repo}"
    fn = row.get("_file_name") or ""
    func = row.get("_func_name") or ""
    if fn or func:
        return f"func::{fn}::{func}"
    return f"row::{hashlib.sha1((row.get('code') or '').encode()).hexdigest()[:12]}"


def load_or_make_split(dataset_path, split_path):
    rows = [json.loads(l) for l in Path(dataset_path).open()]
    g = {i: pair_group_key(r) for i, r in enumerate(rows)}
    heldout = set(json.loads(Path(split_path).read_text())["heldout_groups"])
    test = {e for e, gg in g.items() if gg in heldout}
    return rows, {e for e in g if e not in test}, test
E17 = REPO / "plans/cross-model-probe-generalization/17-verbalized-logit-dump/results"
E29 = REPO / "plans/cross-model-probe-generalization/29-last-token-readout/results/last_token_readout.json"
DATASET = REPO / "data/dataset.jsonl"
SPLIT = REPO / "data/sven_split_meta.json"

MODELS = {
    "Qwen2.5-Coder-32B-Instruct": ("Qwen_Qwen2.5-Coder-32B-Instruct", "logitdump_Qwen_Qwen2.5-Coder-32B-Instruct"),
    "Qwen2.5-Coder-7B-Instruct": ("Qwen_Qwen2.5-Coder-7B-Instruct", "logitdump_Qwen_Qwen2.5-Coder-7B-Instruct"),
    "gemma-3-1b-it": ("google_gemma-3-1b-it", "logitdump_google_gemma-3-1b-it"),
    "gemma-3-4b-it": ("google_gemma-3-4b-it", "logitdump_google_gemma-3-4b-it"),
    "gemma-3-12b-it": ("google_gemma-3-12b-it", "logitdump_google_gemma-3-12b-it"),
    "gemma-3-27b-it": ("google_gemma-3-27b-it", "logitdump_google_gemma-3-27b-it"),
}
C_GRID = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
SEED = 42
MAX_ITER = 400


def _auc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")


def _val_split(train_eids, rows, frac=0.15, seed=SEED):
    groups = sorted({pair_group_key(rows[e]) for e in train_eids})
    rng = np.random.default_rng(seed); rng.shuffle(groups)
    val_g = set(groups[:max(1, int(round(frac * len(groups))))])
    val = {e for e in train_eids if pair_group_key(rows[e]) in val_g}
    return sorted(e for e in train_eids if e not in val), sorted(val)


def _lr(C):
    return LogisticRegression(C=C, max_iter=MAX_ITER, solver="lbfgs")


def _fit_score(Xtr, ytr, Xeval, C):
    return _lr(C).fit(Xtr, ytr).decision_function(Xeval)


def _select_task(L, C, Atr, Aval, ytr, yva):
    """val AUC of layer L at regularization C (module-level for loky)."""
    return (L, C, _auc(yva, _fit_score(Atr[L], ytr, Aval[L], C)))


def _oracle_task(L, Atr, Aval, Atrva, Ate, ytr, yva, ytrva, yte, c_grid):
    """per-layer: best-C-on-val, then TEST AUC (the oracle upper bound)."""
    bc = max(c_grid, key=lambda C: _auc(yva, _fit_score(Atr[L], ytr, Aval[L], C)))
    return (L, _auc(yte, _fit_score(Atrva[L], ytrva, Ate[L], bc)))


def _perm_worker(seed, Atr, Aval, Atrva, Ate, ytr0, yva0, yte, C):
    """One label-permutation draw (module-level so loky memmaps the big stacks).

    Shuffle the combined train+val labels (test labels untouched), reselect the
    best layer by val AUC at fixed C, refit on train+val, return TEST AUC vs TRUE
    yte. Stacks are [n_layers, n_rows, hidden]; row order matches the real run
    (Atr/Aval = tr/val under the tr-fit scaler, Atrva/Ate under the trva-fit one).
    """
    rng = np.random.default_rng(seed)
    n_tr = len(ytr0)
    comb = np.concatenate([ytr0, yva0]).copy(); rng.shuffle(comb)
    ytr_p, yva_p = comb[:n_tr], comb[n_tr:]
    bestL, bestA = 0, -1.0
    for L in range(Atr.shape[0]):
        a = _auc(yva_p, _fit_score(Atr[L], ytr_p, Aval[L], C))
        if a == a and a > bestA:
            bestA, bestL = a, L
    return _auc(yte, _fit_score(Atrva[bestL], comb, Ate[bestL], C))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden-dir", required=True)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--n-rand", type=int, default=1000)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--dataset", default=str(DATASET))
    ap.add_argument("--split", default=str(SPLIT))
    ap.add_argument("--hidden-glob", default="lasttoken_hidden_{slug}.npz",
                    help="npz name template; neutral run uses lasttoken_hidden_neutral_{slug}.npz")
    ap.add_argument("--no-verbalized-gate", action="store_true",
                    help="neutral prompt: skip the npz-p_yes==exp17 gate (still compares vs primed verbalized)")
    ap.add_argument("--out", default=None, help="output JSON path (default exp-30 results/)")
    args = ap.parse_args()

    rows, train_eids, test_eids = load_or_make_split(args.dataset, args.split)
    true_label = np.array([int(rows[i]["label"]) for i in range(len(rows))])
    lang = np.array([(rows[i].get("lang") or "").lower() for i in range(len(rows))])
    code_len = np.array([len(rows[i].get("code") or "") for i in range(len(rows))], float)
    tr_e, va_e = _val_split(train_eids, rows)

    out_path = Path(args.out) if args.out else (HERE / "results/introspection_probe.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # resume: keep already-computed models (incremental write -> timeout-safe)
    out = (json.loads(out_path.read_text()) if out_path.exists()
           else {"n_perm": args.n_perm, "n_rand": args.n_rand, "C_grid": C_GRID, "models": {}})
    for name, (npz_slug, e29_slug) in MODELS.items():
        if name in out["models"]:
            print(f"[resume] {name}: already in JSON, skip", file=sys.stderr); continue
        p = Path(args.hidden_dir) / args.hidden_glob.format(slug=npz_slug)
        if not p.exists():
            print(f"[skip] {name}: missing {p}", file=sys.stderr); continue
        out["models"][name] = run_model(name, npz_slug, e29_slug, p, rows, true_label,
                                        lang, code_len, tr_e, va_e, test_eids, args)
        out_path.write_text(json.dumps(out, indent=2))   # write after EACH model
    print(f"[done] {len(out['models'])} models -> {out_path}", file=sys.stderr)


def run_model(name, npz_slug, e29_slug, npz_path, rows, true_label, lang, code_len,
              tr_e, va_e, test_eids, args):
    z = np.load(npz_path, allow_pickle=True)
    H = z["H"].astype(np.float32)                       # [n, n_layers+1, hidden]
    eid = z["eid"]; p_yes = z["p_yes"]
    assert np.isfinite(H).all(), f"{name}: non-finite H"
    by = {int(e): i for i, e in enumerate(eid)}
    n_layers = H.shape[1] - 1
    y = true_label[eid]                                 # aligned to H rows
    idx = lambda es: np.array([by[e] for e in es if e in by])
    tr, va, te = idx(tr_e), idx(va_e), idx(sorted(test_eids))
    trva = np.concatenate([tr, va])
    yte = y[te]

    # Precompute per-layer standardized matrices ONCE (standardization is
    # label-independent), stacked to [n_layers, n_rows, hidden] so the parallel
    # null can pass them as args -> loky memmaps + shares them across workers.
    Atr, Aval, Atrva, Ate = [], [], [], []
    for L in range(n_layers):
        col = L + 1                                     # repo-layer L = hidden_states[L+1]
        s_sel = StandardScaler().fit(H[tr, col]);  Atr.append(s_sel.transform(H[tr, col])); Aval.append(s_sel.transform(H[va, col]))
        s_dep = StandardScaler().fit(H[trva, col]); Atrva.append(s_dep.transform(H[trva, col])); Ate.append(s_dep.transform(H[te, col]))
    Atr, Aval, Atrva, Ate = np.stack(Atr), np.stack(Aval), np.stack(Atrva), np.stack(Ate)
    del H                                               # free the raw hidden states

    def test_at(ytrva, L, C):
        return _fit_score(Atrva[L], ytrva, Ate[L], C)

    # loky memmaps the big stacks across workers; a FRESH Parallel per phase
    # (reusing one across phases tripped a memmap-registry KeyError on the cluster).
    def P():
        return Parallel(n_jobs=args.n_jobs, max_nbytes="1M")

    # ---- real probe: parallel layer×C val sweep, pick best-on-val ----
    sel = P()(delayed(_select_task)(L, C, Atr, Aval, y[tr], y[va])
              for L in range(n_layers) for C in C_GRID)
    Lstar, Cstar, val_auc = max((r for r in sel if r[2] == r[2]), key=lambda r: r[2])
    s_test = test_at(y[trva], Lstar, Cstar)
    test_auc = _auc(yte, s_test)
    c_edge = Cstar in (min(C_GRID), max(C_GRID))

    # ---- oracle layer (upper bound: best TEST AUC at each layer's own best-val-C) ----
    orc = P()(delayed(_oracle_task)(L, Atr, Aval, Atrva, Ate, y[tr], y[va], y[trva], yte, C_GRID)
              for L in range(n_layers))
    oracle = max(orc, key=lambda r: r[1])

    # ---- label-permutation null (fixed C=Cstar; loky-parallel; proper p-value) ----
    perm = np.array(P()(
        delayed(_perm_worker)(SEED + 1 + k, Atr, Aval, Atrva, Ate, y[tr], y[va], yte, Cstar)
        for k in range(args.n_perm)))
    perm = perm[np.isfinite(perm)]
    perm_p = float((1 + (perm >= test_auc).sum()) / (len(perm) + 1))

    # ---- random-direction null at the deployable layer (untrained geometry) ----
    rng = np.random.default_rng(SEED)
    Xte = Ate[Lstar]; d = Xte.shape[1]
    W = rng.standard_normal((args.n_rand, d)); W /= np.linalg.norm(W, axis=1, keepdims=True)
    rand_auc = np.array([max(a, 1 - a) for a in (_auc(yte, Xte @ w) for w in W)])
    rand_p = float((1 + (rand_auc >= test_auc).sum()) / (len(rand_auc) + 1))

    # ---- verbalized reference = exp-17 PRIMED P(yes) (ask-the-model-with-the-vuln-Q);
    #      the neutral-prompt run (exp-31) has no yes/no answer, so always compare to
    #      this primed reference. HARD gate (this run's npz p_yes reproduces exp-17)
    #      applies only to the primed run (--no-verbalized-gate off). ----
    v17 = {r["eid"]: float(r["p_yes"])
           for r in json.loads((E17 / npz_slug / "example_scores_verbalized.json").read_text())}
    verb_test = np.array([v17[int(e)] for e in eid[te]])
    verb_auc = _auc(yte, verb_test)
    stored = json.loads((E17 / npz_slug / "metrics_verbalized_logits.json").read_text()).get("verbalized_auc_test")
    if not args.no_verbalized_gate:
        npz_verb_auc = _auc(yte, p_yes[te])          # this run's npz p_yes on test
        if stored is not None and abs(npz_verb_auc - stored) > 5e-3:
            raise SystemExit(f"[gate] {name}: npz p_yes AUC {npz_verb_auc:.4f} != exp-17 {stored:.4f}")

    # ---- confounds: language indicator, length, within-language probe AUC ----
    lang_te = lang[eid[te]]; clen_te = code_len[eid[te]]
    py_pos = (lang_te == "python").astype(float)
    lang_auc = _auc(yte, py_pos)
    len_auc = _auc(yte, clen_te)
    is_py = lang_te == "python"; is_c = np.isin(lang_te, ["c", "cpp", "c++"])
    within = {"python": {"n": int(is_py.sum()), "n_pos": int(yte[is_py].sum()), "auc": _auc(yte[is_py], s_test[is_py])},
              "c_cpp": {"n": int(is_c.sum()), "n_pos": int(yte[is_c].sum()), "auc": _auc(yte[is_c], s_test[is_c])}}

    # ---- bootstrap CIs + paired Δ vs verbalized ----
    def boot_ci(s):
        r = np.random.default_rng(SEED); ii = np.arange(len(yte))
        v = [roc_auc_score(yte[b], s[b]) for b in (r.choice(ii, len(ii), True) for _ in range(args.n_boot))
             if len(np.unique(yte[b])) > 1]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    r2 = np.random.default_rng(SEED); ii = np.arange(len(yte)); dd = []
    for _ in range(args.n_boot):
        b = r2.choice(ii, len(ii), True)
        if len(np.unique(yte[b])) > 1:
            dd.append(roc_auc_score(yte[b], s_test[b]) - roc_auc_score(yte[b], verb_test[b]))
    delta = {"mean": float(np.mean(dd)), "ci": [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]}

    e29 = (json.loads(E29.read_text())["models"].get(e29_slug, {}).get("full_test", {})
           if E29.exists() else {})
    res = {"n_layers": n_layers, "n_test": int(len(te)), "n_test_pos": int(yte.sum()),
           "deployable": {"layer": Lstar, "C": Cstar, "C_at_grid_edge": c_edge,
                          "val_auc": val_auc, "test_auc": test_auc, "test_ci": boot_ci(s_test)},
           "oracle": {"layer": oracle[0], "test_auc": oracle[1]},
           "perm_null": {"n": int(len(perm)), "mean": float(perm.mean()), "p95": float(np.percentile(perm, 95)),
                         "max": float(perm.max()), "p_value": perm_p},
           "random_dir_null": {"n": args.n_rand, "mean": float(rand_auc.mean()),
                               "p95": float(np.percentile(rand_auc, 95)), "max": float(rand_auc.max()), "p_value": rand_p},
           "verbalized": {"test_auc": verb_auc, "test_ci": boot_ci(verb_test), "gate_stored": stored},
           "confounds": {"language_indicator_auc": lang_auc, "code_length_auc": len_auc, "within_language": within},
           "exp29": {"max_pool": e29.get("max_pool", {}).get("auc"), "last_code_token": e29.get("last_code_token", {}).get("auc")},
           "paired_probe_minus_verbalized": delta}
    sep = "*" if (delta["ci"][0] > 0 or delta["ci"][1] < 0) else " "
    print(f"[{name}] L{Lstar}(C={Cstar}{'!edge' if c_edge else ''}) val={val_auc:.3f} test={test_auc:.3f} "
          f"| permP={perm_p:.3f}(p95={res['perm_null']['p95']:.3f}) randP={rand_p:.3f}(p95={res['random_dir_null']['p95']:.3f}) "
          f"| verb={verb_auc:.3f} Δ={delta['mean']:+.3f}{sep} | lang={lang_auc:.3f} len={len_auc:.3f} "
          f"wPy={within['python']['auc']:.3f} wC={within['c_cpp']['auc']:.3f} | oracleL{oracle[0]}={oracle[1]:.3f}",
          file=sys.stderr)
    return res


if __name__ == "__main__":
    main()
