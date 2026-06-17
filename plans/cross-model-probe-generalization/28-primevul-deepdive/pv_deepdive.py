# [ai-generated]
"""exp-28 — PrimeVul deep-dive: surface baselines, cross-dataset CIs, matched-pair.

Pure CPU, local. NO re-extraction, NO probe retraining: the probe side reads
exp-26's cached per-token logits (assets/*.npz, pulled from cluster scratch);
the surface side recomputes char-n-gram features from raw text on the IDENTICAL
token axis (assets/pv_offsets.npz = the qwen7b-L16 extractor's offsets).

Stages (each writes results/<name>.json, skip-if-exists, --force to redo):
  gate : rebuild PV+SVEN per-token substrate (eids/y_tok/is_code) and HARD-verify
         against exp-26 (n_neg_test_tokens=287864, per-CWE n_pos_tokens, exact
         reproduction of pv_within.json diagonal + cross_shared.json cells).
  A    : surface within-PV — per-CWE char-3-5-gram LR (exp-24 design-2 recipe,
         token-unigram secondary) -> 12x12 matrix, family blocks (token-weighted,
         exp-26 agg_blocks) + 500-rep CIs, diagonal + 1000-rep CIs.
  B    : bootstrap-over-examples CIs for the SVEN<->PV shared-CWE cells
         (SVEN->PV primary; PV->SVEN-C / SVEN->SVEN-C from the sven-side logits).
  C    : PV matched-pair (matched-patch) eval — per-CWE probe vs surface, eval
         pool = CWE test vuln code tokens (y_tok labels) u their OWN fixes' code
         tokens; CIs resample PAIRS. pairAcc secondary (all + subtractive slice).

Recipe parity:
  * eval cells: exp-26 pv_family.eval_cell / exp-25 deconfound.py:209-242.
  * surface training: exp-24 features.py (HashingVectorizer char 3-5, 2^18,
    +-48 win, LR liblinear C=1, NEG_CAP=60k, ratio 25), per-CWE design-2 pools
    (pos = annotated y==1 live-code train tokens of CWE-X; neg = clean-train
    live-code tokens, subsampled).
  * family blocks: exp-26 agg_blocks (token-weighted by target positive tokens).
"""
from __future__ import annotations
import argparse, json, sys, time
from collections import Counter
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
A = HERE / "assets"
RES = HERE / "results"
RES.mkdir(exist_ok=True)

from sklearn.metrics import roc_auc_score                                         # noqa: E402
from sklearn.feature_extraction.text import HashingVectorizer                     # noqa: E402
from sklearn.linear_model import LogisticRegression                                # noqa: E402
import scipy.sparse as sp                                                          # noqa: E402
from src.eval.token_data import parse_spans, char_spans_to_token_spans, token_labels_array  # noqa: E402
from src.eval.code_mask import code_only_mask                                      # noqa: E402
from src.remotes.train_eval import load_or_make_split                              # noqa: E402

PV_DS = REPO / "plans/cross-model-probe-generalization/22-primevul-paired/primevul_dataset.jsonl"
PV_MEMBERSHIP = REPO / "plans/cross-model-probe-generalization/22-primevul-paired/primevul_membership.json"
SVEN_DS = REPO / "data/dataset.jsonl"
SVEN_SPLIT = REPO / "data/sven_split_meta.json"
EXP26 = HERE.parent / "26-primevul-within-family/results"

MEM_CWES = {"CWE-787", "CWE-125", "CWE-476", "CWE-416", "CWE-119", "CWE-190", "CWE-415"}
OTHER_CWES = {"CWE-703", "CWE-200", "CWE-369", "CWE-20", "CWE-617"}
SHARED_MEM = ["CWE-125", "CWE-416", "CWE-476", "CWE-787", "CWE-190"]
CPP = {"c", "cpp", "c++"}
MIN_TRUST_POS = 10
SEED = 42
WINDOW = 48
NEG_CAP, NEG_POS_RATIO = 60_000, 25
N_BOOT_DIAG, N_BOOT_BLOCK = 1000, 500
BOOT_JOBS = 8


def log(*a):
    print("[exp28]", *a, file=sys.stderr, flush=True)


def auc_or_nan(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")


def fast_auc_batch(S, y):
    """Vectorized Mann-Whitney AUC, K probes x one token set — TIE-CORRECT
    (average ranks, = sklearn roc_auc_score). exp-26's argsort version was
    tie-blind: fine for continuous probe/char logits, WRONG for tie-heavy
    unigram scores (codex mini-review catch, 2026-06-12)."""
    from scipy.stats import rankdata
    npos = int((y == 1).sum()); nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return np.full(S.shape[0], np.nan)
    ranks = rankdata(S, method="average", axis=1)
    rsum = ranks[:, y == 1].sum(axis=1)
    return (rsum - npos * (npos + 1) / 2.0) / (npos * nneg)


def _pmap(fn, seeds, label):
    res = []
    batch = max(1, len(seeds) // 8)
    try:
        from joblib import Parallel, delayed
        for i in range(0, len(seeds), batch):
            res += Parallel(n_jobs=BOOT_JOBS, backend="threading")(
                delayed(fn)(s) for s in seeds[i:i + batch])
            log(f"[boot] {label} {min(i + batch, len(seeds))}/{len(seeds)}")
    except Exception as ex:
        log(f"[boot] {label}: joblib unavailable ({ex}); serial")
        res = [fn(s) for s in seeds]
    return res


def fam_of(c):
    return "mem" if c in MEM_CWES else ("other" if c in OTHER_CWES else "?")


# ------------------------------ substrate ------------------------------------
def build_substrate(name, rows, offsets_path, cache):
    """Per-token eid / y_tok / is_code / char offsets, aligned to the extractor's
    token axis (offsets.npz row order = jsonl row order; contiguity was asserted
    by exp-26 build_state on the same files)."""
    if cache.exists():
        z = np.load(cache)
        log(f"[{name}] substrate cache hit ({len(z['eids'])} tokens)")
        return {k: z[k] for k in ("eids", "y_tok", "is_code", "cs", "ce")}
    offs = np.load(offsets_path)
    n_rows = len(offs.files)
    assert n_rows == len(rows), f"{name}: offsets rows {n_rows} != dataset rows {len(rows)}"
    per_row = [offs[f"offsets_row_{i:04d}"] for i in range(n_rows)]
    counts = np.array([o.shape[0] for o in per_row], np.int64)
    eids = np.repeat(np.arange(n_rows, dtype=np.int64), counts)
    N = int(counts.sum())
    y_tok = np.zeros(N, np.int8)
    is_code = np.zeros(N, bool)
    cs = np.zeros(N, np.int64)
    ce = np.zeros(N, np.int64)
    cur = 0
    t0 = time.time()
    for e in range(n_rows):
        o = per_row[e]; n = o.shape[0]
        off_list = [(int(s), int(t)) for s, t in o]
        tok_spans = char_spans_to_token_spans(parse_spans(rows[e]), off_list)
        lab, _ = token_labels_array(n, tok_spans)
        y_tok[cur:cur + n] = lab
        is_code[cur:cur + n] = code_only_mask(rows[e].get("code", ""),
                                              rows[e].get("lang", "") or "", o).astype(bool)
        cs[cur:cur + n] = o[:, 0]; ce[cur:cur + n] = o[:, 1]
        cur += n
        if e % 2000 == 0:
            log(f"[{name}] substrate {e}/{n_rows} ({time.time()-t0:.0f}s)")
    assert cur == N
    np.savez_compressed(cache, eids=eids, y_tok=y_tok, is_code=is_code, cs=cs, ce=ce)
    log(f"[{name}] substrate built: {N} tokens, {int(y_tok.sum())} pos, "
        f"{int(is_code.sum())} code ({time.time()-t0:.0f}s) -> {cache.name}")
    return dict(eids=eids, y_tok=y_tok, is_code=is_code, cs=cs, ce=ce)


def pv_state():
    rows = [json.loads(l) for l in PV_DS.open()]
    st = build_substrate("pv", rows, A / "pv_offsets.npz", A / "pv_substrate.npz")
    st["rows"] = rows
    st["cwe_of"] = {e: (rows[e].get("vuln_type") if rows[e].get("label") == 1 else None)
                    for e in range(len(rows))}
    st["is_clean"] = {e: rows[e].get("label") == 0 for e in range(len(rows))}
    sp_ = {e: rows[e].get("_split") for e in range(len(rows))}
    st["fit_eids"] = {e for e in range(len(rows)) if sp_[e] == "train"}
    st["test_eids"] = {e for e in range(len(rows)) if sp_[e] == "test"}
    # pairing: row 2i = vuln (label 1), 2i+1 = its fix (label 0); verified globally.
    for i in range(0, len(rows), 2):
        assert rows[i]["label"] == 1 and rows[i + 1]["label"] == 0 \
            and rows[i]["_func_name"] == rows[i + 1]["_func_name"] \
            and rows[i]["_split"] == rows[i + 1]["_split"]   # pair-integral split
    st["fix_of"] = {i: i + 1 for i in range(0, len(rows), 2)}
    code_idx = {}
    bounds = np.searchsorted(st["eids"], np.arange(len(rows) + 1))
    for e in range(len(rows)):
        t = np.arange(bounds[e], bounds[e + 1])
        code_idx[e] = t[st["is_code"][t]]
    st["ctok"] = code_idx
    return st


def sven_state():
    rows = [json.loads(l) for l in SVEN_DS.open()]
    st = build_substrate("sven", rows, A / "sven_offsets.npz", A / "sven_substrate.npz")
    st["rows"] = rows
    _, train_eids, test_eids = load_or_make_split(SVEN_DS, SVEN_SPLIT)
    st["test_eids"] = set(int(x) for x in test_eids)
    st["cwe_of"] = {e: rows[e].get("cwe") for e in range(len(rows))}
    st["is_clean"] = {e: (rows[e].get("label") == 0 and not rows[e].get("cwe"))
                      for e in range(len(rows))}
    st["lang_of"] = {e: (rows[e].get("lang") or "").lower() for e in range(len(rows))}
    code_idx = {}
    bounds = np.searchsorted(st["eids"], np.arange(len(rows) + 1))
    for e in range(len(rows)):
        t = np.arange(bounds[e], bounds[e + 1])
        code_idx[e] = t[st["is_code"][t]]
    st["ctok"] = code_idx
    return st


def ctoks(st, eid_list):
    if not eid_list:
        return np.array([], np.int64)
    return np.concatenate([st["ctok"][e] for e in eid_list])


def load_logits(fname, expect_n):
    z = np.load(A / fname, allow_pickle=True)
    assert int(z["n"]) == expect_n, f"{fname}: n {int(z['n'])} != substrate {expect_n}"
    cwes = [str(c) for c in z["cwes"]]
    Lg = z["Lg"]
    return {c: Lg[:, i] for i, c in enumerate(cwes)}


def eval_cell(st, lg, pos_eids, neg_tok):
    pt = ctoks(st, pos_eids)
    if pt.size == 0 or neg_tok.size == 0:
        return float("nan"), 0
    ev = np.concatenate([pt, neg_tok])
    lab = st["y_tok"][ev]
    if len(np.unique(lab)) < 2:
        return float("nan"), int((st["y_tok"][pt] == 1).sum())
    return auc_or_nan(lab, lg[ev]), int((st["y_tok"][pt] == 1).sum())


def feasible_cwes(pv):
    test_pos = Counter(pv["cwe_of"][e] for e in pv["test_eids"] if pv["cwe_of"][e])
    return sorted([c for c, n in test_pos.items() if n >= MIN_TRUST_POS],
                  key=lambda c: (fam_of(c) != "mem", c)), test_pos


# ------------------------------ stage: gate ----------------------------------
def stage_gate(pv, sven):
    out = {"gates": {}}
    ok = True

    def gate(name, got, want, exact=True, tol=2e-3):
        nonlocal ok
        good = (got == want) if exact else (abs(got - want) <= tol)
        out["gates"][name] = {"got": got, "want": want, "pass": bool(good)}
        if not good:
            ok = False
            log(f"GATE FAIL {name}: got {got} want {want}")
        else:
            log(f"gate ok  {name}: {got}")

    w26 = json.load(open(EXP26 / "pv_within.json"))
    cross26 = json.load(open(EXP26 / "cross_shared.json"))

    clean_test = [e for e in pv["test_eids"] if pv["is_clean"][e]]
    neg_tok = ctoks(pv, clean_test)
    gate("pv_n_clean_test", len(clean_test), w26["n_clean_test"])
    gate("pv_n_neg_test_tokens", int(neg_tok.size), w26["n_neg_test_tokens"])

    feas, test_pos = feasible_cwes(pv)
    gate("pv_feasible_cwes", feas, w26["feasible_cwes"])
    for c in feas:
        pos_eids = [e for e in pv["test_eids"] if pv["cwe_of"][e] == c]
        npos = int((pv["y_tok"][ctoks(pv, pos_eids)] == 1).sum())
        gate(f"pv_npos_tok_{c}", npos, w26["n_pos_tokens"][c])

    # exact reproduction of the exp-26 within-PV diagonal from cached logits
    Lg_pv = load_logits("logits_pv.npz", len(pv["eids"]))
    for c in feas:
        pos_eids = [e for e in pv["test_eids"] if pv["cwe_of"][e] == c]
        a, _ = eval_cell(pv, Lg_pv[c], pos_eids, neg_tok)
        gate(f"pv_diag_{c}", round(a, 6), round(w26["matrix"][c][c], 6))

    # cross_shared reproduction (all 4 columns, 5 shared CWEs)
    Lg_sv_on_pv = load_logits("logits_pv_svenprobes.npz", len(pv["eids"]))
    Lg_pv_on_sv = load_logits("logits_sven_pvprobes.npz", len(sven["eids"]))
    Lg_sv_on_sv = load_logits("logits_sven_svenprobes.npz", len(sven["eids"]))
    sven_clean_test_c = [e for e in sven["test_eids"]
                         if sven["is_clean"][e] and sven["lang_of"][e] in CPP]
    sven_neg_c = ctoks(sven, sven_clean_test_c)
    gate("sven_n_clean_test_c", len(sven_clean_test_c), cross26["n_sven_c_clean_test"])
    for c in SHARED_MEM:
        pvp = [e for e in pv["test_eids"] if pv["cwe_of"][e] == c]
        svp = [e for e in sven["test_eids"]
               if sven["cwe_of"][e] == c and sven["lang_of"][e] in CPP]
        t = cross26["table"][c]
        a, _ = eval_cell(pv, Lg_sv_on_pv[c], pvp, neg_tok)
        gate(f"x_SVEN->PV_{c}", round(a, 6), round(t["SVEN->PV"], 6))
        a, _ = eval_cell(pv, Lg_pv[c], pvp, neg_tok)
        gate(f"x_PV->PV_{c}", round(a, 6), round(t["PV->PV"], 6))
        a, _ = eval_cell(sven, Lg_pv_on_sv[c], svp, sven_neg_c)
        gate(f"x_PV->SVEN_C_{c}", round(a, 6), round(t["PV->SVEN_C"], 6))
        a, _ = eval_cell(sven, Lg_sv_on_sv[c], svp, sven_neg_c)
        gate(f"x_SVEN->SVEN_C_{c}", round(a, 6), round(t["SVEN->SVEN_C"], 6))

    out["all_pass"] = ok
    (RES / "gate.json").write_text(json.dumps(out, indent=2))
    log(f"GATE {'PASS' if ok else 'FAIL'} -> results/gate.json")
    if not ok:
        raise SystemExit("substrate does not reproduce exp-26 — STOP")


# --------------------------- surface machinery -------------------------------
HV = HashingVectorizer(analyzer="char", ngram_range=(3, 5), n_features=2 ** 18,
                       alternate_sign=False, norm=None, dtype=np.float32)


def windows_of(pv, idx):
    rows = pv["rows"]; eids = pv["eids"]; cs = pv["cs"]; ce = pv["ce"]
    out = []
    for i in idx:
        code = rows[int(eids[i])]["code"]
        a, b = int(cs[i]), int(ce[i])
        out.append(code[max(0, a - WINDOW): b + WINDOW])
    return out


def toks_str_of(pv, idx):
    rows = pv["rows"]; eids = pv["eids"]; cs = pv["cs"]; ce = pv["ce"]
    return [rows[int(eids[i])]["code"][int(cs[i]):int(ce[i])].strip() for i in idx]


def fit_lr(feat_tr, ytr):
    clf = LogisticRegression(solver="liblinear", C=1.0, max_iter=1000)
    clf.fit(feat_tr, ytr)
    return clf


def subsample_negs(pos_n, neg_idx, rng):
    cap = min(NEG_CAP, max(NEG_POS_RATIO * max(pos_n, 1), 2000))
    if len(neg_idx) > cap:
        neg_idx = rng.choice(neg_idx, cap, replace=False)
    return neg_idx


def build_surface_scores(pv, feas):
    """Train per-CWE char-ngram (+ unigram) LRs (exp-24 design-2 recipe); return
    scores over the TEST-token union (code tokens of all test examples)."""
    cache = A / "surface_scores.npz"
    test_union = ctoks(pv, sorted(pv["test_eids"]))
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if list(z["cwes"]) == feas and int(z["n"]) == len(test_union):
            log(f"[surface] cache hit ({len(feas)} cwes, {len(test_union)} union toks)")
            return (test_union, {c: z["H"][i] for i, c in enumerate(feas)},
                    {c: z["U"][i] for i, c in enumerate(feas)})
    t0 = time.time()
    log(f"[surface] featurizing eval union: {len(test_union)} tokens")
    H_eval = HV.transform(windows_of(pv, test_union))
    log(f"[surface] H_eval {H_eval.shape} nnz={H_eval.nnz} ({time.time()-t0:.0f}s)")

    clean_fit = [e for e in pv["fit_eids"] if pv["is_clean"][e]]
    neg_pool = ctoks(pv, clean_fit)                      # clean-train code tokens
    log(f"[surface] neg pool {len(neg_pool)} tokens")

    # unigram vocab from ALL train live-code tokens (exp-24: vocab = train-observed)
    fit_union = ctoks(pv, sorted(pv["fit_eids"]))
    vocab = {}
    for s in toks_str_of(pv, fit_union):
        if s and s not in vocab:
            vocab[s] = len(vocab)
    log(f"[surface] unigram vocab {len(vocab)} ({time.time()-t0:.0f}s)")

    def U_feat(idx):
        ss = toks_str_of(pv, idx)
        rows_, cols_ = [], []
        for r, s in enumerate(ss):
            j = vocab.get(s)
            if j is not None:
                rows_.append(r); cols_.append(j)
        return sp.csr_matrix((np.ones(len(rows_), np.float32), (rows_, cols_)),
                             shape=(len(idx), len(vocab)))

    U_eval = U_feat(test_union)
    Hs, Us = {}, {}
    for k, c in enumerate(feas):
        rng = np.random.default_rng(SEED + 17 * k)
        pos_e = [e for e in pv["fit_eids"] if pv["cwe_of"][e] == c]
        pt = ctoks(pv, pos_e)
        pos_idx = pt[pv["y_tok"][pt] == 1]               # annotated y==1 live-code
        neg_idx = subsample_negs(len(pos_idx), neg_pool, rng)
        tr = np.concatenate([pos_idx, neg_idx])
        ytr = np.concatenate([np.ones(len(pos_idx), np.int8),
                              np.zeros(len(neg_idx), np.int8)])
        Hs[c] = fit_lr(HV.transform(windows_of(pv, tr)), ytr) \
            .decision_function(H_eval).astype(np.float32)
        Us[c] = fit_lr(U_feat(tr), ytr).decision_function(U_eval).astype(np.float32)
        log(f"[surface] {c} ({fam_of(c)}): n_pos_tok={len(pos_idx)} n_neg={len(neg_idx)} "
            f"({time.time()-t0:.0f}s)")
    np.savez_compressed(cache, cwes=np.array(feas),
                        H=np.stack([Hs[c] for c in feas]),
                        U=np.stack([Us[c] for c in feas]), n=np.int64(len(test_union)))
    return test_union, Hs, Us


def union_map(test_union):
    m = {}
    for u, g in enumerate(test_union):
        m[int(g)] = u
    return m


def agg_blocks(Mb, w, trained, grp):
    o = {}
    for gtr in ("mem", "other"):
        for gte in ("mem", "other"):
            for tag, skip in (("", False), ("_offdiag", True)):
                num = den = 0.0
                for ix in grp[gtr]:
                    for iy in grp[gte]:
                        if skip and ix == iy:
                            continue
                        a = Mb[ix, iy]; ww = w[iy]
                        if a == a and ww:
                            num += a * ww; den += ww
                o[f"{gtr}->{gte}{tag}"] = (num / den) if den else float("nan")
    return o


# ------------------------------ stage A --------------------------------------
def stage_a(pv, force=False):
    of = RES / "pv_surface.json"
    if of.exists() and not force:
        log("[A] pv_surface.json exists, skip"); return
    w26 = json.load(open(EXP26 / "pv_within.json"))
    feas, test_pos = feasible_cwes(pv)
    test_union, Hs, Us = build_surface_scores(pv, feas)
    um = union_map(test_union)
    y_u = pv["y_tok"][test_union]

    clean_test = [e for e in pv["test_eids"] if pv["is_clean"][e]]
    pos_test_of = {c: [e for e in pv["test_eids"] if pv["cwe_of"][e] == c] for c in feas}
    ctok_u = {e: np.array([um[int(t)] for t in pv["ctok"][e]], np.int64)
              for e in sorted(pv["test_eids"])}

    def utoks(eids):
        return (np.concatenate([ctok_u[e] for e in eids]) if eids
                else np.array([], np.int64))

    neg_u = utoks(clean_test)
    K = len(feas)
    Smat = {"char_ngram_lr": np.stack([Hs[c] for c in feas]),
            "token_unigram_lr": np.stack([Us[c] for c in feas])}
    grp = {"mem": [i for i, c in enumerate(feas) if fam_of(c) == "mem"],
           "other": [i for i, c in enumerate(feas) if fam_of(c) == "other"]}
    npos_vec = np.array([w26["n_pos_tokens"][c] for c in feas], float)

    out = {"analysis": "pv_surface_within", "metric": "tokens_code_auc",
           "cwes": feas, "recipe": "exp-24 design-2 per-CWE (char-ngram primary)",
           "n_neg_test_tokens": int(neg_u.size), "baselines": {}}

    for bname, S in Smat.items():
        M = np.full((K, K), np.nan)
        for ix in range(K):
            for iy, cy in enumerate(feas):
                ev = np.concatenate([utoks(pos_test_of[cy]), neg_u])
                lab = y_u[ev]
                M[ix, iy] = auc_or_nan(lab, S[ix][ev])
        blk = agg_blocks(M, npos_vec, feas, grp)

        def block_rep(rep, S=S):
            r = np.random.default_rng(SEED + 101 + rep)
            cs_ = [clean_test[j] for j in r.integers(0, len(clean_test), len(clean_test))]
            negb = utoks(cs_)
            Mb = np.full((K, K), np.nan); w = np.zeros(K)
            for iy, cy in enumerate(feas):
                pe = pos_test_of[cy]
                pt = utoks([pe[j] for j in r.integers(0, len(pe), len(pe))])
                ev = np.concatenate([pt, negb]); yy = y_u[ev]
                w[iy] = int((y_u[pt] == 1).sum())
                Mb[:, iy] = fast_auc_batch(S[:, ev], yy)
            return agg_blocks(Mb, w, feas, grp)

        def diag_rep(rep, S=S):
            r = np.random.default_rng(SEED + 5001 + rep)
            o = np.full(K, np.nan)
            for ix, c in enumerate(feas):
                pe = pos_test_of[c]
                pt = utoks([pe[j] for j in r.integers(0, len(pe), len(pe))])
                cs_ = [clean_test[j] for j in r.integers(0, len(clean_test), len(clean_test))]
                ev = np.concatenate([pt, utoks(cs_)]); yy = y_u[ev]
                o[ix] = fast_auc_batch(S[ix:ix + 1, ev], yy)[0]
            return o

        bres = _pmap(block_rep, list(range(N_BOOT_BLOCK)), f"{bname}-block")
        blk_ci = {k: [float(np.percentile([v[k] for v in bres if v[k] == v[k]], 2.5)),
                      float(np.percentile([v[k] for v in bres if v[k] == v[k]], 97.5))]
                  for k in blk}
        dres = np.array(_pmap(diag_rep, list(range(N_BOOT_DIAG)), f"{bname}-diag"))
        diag = {}
        for ix, c in enumerate(feas):
            col = dres[:, ix]; col = col[~np.isnan(col)]
            diag[c] = {"auc": M[ix, ix],
                       "ci": [float(np.percentile(col, 2.5)), float(np.percentile(col, 97.5))],
                       "family": fam_of(c), "n_test_pos": test_pos[c],
                       "probe_auc_exp26": w26["diagonal"][c]["auc"],
                       "probe_ci_exp26": w26["diagonal"][c]["ci"]}
        out["baselines"][bname] = {
            "matrix": {cx: {cy: M[i][j] if (M[i][j] == M[i][j]) else None
                            for j, cy in enumerate(feas)} for i, cx in enumerate(feas)},
            "family_blocks": blk, "family_blocks_ci": blk_ci, "diagonal": diag,
        }
        log(f"[A] {bname}: mem->mem_offdiag={blk['mem->mem_offdiag']:.3f} "
            f"mem->other={blk['mem->other']:.3f}")

    out["probe_family_blocks_exp26"] = w26["family_blocks"]
    out["probe_family_blocks_ci_exp26"] = w26["family_blocks_ci"]
    of.write_text(json.dumps(out, indent=2, default=lambda x: None if x != x else x))
    log(f"[A] wrote {of}")


# ------------------------------ stage B --------------------------------------
def stage_b(pv, sven, force=False):
    of = RES / "cross_cis.json"
    if of.exists() and not force:
        log("[B] cross_cis.json exists, skip"); return
    cross26 = json.load(open(EXP26 / "cross_shared.json"))
    Lg_pv = load_logits("logits_pv.npz", len(pv["eids"]))
    Lg_sv_on_pv = load_logits("logits_pv_svenprobes.npz", len(pv["eids"]))
    Lg_pv_on_sv = load_logits("logits_sven_pvprobes.npz", len(sven["eids"]))
    Lg_sv_on_sv = load_logits("logits_sven_svenprobes.npz", len(sven["eids"]))

    pv_clean = [e for e in pv["test_eids"] if pv["is_clean"][e]]
    sv_clean_c = [e for e in sven["test_eids"]
                  if sven["is_clean"][e] and sven["lang_of"][e] in CPP]

    def ci_cell(st, lg, pos_eids, clean_pool, n_boot=N_BOOT_DIAG):
        def rep(r_):
            r = np.random.default_rng(SEED + 7000 + r_)
            ps = [pos_eids[j] for j in r.integers(0, len(pos_eids), len(pos_eids))]
            cs_ = [clean_pool[j] for j in r.integers(0, len(clean_pool), len(clean_pool))]
            ev = np.concatenate([ctoks(st, ps), ctoks(st, cs_)])
            lab = st["y_tok"][ev]
            if len(np.unique(lab)) < 2:
                return np.nan
            return fast_auc_batch(lg[ev][None, :], lab)[0]
        b = np.array(_pmap(rep, list(range(n_boot)), "cross-ci"))
        b = b[~np.isnan(b)]
        return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), int(b.size)]

    out = {"analysis": "cross_dataset_cis", "metric": "tokens_code_auc", "table": {}}
    for c in SHARED_MEM:
        pvp = [e for e in pv["test_eids"] if pv["cwe_of"][e] == c]
        svp = [e for e in sven["test_eids"]
               if sven["cwe_of"][e] == c and sven["lang_of"][e] in CPP]
        row = {"n_pv_test_pos": len(pvp), "n_sven_c_test_pos": len(svp),
               "trust_pv": len(pvp) >= MIN_TRUST_POS,
               "trust_sven_c": len(svp) >= MIN_TRUST_POS}
        for col, st, lg, pos, cl in (
                ("SVEN->PV", pv, Lg_sv_on_pv[c], pvp, pv_clean),
                ("PV->PV", pv, Lg_pv[c], pvp, pv_clean),
                ("PV->SVEN_C", sven, Lg_pv_on_sv[c], svp, sv_clean_c),
                ("SVEN->SVEN_C", sven, Lg_sv_on_sv[c], svp, sv_clean_c)):
            a, _ = eval_cell(st, lg, pos, ctoks(st, cl))
            assert abs(a - cross26["table"][c][col]) < 1e-6, (c, col, a)
            row[col] = {"auc": a, "ci": ci_cell(st, lg, pos, cl)}
            log(f"[B] {c} {col}: {a:.3f} CI {row[col]['ci'][:2]}")
        out["table"][c] = row
    out["note"] = ("bootstrap over examples (pos + clean resampled, 1000 reps); "
                   "point estimates reproduce exp-26 cross_shared.json exactly. "
                   "SVEN-C cells with <10 test pos remain untrusted.")
    of.write_text(json.dumps(out, indent=2))
    log(f"[B] wrote {of}")


# ------------------------------ stage C --------------------------------------
def stage_c(pv, force=False):
    of = RES / "pv_matchedpair.json"
    if of.exists() and not force:
        log("[C] pv_matchedpair.json exists, skip"); return
    feas, test_pos = feasible_cwes(pv)
    Lg_pv = load_logits("logits_pv.npz", len(pv["eids"]))
    Lg_sv_on_pv = load_logits("logits_pv_svenprobes.npz", len(pv["eids"]))
    test_union, Hs, Us = build_surface_scores(pv, feas)
    # expand surface scores to global token axis (NaN off-test); pools are test-only
    Sg = {}
    for c in feas:
        g = np.full(len(pv["eids"]), np.nan, np.float32)
        g[test_union] = Hs[c]
        Sg[c] = g

    membership = json.load(open(PV_MEMBERSHIP))
    sub_vulns = {p[0] for p in membership["subtractive_pairs"]}   # [vuln_eid, fix_eid]

    pos_test_of = {c: [e for e in pv["test_eids"] if pv["cwe_of"][e] == c] for c in feas}

    def mp_eval(score, pos_eids):
        fixes = [pv["fix_of"][v] for v in pos_eids]
        pt, nt = ctoks(pv, pos_eids), ctoks(pv, fixes)
        if pt.size == 0 or nt.size == 0:
            return float("nan")
        ev = np.concatenate([pt, nt])
        return auc_or_nan(pv["y_tok"][ev], score[ev])

    def mp_ci(score, pos_eids, n_boot=N_BOOT_DIAG):
        def rep(r_):
            r = np.random.default_rng(SEED + 9000 + r_)
            ps = [pos_eids[j] for j in r.integers(0, len(pos_eids), len(pos_eids))]
            return mp_eval(score, ps)           # fixes follow their vulns (pair unit)
        b = np.array(_pmap(rep, list(range(n_boot)), "mp-ci"))
        b = b[~np.isnan(b)]
        return ([float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), int(b.size)]
                if b.size else [float("nan")] * 3)

    def pair_acc(score, pos_eids):
        wins = ties = tot = 0
        for v in pos_eids:
            tv, tf = pv["ctok"][v], pv["ctok"][pv["fix_of"][v]]
            if tv.size == 0 or tf.size == 0:
                continue
            mv, mf = float(np.nanmax(score[tv])), float(np.nanmax(score[tf]))
            tot += 1
            if mv > mf:
                wins += 1
            elif mv == mf:
                ties += 1
        return (wins + 0.5 * ties) / tot if tot else float("nan"), tot

    out = {"analysis": "pv_matchedpair", "metric": "tokens_code_auc",
           "regime": "matched-patch: negatives = the CWE's OWN test-pair fixes "
                     "(exp-25 construction); positives identical to all-clean cells",
           "pair_acc_note": "SECONDARY metric (exp-19): P(max code-token score "
                            "vuln > its fix) + 0.5 ties, over test pairs",
           "table": {}}
    for c in feas:
        pe = pos_test_of[c]
        pe_sub = [v for v in pe if v in sub_vulns]
        row = {"family": fam_of(c), "n_test_pairs": len(pe),
               "trust": len(pe) >= MIN_TRUST_POS}
        row["probe_pv"] = {"auc": mp_eval(Lg_pv[c], pe), "ci": mp_ci(Lg_pv[c], pe)}
        pa, n = pair_acc(Lg_pv[c], pe)
        row["probe_pv"]["pair_acc"] = pa
        if c in SHARED_MEM:
            row["probe_sven"] = {"auc": mp_eval(Lg_sv_on_pv[c], pe),
                                 "ci": mp_ci(Lg_sv_on_pv[c], pe),
                                 "pair_acc": pair_acc(Lg_sv_on_pv[c], pe)[0]}
        row["surface_char_ngram"] = {"auc": mp_eval(Sg[c], pe), "ci": mp_ci(Sg[c], pe),
                                     "pair_acc": pair_acc(Sg[c], pe)[0]}
        sub_note = {}
        if pe_sub:
            sub_note = {"n_sub_pairs": len(pe_sub),
                        "probe_pv_auc_sub": mp_eval(Lg_pv[c], pe_sub),
                        "probe_pv_pair_acc_sub": pair_acc(Lg_pv[c], pe_sub)[0],
                        "surface_auc_sub": mp_eval(Sg[c], pe_sub),
                        "surface_pair_acc_sub": pair_acc(Sg[c], pe_sub)[0]}
        row["subtractive_slice"] = sub_note
        out["table"][c] = row
        log(f"[C] {c}: probe={row['probe_pv']['auc']:.3f} CI {row['probe_pv']['ci'][:2]} "
            f"surf={row['surface_char_ngram']['auc']:.3f} pairAcc={pa:.3f} (n={len(pe)})")
    of.write_text(json.dumps(out, indent=2, default=lambda x: None if x != x else x))
    log(f"[C] wrote {of}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["gate", "A", "B", "C", "all"], default="all")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    pv = pv_state()
    need_sven = args.stage in ("gate", "B", "all")
    sven = sven_state() if need_sven else None
    if args.stage in ("gate", "all"):
        stage_gate(pv, sven)
    if args.stage in ("A", "all"):
        stage_a(pv, force=args.force)
    if args.stage in ("B", "all"):
        stage_b(pv, sven, force=args.force)
    if args.stage in ("C", "all"):
        stage_c(pv, force=args.force)
    log("done")


if __name__ == "__main__":
    main()
