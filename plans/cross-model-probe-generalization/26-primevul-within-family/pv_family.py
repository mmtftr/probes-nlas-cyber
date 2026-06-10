# [ai-generated]
"""exp-26 — PrimeVul within-C/C++ family structure + SVEN<->PV shared-CWE transfer.

Two analyses, ONE model (Qwen2.5-Coder-7B-Instruct), ONE layer (L16), exp-10
"all-clean" recipe in token granularity, honest `tokens_code_auc` headline:

  A) within-PV transfer matrix (primary). Per-CWE probes over the feasible CWEs
     (>=10 PV test positives), full train-CWE x test-CWE matrix of tokens_code_auc,
     family blocks (memory vs other) on/off-diagonal with bootstrap-over-examples
     CIs, diagonal CIs, n flags. Language held FIXED (all PV is C/C++).

  B) SVEN<->PV shared-CWE memory transfer. For each memory CWE present in BOTH
     datasets (125/416/476/787/190): {SVEN->PV, PV->PV, PV->SVEN-C, SVEN->SVEN-C}
     tokens_code_auc. Same model+layer both sides => probe dims match.

RECIPE PARITY with exp-10 / exp-21 transfer_allclean.py:
  * positives = annotated `token_labels` spans (parse_spans, label==1) — the same
    label the extractor populated; works on PV (evidence/vulnerable_line keys);
  * train CWE-X probe on {CWE-X vuln examples} u {ALL clean examples}, on ALL
    tokens (span-max pools per example; NO is_code gate at train);
  * eval cell = tokens_code_auc of probe-X on {CWE-Y test positives, code-only,
    annotated y_tok} u {shared clean-test negatives, code-only}. Live-code only.

PV SPECIFICS (vs SVEN):
  * cwe = `vuln_type` (clean string), NOT the stringified `cwe` list.
  * clean pool = label==0 rows. In PV-Paired every label==0 row is a paired
    safe-half (carries a vuln_type) — there is NO cwe==null unpaired-clean pool.
    The all-clean negative pool is therefore the safe-halves; language-clean BY
    CONSTRUCTION (all PV is C/C++) — no Python contrast can leak in.
  * split from embedded `_split` (train/valid/test). Verified locally: 0 groups
    straddle splits => no train/test leak via paired safe-halves.

MEMORY: PV acts are ~118 GB => loaded on CPU (device="cpu"). Eval logits are
computed ONLY on the needed token subsets (never a global 8M-token X@W), so RAM
stays bounded by X itself. Training indexes X[subset] (small).
"""
from __future__ import annotations
import argparse, json, os, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

BOOT_JOBS = int(os.environ.get("BOOT_JOBS", "48"))   # threads for bootstrap reps
LOGIT_CHUNK = int(os.environ.get("LOGIT_CHUNK", "200000"))  # rows/chunk in Stage A

REPO = Path(os.environ.get("REPO", Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(REPO))
from sklearn.metrics import roc_auc_score                                        # noqa: E402
from src.eval.token_data import parse_spans, char_spans_to_token_spans, token_labels_array  # noqa: E402
from src.eval.code_mask import code_only_mask                                    # noqa: E402
from src.remotes.the cluster.train_eval import load_or_make_split, pair_group_key   # noqa: E402
from src.training.train_probe_spanmax import train_one_layer                     # noqa: E402

# family taxonomy (CWE-Y -> family); feasible = >=10 PV test positives
MEM_CWES = {"CWE-787", "CWE-125", "CWE-476", "CWE-416", "CWE-119", "CWE-190", "CWE-415"}
OTHER_CWES = {"CWE-703", "CWE-200", "CWE-369", "CWE-20", "CWE-617"}
SHARED_MEM = ["CWE-125", "CWE-416", "CWE-476", "CWE-787", "CWE-190"]  # SVEN ∩ PV memory
CPP = {"c", "cpp", "c++"}
VAL_FRAC, VAL_SEED = 0.15, 42
MIN_TRUST_POS = 10
N_BOOT = 1000


def auc_or_nan(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")


def fast_auc_batch(S, y):
    """Vectorized Mann-Whitney AUC for K probes over ONE token set in a single
    batched argsort. S=(K, N) logits, y=(N,) binary. Returns (K,) AUCs.

    Equals sklearn.roc_auc_score to <1e-6 on continuous logits (ties — exact
    float equality — are astronomically rare; verified in smoke test). Used
    ONLY inside the CI bootstrap (where sub-1e-6 wobble is irrelevant); every
    REPORTED point estimate (matrix, diagonal, blocks) uses sklearn roc_auc_score
    via eval_cell, so headline numbers are bit-for-bit the prior harness."""
    npos = int((y == 1).sum()); nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return np.full(S.shape[0], np.nan)
    order = np.argsort(S, axis=1, kind="stable")          # ascending, per row
    ys = y[order].astype(np.float64)                       # (K, N)
    ranks = np.arange(1, len(y) + 1)
    rsum = (ys * ranks).sum(axis=1)                        # rank-sum of positives
    return (rsum - npos * (npos + 1) / 2.0) / (npos * nneg)


def _pmap(fn, seeds, label):
    """Map fn over seeds with threaded parallelism (numpy argsort releases the
    GIL) + a heartbeat per batch. Falls back to serial if joblib is absent."""
    res = []
    batch = max(1, len(seeds) // 8)
    try:
        from joblib import Parallel, delayed
        for i in range(0, len(seeds), batch):
            res += Parallel(n_jobs=BOOT_JOBS, backend="threading")(
                delayed(fn)(s) for s in seeds[i:i + batch])
            print(f"[boot] {label} {min(i + batch, len(seeds))}/{len(seeds)}", file=sys.stderr)
    except Exception as ex:                                 # serial fallback
        print(f"[boot] {label}: joblib unavailable ({ex}); serial", file=sys.stderr)
        res = []
        for j, s in enumerate(seeds):
            res.append(fn(s))
            if j % 100 == 0:
                print(f"[boot] {label} {j}/{len(seeds)} (serial)", file=sys.stderr)
    return res


def fam_of(c):
    return "mem" if c in MEM_CWES else ("other" if c in OTHER_CWES else "?")


def build_state(name, acts_dir, rows, L, kind, split_file=None):
    """Load acts + per-token annotated label/is_code/lang, aligned to X.

    kind="pv":  cwe=vuln_type; clean=label==0; split from row['_split'].
    kind="sven":cwe=row['cwe']; clean=label==0 & not cwe; split via split_file.
    Returns dict with X, eids, y_tok, is_code, is_cpp, tok_by_eid, cwe_of,
    is_clean, fit_eids, test_eids (sets), lang_of.
    """
    acts = Path(acts_dir)
    offs = np.load(acts / "offsets.npz")
    n_rows = len(offs.files)
    assert n_rows == len(rows), f"{name}: acts rows {n_rows} != dataset rows {len(rows)}"
    offsets_per_row = [offs[f"offsets_row_{i:04d}"] for i in range(n_rows)]

    cand = [p for p in acts.glob("token_activations_layer*.npz")
            if int(p.stem.replace("token_activations_layer", "")) == L]
    if not cand:
        raise SystemExit(f"{name}: no token_activations for layer {L} under {acts}")
    npz = np.load(cand[0])
    X = npz["X"]
    if X.dtype != np.float32:                      # avoid needless 2x copy for big PV X
        X = X.astype(np.float32)
    eids = npz["example_ids"].astype(np.int64)
    print(f"[{name}] X {X.shape} {X.dtype} ({X.nbytes/1e9:.1f} GB), {n_rows} rows", file=sys.stderr)

    y_tok = np.zeros(len(eids), np.int8)
    is_code = np.zeros(len(eids), bool)
    cur = 0
    for e in range(n_rows):
        n = int((eids == e).sum())
        o = offsets_per_row[e]
        assert o.shape[0] == n, f"{name}: offset/token mismatch eid {e}: {o.shape[0]} vs {n}"
        assert (eids[cur:cur + n] == e).all(), f"{name}: non-contiguous eid block at {e}"
        off_list = [(int(s), int(t)) for s, t in o]
        tok_spans = char_spans_to_token_spans(parse_spans(rows[e]), off_list)
        lab, _ = token_labels_array(n, tok_spans)
        y_tok[cur:cur + n] = lab
        is_code[cur:cur + n] = code_only_mask(rows[e].get("code", ""),
                                              rows[e].get("lang", "") or "", o).astype(bool)
        cur += n
    assert cur == len(eids), f"{name}: consumed {cur} of {len(eids)}"

    if kind == "pv":
        cwe_of = {e: (rows[e].get("vuln_type") if rows[e].get("label") == 1 else None)
                  for e in range(n_rows)}
        is_clean = {e: (rows[e].get("label") == 0) for e in range(n_rows)}
        sp = {e: rows[e].get("_split") for e in range(n_rows)}
        fit_eids = {e for e in range(n_rows) if sp[e] == "train"}
        test_eids = {e for e in range(n_rows) if sp[e] == "test"}
    elif kind == "sven":
        assert Path(split_file[1]).exists(), \
            f"split file missing: {split_file[1]} (would mint a NEW split)"
        rows2, train_eids, te = load_or_make_split(Path(split_file[0]), Path(split_file[1]))
        train_set = set(int(x) for x in train_eids)
        test_eids = set(int(x) for x in te)
        cwe_of = {e: rows[e].get("cwe") for e in range(n_rows)}
        is_clean = {e: (rows[e].get("label") == 0 and not rows[e].get("cwe")) for e in range(n_rows)}
        # exp-10 parity: 15% group-aware VAL carve of TRAIN excluded from fit
        tr_eid_grp = {e: pair_group_key(rows[e]) for e in train_set}
        tr_groups = sorted(set(tr_eid_grp.values()))
        vr = np.random.default_rng(VAL_SEED); vr.shuffle(tr_groups)
        n_val = max(1, int(round(VAL_FRAC * len(tr_groups))))
        val_groups = set(tr_groups[:n_val])
        val_eids = {e for e, g in tr_eid_grp.items() if g in val_groups}
        fit_eids = train_set - val_eids
    else:
        raise ValueError(kind)

    lang_of = {e: (rows[e].get("lang") or "").lower() for e in range(n_rows)}
    is_cpp = np.fromiter((lang_of[int(e)] in CPP for e in eids), bool, len(eids))

    # recipe invariant: clean (negative-pool) examples must carry NO positive
    # tokens, else the all-clean negative pool is silently mislabeled. Verified
    # true locally for PV (0/4704) and SVEN (0/715); assert so a future dataset
    # that breaks it fails loud rather than corrupting AUCs.
    cur2 = 0
    for e in range(n_rows):
        n = int((eids == e).sum())
        if is_clean[e]:
            assert not (y_tok[cur2:cur2 + n] == 1).any(), \
                f"{name}: clean eid {e} has positive token_labels (mislabels negatives)"
        cur2 += n

    tok_by_eid = defaultdict(list)
    for i, e in enumerate(eids):
        tok_by_eid[int(e)].append(i)
    tok_by_eid = {e: np.asarray(ix, np.int64) for e, ix in tok_by_eid.items()}

    return dict(name=name, rows=rows, X=X, eids=eids, y_tok=y_tok, is_code=is_code,
                is_cpp=is_cpp, tok_by_eid=tok_by_eid, cwe_of=cwe_of, is_clean=is_clean,
                fit_eids=fit_eids, test_eids=test_eids, lang_of=lang_of)


def toks(st, eid_list, code_only=False):
    if not eid_list:
        return np.array([], np.int64)
    t = np.concatenate([st["tok_by_eid"][e] for e in eid_list])
    return t[st["is_code"][t]] if code_only else t


def train_cwe_probe(st, c, clean_fit, epochs, device):
    """Train CWE-X all-clean probe: {CWE-X vuln fit} u {clean fit}, ALL tokens."""
    pos_fit = [e for e in st["fit_eids"] if st["cwe_of"][e] == c]
    if not pos_fit:
        return None, 0
    fit_tok = toks(st, pos_fit + clean_fit)             # no is_code gate (exp-10)
    r = train_one_layer(st["X"][fit_tok], st["y_tok"][fit_tok], st["eids"][fit_tok],
                        epochs=epochs, device=device, verbose=False)
    return (np.asarray(r["w"], np.float32), float(r["b"])), len(pos_fit)


def precompute_logits(st, W, b, cache=None, chunk=LOGIT_CHUNK):
    """Stage A — global per-probe token logits via a CHUNKED single pass over X.

    Instead of one opaque `X @ Wm` over a 118 GB array (silent, unbounded temps,
    nothing persisted on a wall), we stream X in `chunk`-row blocks: bounded
    memory, a heartbeat per few chunks, and the (n_tok x k) f32 logit matrix is
    cached to `cache` (skip-if-exists). Bootstrap then only INDEXES these logits,
    never X. Returns {cwe: logit_array}."""
    cs = sorted(W)
    if not cs:
        return {}
    n = st["X"].shape[0]
    if cache is not None and Path(cache).exists():
        z = np.load(cache, allow_pickle=True)
        zc = [str(c) for c in z["cwes"]]
        if set(cs) <= set(zc) and int(z["n"]) == n and int(z["dim"]) == st["X"].shape[1]:
            # exact OR superset cache (e.g. shared reuses the 12-probe within cache):
            # slice requested columns, never touch X.
            idx = {c: i for i, c in enumerate(zc)}
            Lg = z["Lg"]
            tag = "exact" if zc == cs else f"subset of {len(zc)}"
            print(f"[logits] reuse {cache} ({len(cs)} probes, {tag}, {n} rows)", file=sys.stderr)
            return {c: Lg[:, idx[c]] for c in cs}
        print(f"[logits] {cache} mismatch (cwes/n/dim) -> recompute", file=sys.stderr)
    Wm = np.stack([W[c] for c in cs], axis=1).astype(np.float32)   # (hidden, k)
    bv = np.array([b[c] for c in cs], np.float32)
    Lg = np.empty((n, len(cs)), np.float32)                        # (n_tok, k) ~ tiny
    t0 = time.time()
    nchunk = (n + chunk - 1) // chunk
    for ci, s in enumerate(range(0, n, chunk)):
        e = min(s + chunk, n)
        Lg[s:e] = st["X"][s:e] @ Wm + bv
        if ci % 5 == 0 or e == n:
            print(f"[logits] {e}/{n} rows ({100*e/n:.0f}%) chunk {ci+1}/{nchunk} "
                  f"{time.time()-t0:.0f}s", file=sys.stderr)
    if cache is not None:
        np.savez(cache, Lg=Lg, cwes=np.array(cs), n=np.int64(n),
                 dim=np.int64(st["X"].shape[1]))
        print(f"[logits] cached -> {cache} ({Lg.nbytes/1e6:.0f} MB)", file=sys.stderr)
    return {c: Lg[:, i] for i, c in enumerate(cs)}


def eval_cell(st, lg, pos_test_eids_c, neg_tok):
    """tokens_code_auc of a precomputed probe-logit array `lg` on
    {CWE-Y test code tokens, y_tok labels} u {neg_tok}."""
    pt = toks(st, pos_test_eids_c, code_only=True)
    if pt.size == 0 or neg_tok.size == 0:
        return float("nan"), 0
    ev = np.concatenate([pt, neg_tok])
    lab = st["y_tok"][ev]
    if len(np.unique(lab)) < 2:
        return float("nan"), int((st["y_tok"][pt] == 1).sum())
    return auc_or_nan(lab, lg[ev]), int((st["y_tok"][pt] == 1).sum())


# ----------------------------- A) within-PV ----------------------------------
def within_pv(st, epochs, device, seed, pretrained=None, save_cb=None, logits_cache=None):
    clean_fit = [e for e in st["fit_eids"] if st["is_clean"][e]]
    clean_test = [e for e in st["test_eids"] if st["is_clean"][e]]
    neg_test_tok = toks(st, clean_test, code_only=True)

    test_pos = Counter(st["cwe_of"][e] for e in st["test_eids"] if st["cwe_of"][e])
    feasible = sorted([c for c, n in test_pos.items() if n >= MIN_TRUST_POS],
                      key=lambda c: (fam_of(c) != "mem", c))
    print(f"[within] feasible CWEs (>=10 test pos): "
          f"{[(c, test_pos[c], fam_of(c)) for c in feasible]}", file=sys.stderr)

    pos_test_of = {c: [e for e in st["test_eids"] if st["cwe_of"][e] == c] for c in feasible}
    n_test_pos = {c: len(pos_test_of[c]) for c in feasible}
    npos_tok = {c: int((st["y_tok"][toks(st, pos_test_of[c], code_only=True)] == 1).sum())
                for c in feasible}

    # n_train_pos needed for diagonal report regardless of train/reuse path
    n_train_pos = {c: len([e for e in st["fit_eids"] if st["cwe_of"][e] == c]) for c in feasible}
    if pretrained is not None:                  # RESUME: reuse cached probes, skip training
        W, b = pretrained
        W = {c: W[c] for c in feasible if c in W}; b = {c: b[c] for c in feasible if c in b}
        print(f"[within] reusing {len(W)} cached probes (skip training)", file=sys.stderr)
    else:
        W, b = {}, {}
        for c in feasible:
            wb, _ = train_cwe_probe(st, c, clean_fit, epochs, device)
            if wb is None:
                continue
            W[c], b[c] = wb
            print(f"[within] trained {c} ({fam_of(c)}) n_pos={n_train_pos[c]} "
                  f"|w|={np.linalg.norm(W[c]):.2f}", file=sys.stderr)
        if save_cb is not None:                 # CHECKPOINT probes BEFORE the costly eval/bootstrap
            save_cb(W, b)
            print(f"[within] checkpointed {len(W)} probes after training", file=sys.stderr)
    trained = [c for c in feasible if c in W]
    assert all(fam_of(c) in ("mem", "other") for c in trained), \
        f"untaxonomied feasible CWE(s): {[c for c in trained if fam_of(c)=='?']}"

    # Stage A: ONE chunked, cached, heartbeat'd matmul over X -> per-probe logits.
    Lg = precompute_logits(st, W, b, cache=logits_cache)
    K = len(trained)
    Lg_mat = np.stack([Lg[c] for c in trained], axis=0)     # (K, n_tok) f32; bootstrap just indexes
    fam = [fam_of(c) for c in trained]
    grp = {"mem": [i for i in range(K) if fam[i] == "mem"],
           "other": [i for i in range(K) if fam[i] == "other"]}

    # code-only token indices per example (built ONCE); bootstrap reps just concat.
    _boot_eids = set(clean_test)
    for c in trained:
        _boot_eids.update(pos_test_of[c])
    ctok = {e: (lambda t: t[st["is_code"][t]])(st["tok_by_eid"][e]) for e in _boot_eids}
    def ftoks(eids):
        return np.concatenate([ctok[e] for e in eids]) if eids else np.array([], np.int64)

    # ---- Stage B (1): point-estimate matrix via sklearn AUC (exact, prior harness)
    M = {cx: {} for cx in trained}
    for i, cx in enumerate(trained):
        for cy in trained:
            a, _ = eval_cell(st, Lg[cx], pos_test_of[cy], neg_test_tok)
            M[cx][cy] = a
        print(f"[matrix] row {i+1}/{K} {cx}", file=sys.stderr)
    Mpt = np.array([[M[cx][cy] for cy in trained] for cx in trained])   # (K,K)
    npos_vec = np.array([npos_tok[c] for c in trained], float)

    def agg_blocks(Mb, w):
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

    blk = agg_blocks(Mpt, npos_vec)
    blk_keys = list(blk.keys())

    # ---- Stage B (2): grouped bootstrap CIs from cached logits (parallel, fast AUC)
    def block_rep(rep):
        r = np.random.default_rng(seed + 101 + rep)
        cs = [clean_test[j] for j in r.integers(0, len(clean_test), len(clean_test))]
        negb = ftoks(cs)
        Mb = np.full((K, K), np.nan); w = np.zeros(K)
        for iy, cy in enumerate(trained):
            pe = pos_test_of[cy]
            pt = ftoks([pe[j] for j in r.integers(0, len(pe), len(pe))])
            ev = np.concatenate([pt, negb]); y = st["y_tok"][ev]
            w[iy] = int((st["y_tok"][pt] == 1).sum())
            Mb[:, iy] = fast_auc_batch(Lg_mat[:, ev], y)
        return agg_blocks(Mb, w)

    bres = _pmap(block_rep, list(range(N_BOOT // 2)), "block")
    boot = {k: [v[k] for v in bres if v[k] == v[k]] for k in blk_keys}
    blk_ci = {k: ([float(np.percentile(boot[k], 2.5)), float(np.percentile(boot[k], 97.5)), len(boot[k])]
                  if boot[k] else [float("nan")] * 3) for k in blk_keys}

    # diagonal CIs (resample test pos + clean pool per CWE per rep)
    def diag_rep(rep):
        r = np.random.default_rng(seed + 5001 + rep)
        out = np.full(K, np.nan)
        for ix, c in enumerate(trained):
            pe = pos_test_of[c]
            pt = ftoks([pe[j] for j in r.integers(0, len(pe), len(pe))])
            cs = [clean_test[j] for j in r.integers(0, len(clean_test), len(clean_test))]
            ev = np.concatenate([pt, ftoks(cs)]); y = st["y_tok"][ev]
            out[ix] = fast_auc_batch(Lg_mat[ix:ix + 1, ev], y)[0]
        return out

    dres = np.array(_pmap(diag_rep, list(range(N_BOOT)), "diag"))       # (N_BOOT, K)
    diag = {}
    for ix, c in enumerate(trained):
        col = dres[:, ix]; col = col[~np.isnan(col)]
        lo, hi = (float(np.percentile(col, 2.5)), float(np.percentile(col, 97.5))) if col.size else (float("nan"),) * 2
        diag[c] = {"auc": M[c][c], "ci": [lo, hi], "family": fam_of(c),
                   "n_train_pos": n_train_pos.get(c, 0), "n_test_pos": n_test_pos[c],
                   "trust": n_test_pos[c] >= MIN_TRUST_POS, "boot_reps": int(col.size)}

    return dict(
        analysis="within_pv", metric="tokens_code_auc",
        feasible_cwes=feasible, trained_cwes=trained,
        mem_cwes=sorted(c for c in trained if fam_of(c) == "mem"),
        other_cwes=sorted(c for c in trained if fam_of(c) == "other"),
        n_train_pos=n_train_pos, n_test_pos=n_test_pos, n_pos_tokens=npos_tok,
        n_clean_fit=len(clean_fit), n_clean_test=len(clean_test),
        n_neg_test_tokens=int(neg_test_tok.size),
        matrix=M, family_blocks=blk, family_blocks_ci=blk_ci, diagonal=diag,
        note="all-clean recipe; clean pool = label==0 PV safe-halves (100% C/C++, "
             "language-clean by construction); language held fixed; CIs over examples.",
    ), (W, b)


# ----------------------- B) SVEN<->PV shared-CWE -----------------------------
def shared_transfer(pv, sven, pv_wb, epochs, device, out=None):
    """For each shared memory CWE: SVEN->PV, PV->PV, PV->SVEN-C, SVEN->SVEN-C."""
    cf = (lambda name: str(Path(out) / name)) if out is not None else (lambda name: None)
    Wpv, bpv = pv_wb
    pv_clean_test = [e for e in pv["test_eids"] if pv["is_clean"][e]]
    pv_neg = toks(pv, pv_clean_test, code_only=True)
    pv_pos = {c: [e for e in pv["test_eids"] if pv["cwe_of"][e] == c] for c in SHARED_MEM}

    # SVEN-C: C/C++ slice only (eval on language-matched SVEN negatives)
    sven_clean_test_c = [e for e in sven["test_eids"]
                         if sven["is_clean"][e] and sven["lang_of"][e] in CPP]
    sven_neg_c = toks(sven, sven_clean_test_c, code_only=True)
    sven_pos = {c: [e for e in sven["test_eids"]
                    if sven["cwe_of"][e] == c and sven["lang_of"][e] in CPP] for c in SHARED_MEM}
    sven_clean_fit = [e for e in sven["fit_eids"] if sven["is_clean"][e]]

    # train SVEN all-clean probes for the shared CWEs (exp-10: CWE-X vuln u SVEN cwe==null clean)
    Wsv, bsv, sv_ntp = {}, {}, {}
    for c in SHARED_MEM:
        wb, ntp = train_cwe_probe(sven, c, sven_clean_fit, epochs, device)
        sv_ntp[c] = ntp
        if wb is not None:
            Wsv[c], bsv[c] = wb

    # Cached chunked logit passes. PV->PV reuses the 12-probe within-PV cache
    # (superset slice -> no extra 118 GB sweep). The one new PV sweep is SVEN
    # probes on PV X; SVEN-side passes are 8 GB. Each cached for resume.
    Lg_pv_pv = precompute_logits(pv, {c: Wpv[c] for c in SHARED_MEM if c in Wpv},
                                 {c: bpv[c] for c in SHARED_MEM if c in Wpv},
                                 cache=cf("logits_pv.npz"))
    Lg_sv_pv = precompute_logits(sven, {c: Wpv[c] for c in SHARED_MEM if c in Wpv},
                                 {c: bpv[c] for c in SHARED_MEM if c in Wpv},
                                 cache=cf("logits_sven_pvprobes.npz"))
    Lg_pv_sv = precompute_logits(pv, Wsv, bsv, cache=cf("logits_pv_svenprobes.npz"))
    Lg_sv_sv = precompute_logits(sven, Wsv, bsv, cache=cf("logits_sven_svenprobes.npz"))

    table = {}
    for c in SHARED_MEM:
        row = {"family": "mem", "n_pv_test_pos": len(pv_pos[c]),
               "n_sven_c_test_pos": len(sven_pos[c]),
               "n_sven_train_pos": sv_ntp[c], "n_pv_train_pos": None}
        if c in Wpv:
            row["n_pv_train_pos"] = int(sum(1 for e in pv["fit_eids"] if pv["cwe_of"][e] == c))
            row["PV->PV"], _ = eval_cell(pv, Lg_pv_pv[c], pv_pos[c], pv_neg)
            row["PV->SVEN_C"], _ = eval_cell(sven, Lg_sv_pv[c], sven_pos[c], sven_neg_c)
        else:
            row["PV->PV"] = float("nan"); row["PV->SVEN_C"] = float("nan")
        if c in Wsv:
            row["SVEN->PV"], _ = eval_cell(pv, Lg_pv_sv[c], pv_pos[c], pv_neg)
            row["SVEN->SVEN_C"], _ = eval_cell(sven, Lg_sv_sv[c], sven_pos[c], sven_neg_c)
        else:
            row["SVEN->PV"] = float("nan"); row["SVEN->SVEN_C"] = float("nan")
        table[c] = row
        print(f"[shared] {c}: SVEN->PV={row['SVEN->PV']:.3f} PV->PV={row['PV->PV']:.3f} "
              f"PV->SVEN_C={row['PV->SVEN_C']:.3f} SVEN->SVEN_C={row['SVEN->SVEN_C']:.3f} "
              f"(nPVpos={row['n_pv_test_pos']} nSVENcpos={row['n_sven_c_test_pos']})", file=sys.stderr)

    return dict(analysis="shared_transfer", metric="tokens_code_auc",
                shared_mem_cwes=SHARED_MEM,
                n_pv_clean_test=len(pv_clean_test), n_sven_c_clean_test=len(sven_clean_test_c),
                table=table,
                note="same model+layer both sides => probe dims match. SVEN-C = SVEN "
                     "C/C++ slice (language-matched negatives). PV clean = label==0 "
                     "safe-halves. MIN_TRUST_POS=10 flagged per cell.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--pv-acts", required=True)
    ap.add_argument("--pv-dataset", required=True)
    ap.add_argument("--sven-acts", default="")
    ap.add_argument("--sven-dataset", default="")
    ap.add_argument("--sven-split", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["within", "shared", "both"], default="both")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    device = "cpu"   # 118 GB PV acts won't fit a 96 GB GPU; repo rule: probes cheap on CPU
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    pv_rows = [json.loads(l) for l in open(args.pv_dataset)]
    pv = build_state("pv", args.pv_acts, pv_rows, args.layer, kind="pv")

    pf = out / "probes_pv.npz"

    def save_pv_probes(W, b):
        cs = sorted(W)
        np.savez_compressed(pf, cwes=np.array(cs), W=np.stack([W[c] for c in cs]),
                            b=np.array([b[c] for c in cs], np.float32), layer=np.int32(args.layer))

    def load_pv_probes():
        z = np.load(pf, allow_pickle=True)
        assert int(z["layer"]) == args.layer, \
            f"cached probes layer {int(z['layer'])} != requested {args.layer}"
        cs = [str(c) for c in z["cwes"]]
        assert z["W"].shape[1] == pv["X"].shape[1], \
            f"cached probe dim {z['W'].shape[1]} != X dim {pv['X'].shape[1]}"
        return ({c: z["W"][i] for i, c in enumerate(cs)},
                {c: float(z["b"][i]) for i, c in enumerate(cs)})

    pv_wb = (None, None)
    if args.mode in ("within", "both"):
        wf = out / "pv_within.json"
        if wf.exists() and pf.exists() and not args.force:
            print(f"[within] {wf} + probes exist, skip (--force to redo)", file=sys.stderr)
            pv_wb = load_pv_probes()
        else:
            # RESUME: if probes were checkpointed (training done) but bootstrap
            # didn't finish, reuse them and skip the costly retrain.
            pre = load_pv_probes() if (pf.exists() and not args.force) else None
            if pre is not None:
                print(f"[within] resuming from checkpointed probes (skip training)", file=sys.stderr)
            res_w, pv_wb = within_pv(pv, args.epochs, device, args.seed,
                                     pretrained=pre, save_cb=save_pv_probes,
                                     logits_cache=str(out / "logits_pv.npz"))
            wf.write_text(json.dumps(res_w, indent=2, default=lambda x: None if x != x else x))
            save_pv_probes(*pv_wb)             # final (idempotent) probe save
            print(f"[within] wrote {wf} + {pf}", file=sys.stderr)

    if args.mode in ("shared", "both"):
        assert args.sven_acts and args.sven_dataset and args.sven_split, \
            "shared mode needs --sven-acts/--sven-dataset/--sven-split"
        sf = out / "cross_shared.json"
        if sf.exists() and not args.force:
            print(f"[shared] {sf} exists, skip (--force to redo)", file=sys.stderr)
            return
        if pv_wb[0] is None:               # need PV probes for PV->* rows
            assert pf.exists(), "shared mode needs PV probes; run within first"
            pv_wb = load_pv_probes()
        sven_rows = [json.loads(l) for l in open(args.sven_dataset)]
        sven = build_state("sven", args.sven_acts, sven_rows, args.layer, kind="sven",
                           split_file=(args.sven_dataset, args.sven_split))
        res_s = shared_transfer(pv, sven, pv_wb, args.epochs, device, out=out)
        sf.write_text(json.dumps(res_s, indent=2, default=lambda x: None if x != x else x))
        print(f"[shared] wrote {sf}", file=sys.stderr)


if __name__ == "__main__":
    main()
