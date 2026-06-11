# [ai-generated]
"""exp-21 RE-EXEC — cross-CWE transfer matrix on the DEFAULT metric `tokens_code_auc`.

The original exp-21 headlined pair-accuracy (rank a vuln above its OWN patched
pair) and wrongly concluded "memory unlearnable" — contradicting exp-10, which
showed specialized memory probes reach tokens_code_auc ~0.64-0.77. This script
REDOES the cross-CWE transfer matrix on the honest token-level AUC, REUSING the
SAVED per-CWE probes/logits (no re-extraction, no re-training).

Cell[train=X][test=Y] = honest token-level ROC-AUC of probe-X's per-token logits
on the held-out TEST split, two negative-pool variants:
  * `own`    (CANONICAL tokens_code_auc, the exp-10 recipe): negatives = code
    tokens of all subtractive safe-half (cwe==null) examples PLUS the CWE-Y vuln
    examples' OWN non-vuln code tokens. This matches honest_scoring (every y==0
    live-code token in the eval subset is a negative). Absolute AUCs here are the
    headline; cross-column negative composition is NOT identical (each column adds
    its own non-vuln tokens) — see `shared` for the comparability check.
  * `shared` (sensitivity / pure-transfer): negatives = the safe-half pool ONLY,
    identical for every column → clean cross-column transfer comparison.
  positives (both) = tight∩is_code vuln tokens (y_tok==1) of CWE-Y vuln examples.
  Diagonal X==Y = subtractive-subset self-detection. NOTE: this is exp-21's OWN
  regime (subtractive subset, tight∩is_code labels, subtractive safe-half
  negatives) — it only QUALITATIVELY tracks exp-10 (full-SVEN, old per-token y,
  all-cwe==null negatives, n 3-8x larger); it is NOT a numeric reproduction.

Natural-probe logits come straight from `logits_percwe.npz` (headline). The
balanced control (every probe capped to 15 train pairs) is recomputed from saved
activations × `probes_percwe.npz` W_balanced when --acts is given; --acts also
triggers a provenance identity check (saved logit_nat == X @ W_natural + b).

Trust: memory test CWEs have tiny n (787~2, 190~3, 476~4) — only block means
(with bootstrap CIs) and diagonals carrying `trust=True` (n_test_vuln_examples>=10)
are interpretable; individual small cells are noise.

Outputs --out/matrix_tokenauc.json.
"""
from __future__ import annotations
import argparse, difflib, json, os, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

REPO = Path(os.environ.get("REPO", Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(REPO))
from sklearn.metrics import roc_auc_score                              # noqa: E402
from src.remotes.the cluster.train_eval import load_or_make_split        # noqa: E402
from src.eval.code_mask import live_code_char_ranges                  # noqa: E402

INJ = {"CWE-089", "CWE-078", "CWE-022", "CWE-079"}
N_BOOT = 1000          # diagonal CI reps
N_BOOT_BLOCK = 500     # block CI reps (each rep recomputes every cell in 4 blocks)
MIN_TRUST_POS = 10     # exp-10 convention: per-cell trust floor (n test vuln examples)


def tight_spans(before: str, after: str):
    sm = difflib.SequenceMatcher(a=before, b=after, autojunk=False)
    return [(i1, i2) for t, i1, i2, j1, j2 in sm.get_opcodes()
            if t in ("replace", "delete") and i2 > i1]


def overlaps_live(spans, live):
    if live is None:
        return bool(spans)
    for s, e in spans:
        for ls, le in live:
            if s < le and e > ls:
                return True
    return False


def auc_or_nan(labels, scores):
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="scratch percwe_<slug> dir")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--acts", default="", help="token_activations dir; enables balanced + identity check")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    run = Path(args.run_dir)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ss = np.random.SeedSequence(args.seed)
    rng_diag, rng_block = (np.random.default_rng(s) for s in ss.spawn(2))

    # ---- saved per-token natural logits + token metadata ----
    lz = np.load(run / "logits_percwe.npz", allow_pickle=True)
    cwes_npz = [str(c) for c in lz["cwes"]]          # ROW ORDER of logit_nat (authoritative)
    row_of = {c: i for i, c in enumerate(cwes_npz)}
    eids = lz["example_id"].astype(np.int64)
    is_code = lz["is_code"].astype(bool)
    y_tok = lz["y_tok"].astype(np.int8)
    logit_nat = lz["logit"].astype(np.float32)            # [ncwe, ntok], row i = probe cwes_npz[i]
    assert logit_nat.shape == (len(cwes_npz), len(eids)), "logit shape mismatch"
    # B1 invariant: y_tok is exactly tight∩is_code on vuln tokens, 0 elsewhere
    assert set(np.unique(y_tok)).issubset({0, 1}), "y_tok not binary"
    assert (y_tok[~is_code] == 0).all(), "y_tok positive on non-code token"

    # ---- dataset + split (must reuse the PERSISTED seed-42 split) ----
    assert Path(args.split).exists(), f"split file missing: {args.split} (would mint a NEW split)"
    rows, train_eids, test_eids = load_or_make_split(Path(args.dataset), Path(args.split))
    train_set = set(int(e) for e in train_eids)
    test_set = set(int(e) for e in test_eids)
    grp = defaultdict(list)
    for eid, r in enumerate(rows):
        grp[(r.get("_file_name"), r.get("_func_name"))].append(eid)
    sub_pairs = []
    for es in grp.values():
        vs = [e for e in es if rows[e]["label"] == 1]
        sf = [e for e in es if rows[e]["label"] == 0]
        for i in range(min(len(vs), len(sf))):
            v, s = vs[i], sf[i]
            sp = tight_spans(rows[v]["code"], rows[s]["code"])
            if overlaps_live(sp, live_code_char_ranges(rows[v]["code"], rows[v].get("lang") or "")):
                sub_pairs.append((v, s))
    cwe_of = {v: rows[v].get("cwe") for v, s in sub_pairs}
    # split-safety: a pair never straddles train/test
    for v, s in sub_pairs:
        assert (v in test_set) == (s in test_set), f"pair {v},{s} straddles split"
        assert (v in train_set) == (s in train_set), f"pair {v},{s} straddles split"

    safe_test = sorted({s for v, s in sub_pairs if v in test_set})        # shared negative pool
    vuln_test_by_cwe = defaultdict(list)
    for v, s in sub_pairs:
        if v in test_set:
            vuln_test_by_cwe[cwe_of[v]].append(v)
    cwes = sorted(set(cwe_of.values()), key=lambda c: (c not in INJ, c))   # iteration order, inj first
    assert set(cwes) == set(cwes_npz), f"CWE set mismatch saved={cwes_npz} vs recomputed={cwes}"

    # token index lists per example (fast masking + bootstrap; grouped resampling)
    tok_by_eid = defaultdict(list)
    for i, e in enumerate(eids):
        tok_by_eid[int(e)].append(i)
    tok_by_eid = {e: np.asarray(ix, np.int64) for e, ix in tok_by_eid.items()}

    def code_tokens(examples):
        if not examples:
            return np.array([], np.int64)
        t = np.concatenate([tok_by_eid[e] for e in examples])
        return t[is_code[t]]

    neg_tok = code_tokens(safe_test)                                       # safe-pool code tokens
    assert int((y_tok[neg_tok] == 1).sum()) == 0, "safe pool carries positive tokens"
    # per-CWE positive (vuln) tokens and own non-vuln code tokens (probe-independent)
    pos_tok_by_cwe, own_neg_by_cwe = {}, {}
    for c in cwes:
        ct = code_tokens(vuln_test_by_cwe.get(c, []))
        pos_tok_by_cwe[c] = ct[y_tok[ct] == 1]
        own_neg_by_cwe[c] = ct[y_tok[ct] == 0]
    n_test = {c: len(vuln_test_by_cwe.get(c, [])) for c in cwes}
    npos_by_cwe = {c: int(len(pos_tok_by_cwe[c])) for c in cwes}

    def cell(probe_logits, c_y, include_own):
        pos = pos_tok_by_cwe[c_y]
        if len(pos) == 0:
            return float("nan")
        neg = np.concatenate([neg_tok, own_neg_by_cwe[c_y]]) if include_own else neg_tok
        ev = np.concatenate([pos, neg])
        lab = np.concatenate([np.ones(len(pos), np.int8), np.zeros(len(neg), np.int8)])
        return auc_or_nan(lab, probe_logits[ev])

    def matrix(logit_arr, include_own):
        return {cx: {cy: cell(logit_arr[row_of[cx]], cy, include_own) for cy in cwes} for cx in cwes}

    def block_means(M):
        groups = {"inj": [c for c in cwes if c in INJ], "mem": [c for c in cwes if c not in INJ]}
        b = {}
        for gtr, ctr in groups.items():
            for gte, cte in groups.items():
                num = den = 0.0
                for cx in ctr:
                    for cy in cte:
                        a = M[cx][cy]; w = npos_by_cwe.get(cy, 0)   # weight by test-col pos tokens
                        if a == a and w:
                            num += a * w; den += w
                b[f"{gtr}->{gte}"] = (num / den) if den else float("nan")
        return b

    M_own = matrix(logit_nat, include_own=True)        # canonical tokens_code_auc (headline)
    M_shared = matrix(logit_nat, include_own=False)    # sensitivity: identical negatives per column
    blk_own = block_means(M_own)
    blk_shared = block_means(M_shared)

    # ---- diagonal bootstrap CIs (stratified: resample vuln & safe pools separately) ----
    diag = {}
    for c in cwes:
        pos_e = vuln_test_by_cwe.get(c, [])
        a = M_own[c][c]
        boots, valid = [], 0
        if pos_e and a == a:
            for _ in range(N_BOOT):
                vs = [pos_e[j] for j in rng_diag.choice(len(pos_e), len(pos_e), replace=True)]
                sf = [safe_test[j] for j in rng_diag.choice(len(safe_test), len(safe_test), replace=True)]
                vt = code_tokens(vs); st = code_tokens(sf)
                lab = np.concatenate([y_tok[vt], np.zeros(len(st), np.int8)])  # own non-vuln stay in vt
                if len(np.unique(lab)) < 2:
                    continue
                boots.append(auc_or_nan(lab, logit_nat[row_of[c]][np.concatenate([vt, st])]))
                valid += 1
        lo, hi = ((float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
                  if boots else (float("nan"), float("nan")))
        diag[c] = {"auc": a, "ci": [lo, hi], "n_test_vuln_examples": n_test[c],
                   "n_pos_tokens": npos_by_cwe[c], "trust": n_test[c] >= MIN_TRUST_POS,
                   "boot_valid_reps": valid}

    # ---- block bootstrap CIs (own variant; resample safe pool + each column's vuln pool) ----
    def block_boot():
        groups = {"inj": [c for c in cwes if c in INJ], "mem": [c for c in cwes if c not in INJ]}
        acc = {f"{a}->{b}": [] for a in groups for b in groups}
        for _ in range(N_BOOT_BLOCK):
            sf = [safe_test[j] for j in rng_block.choice(len(safe_test), len(safe_test), replace=True)]
            negb = code_tokens(sf)
            colpos, colw = {}, {}
            for cy in cwes:
                ce = vuln_test_by_cwe.get(cy, [])
                if not ce:
                    colpos[cy] = None; colw[cy] = 0; continue
                cs = [ce[j] for j in rng_block.choice(len(ce), len(ce), replace=True)]
                colpos[cy] = code_tokens(cs); colw[cy] = int((y_tok[colpos[cy]] == 1).sum())
            for gtr, ctr in groups.items():
                for gte, cte in groups.items():
                    num = den = 0.0
                    for cx in ctr:
                        for cy in cte:
                            if colpos[cy] is None:
                                continue
                            ev = np.concatenate([colpos[cy], negb])
                            lab = np.concatenate([y_tok[colpos[cy]], np.zeros(len(negb), np.int8)])
                            if len(np.unique(lab)) < 2:
                                continue
                            a = auc_or_nan(lab, logit_nat[row_of[cx]][ev]); w = colw[cy]
                            if a == a and w:
                                num += a * w; den += w
                    if den:
                        acc[f"{gtr}->{gte}"].append(num / den)
        return {k: ([float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), len(v)]
                    if v else [float("nan"), float("nan"), 0]) for k, v in acc.items()}

    res = dict(
        model=args.model, layer=args.layer, metric="tokens_code_auc",
        eval_regime="subtractive_subset; tight∩is_code labels; neg=subtractive_safe_halves(+own_nonvuln for 'own')",
        note_vs_exp10="diagonal QUALITATIVELY tracks exp-10; NOT a numeric reproduction (diff subset/labels/neg-pool/n).",
        cwes=cwes, cwes_row_order=cwes_npz, inj=sorted(INJ), min_trust_pos=MIN_TRUST_POS,
        n_test_vuln_examples=n_test, n_pos_tokens=npos_by_cwe,
        n_neg_tokens_safe_pool=int(len(neg_tok)), n_safe_test_examples=len(safe_test),
        n_own_neg_tokens_by_cwe={c: int(len(own_neg_by_cwe[c])) for c in cwes},
        auc_natural_own=M_own, auc_natural_shared=M_shared,
        block_auc_natural_own=blk_own, block_auc_natural_shared=blk_shared,
        block_auc_natural_own_ci=block_boot(),
        diagonal_natural=diag,
    )

    # ---- balanced control + provenance identity check (needs activations) ----
    if args.acts:
        acts = Path(args.acts)
        cand = [p for p in acts.glob("token_activations_layer*.npz")
                if int(p.stem.replace("token_activations_layer", "")) == args.layer]
        pz = np.load(run / "probes_percwe.npz", allow_pickle=True)
        pc = [str(c) for c in pz["cwes"]]
        assert pc == cwes_npz, f"probes npz CWE order {pc} != logits order {cwes_npz}"
        Wn = {c: pz["W_natural"][i].astype(np.float32) for i, c in enumerate(pc)}
        bn = {c: float(pz["b_natural"][i]) for i, c in enumerate(pc)}
        Wb = {c: pz["W_balanced"][i].astype(np.float32) for i, c in enumerate(pc)}
        bb = {c: float(pz["b_balanced"][i]) for i, c in enumerate(pc)}
        if cand:
            npz = np.load(cand[0])
            X = npz["X"].astype(np.float32)
            xe = npz["example_ids"].astype(np.int64)
            assert np.array_equal(xe, eids), "activation token order != logits token order"
            # provenance: saved natural logits must equal X @ W_natural + b
            recon = np.stack([(X @ Wn[c] + bn[c]).astype(np.float32) for c in cwes_npz])
            mad = float(np.abs(recon - logit_nat).max())
            res["identity_check_max_abs_diff"] = mad
            assert mad < 1e-2, f"saved logits disagree with X@W_natural (max abs diff {mad})"
            logit_bal = np.stack([(X @ Wb[c] + bb[c]).astype(np.float32) for c in cwes_npz])
            res["auc_balanced_own"] = matrix(logit_bal, include_own=True)
            res["block_auc_balanced_own"] = block_means(matrix(logit_bal, include_own=True))
        else:
            res["auc_balanced_own"] = "NO_ACTS_LAYER_NPZ"

    (out / "matrix_tokenauc.json").write_text(json.dumps(res, indent=2,
                                              default=lambda x: None if x != x else x))
    print(f"[recompute] {args.model}: diagonal self-detect (token-AUC, 'own'):", file=sys.stderr)
    for c in cwes:
        d = diag[c]; fam = "INJ" if c in INJ else "mem"
        print(f"  {c} {fam}: {d['auc']:.3f} CI{[round(x,3) for x in d['ci']]} "
              f"trust={d['trust']} (n_ex={d['n_test_vuln_examples']}, reps={d['boot_valid_reps']})",
              file=sys.stderr)
    print(f"[recompute] blocks(own) = { {k: round(v,3) for k,v in blk_own.items()} }", file=sys.stderr)
    print(f"[recompute] blocks(shared) = { {k: round(v,3) for k,v in blk_shared.items()} }", file=sys.stderr)


if __name__ == "__main__":
    main()
