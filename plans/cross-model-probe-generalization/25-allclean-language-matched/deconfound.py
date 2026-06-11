# [ai-generated]
"""exp-25 — does the exp-10/21 memory recovery survive a LANGUAGE/FUNCTION-matched eval?

Background. exp-10/21 trained per-CWE probes the "all-clean" way (CWE-X vuln vs
ALL cwe==null clean, full SVEN, annotated token_labels positives) and reported a
memory diagonal of 0.73/0.77/0.64 (CWE-125/416/476) — the spine of claim #3
("memory signal exists; under-allocation not absence"). But in SVEN the memory
CWEs are 100% C/C++ and the all-clean negative pool is 53% Python: a pure
C-vs-Python token detector scores ~0.765 on that exact split. So the all-clean
diagonal CANNOT distinguish "memory-safety signal" from "language detector".

This script holds the all-clean-trained probes FIXED and re-evaluates them under
negative pools that remove the language confound:
  * allclean      — original pool (reproduction gate; MUST match the saved target).
  * conly         — negatives = C/C++-only clean test tokens (language-matched for memory).
  * pyonly        — negatives = Python-only clean test tokens (language-matched for injection).
  * matchedpatch  — negatives = the CWE's OWN paired safe-half (patched) code tokens
                    (function + language matched; the decisive control).
Positives are held identical across regimes (annotated token_labels==1, code-only,
CWE-Y vuln test examples) so ONLY the negative composition changes.

It also (a) retrains per-CWE probes with C-only-clean negatives ("conly-trained")
and a single pooled memory-family probe, re-evals them under the matched regimes;
(b) computes a LANGUAGE-NULL column next to every probe number — the ROC-AUC of a
pure C-vs-Python indicator on the *exact same* (pos,neg) token set; and (c) writes
grouped 5-fold x 3-seed CV cells (resumable) for the trusted CWEs in both regimes.

Reuses the on-scratch KEPT full-SVEN activations (token_activations_layer25.npz +
offsets.npz) — NO re-extraction. Diagonal of the allclean regime reproduces
exp-10/21 bit-ish (train_one_layer parity).

Outputs (all under --out):
  repro_gate.json     diagonal allclean vs the saved target
  deconfound.json     per-regime diagonal + matrix + blocks + language-null, both probe sets
  cv/<regime>_<cwe>_<fold>_<seed>.json   incremental CV cells (skip-if-exists)
  probes_dc.npz       saved W/b for allclean + conly + pooled probes
"""
from __future__ import annotations
import argparse, json, os, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

REPO = Path(os.environ.get("REPO", Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(REPO))
from sklearn.metrics import roc_auc_score                                  # noqa: E402
from src.eval.token_data import parse_spans, char_spans_to_token_spans, token_labels_array  # noqa: E402
from src.eval.code_mask import code_only_mask                             # noqa: E402
from src.remotes.train_eval import load_or_make_split, pair_group_key  # noqa: E402
from src.training.train_probe_spanmax import train_one_layer             # noqa: E402

INJ = {"CWE-089", "CWE-078", "CWE-022", "CWE-079"}
MEM_POOL = ["CWE-125", "CWE-416", "CWE-476", "CWE-787", "CWE-190"]
VAL_FRAC, VAL_SEED = 0.15, 42
MIN_TRUST_POS = 10
N_BOOT = 1000
TRUSTED_CV = ["CWE-089", "CWE-078", "CWE-022", "CWE-079", "CWE-125", "CWE-416", "CWE-476"]
CV_FOLDS, CV_SEEDS = 5, [0, 1, 2]


def auc_or_nan(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--min-train-pos", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--do-cv", action="store_true")
    ap.add_argument("--cv-only", action="store_true", help="skip the main eval, only fill CV cells")
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    acts = Path(args.acts); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "cv").mkdir(exist_ok=True)
    L = args.layer
    rng = np.random.default_rng(args.seed)

    assert Path(args.split).exists(), f"split file missing: {args.split}"
    rows, train_eids, test_eids = load_or_make_split(Path(args.dataset), Path(args.split))
    train_set, test_set = set(int(e) for e in train_eids), set(int(e) for e in test_eids)

    offs = np.load(acts / "offsets.npz")
    n_rows = len(offs.files)
    assert n_rows == len(rows), f"acts rows {n_rows} != dataset rows {len(rows)}"
    offsets_per_row = [offs[f"offsets_row_{i:04d}"] for i in range(n_rows)]

    cand = [p for p in acts.glob("token_activations_layer*.npz")
            if int(p.stem.replace("token_activations_layer", "")) == L]
    if not cand:
        raise SystemExit(f"no token_activations for layer {L} under {acts}")
    npz = np.load(cand[0])
    X = npz["X"].astype(np.float32)
    eids = npz["example_ids"].astype(np.int64)

    def is_c_lang(e):
        return (rows[e].get("lang") or "").lower() in ("c", "cpp", "c++")

    # per-token annotated label + code mask + language indicator, aligned to X
    y_tok = np.zeros(len(eids), np.int8)
    is_code = np.zeros(len(eids), bool)
    is_c_tok = np.zeros(len(eids), bool)
    cur = 0
    for e in range(n_rows):
        n = int((eids == e).sum())
        o = offsets_per_row[e]
        if o.shape[0] != n:
            raise SystemExit(f"offset/token mismatch eid {e}: {o.shape[0]} vs {n}")
        assert (eids[cur:cur + n] == e).all(), f"non-contiguous eid block at {e}"
        off_list = [(int(s), int(t)) for s, t in o]
        tok_spans = char_spans_to_token_spans(parse_spans(rows[e]), off_list)
        lab, _ = token_labels_array(n, tok_spans)
        y_tok[cur:cur + n] = lab
        is_code[cur:cur + n] = code_only_mask(rows[e].get("code", ""),
                                              rows[e].get("lang", "") or "", o).astype(bool)
        is_c_tok[cur:cur + n] = is_c_lang(e)
        cur += n
    if cur != len(eids):
        raise SystemExit(f"consumed {cur} of {len(eids)}")

    cwe_of = {e: rows[e].get("cwe") for e in range(n_rows)}
    is_clean = {e: (rows[e].get("label") == 0 and not rows[e].get("cwe")) for e in range(n_rows)}

    # ---- matched-patch pairing (same as recompute_tokenauc): (_file_name,_func_name) groups,
    # pair i-th vuln with i-th safe-half. safe_of_cwe[c] = safe halves paired to CWE-c vuln. ----
    grp = defaultdict(list)
    for e in range(n_rows):
        grp[(rows[e].get("_file_name"), rows[e].get("_func_name"))].append(e)
    safe_of_vuln = {}
    for es in grp.values():
        vs = [e for e in es if rows[e].get("label") == 1]
        sf = [e for e in es if rows[e].get("label") == 0]
        for i in range(min(len(vs), len(sf))):
            safe_of_vuln[vs[i]] = sf[i]

    # 15% group-aware VAL carve of TRAIN (exp-10 parity)
    tr_eid_grp = {e: pair_group_key(rows[e]) for e in train_set}
    tr_groups = sorted(set(tr_eid_grp.values()))
    vr = np.random.default_rng(VAL_SEED)
    vr.shuffle(tr_groups)
    n_val = max(1, int(round(VAL_FRAC * len(tr_groups))))
    val_groups = set(tr_groups[:n_val])
    val_eids = {e for e, g in tr_eid_grp.items() if g in val_groups}
    fit_eids = train_set - val_eids

    clean_fit = [e for e in fit_eids if is_clean[e]]
    clean_fit_c = [e for e in clean_fit if is_c_lang(e)]
    clean_test = [e for e in test_set if is_clean[e]]

    train_pos = Counter(cwe_of[e] for e in fit_eids if cwe_of[e])
    cwes = sorted([c for c, n in train_pos.items() if n >= args.min_train_pos],
                  key=lambda c: (c not in INJ, c))
    print(f"[dc] {args.model} CWEs: {[(c, train_pos[c]) for c in cwes]}", file=sys.stderr)
    print(f"[dc] clean_fit={len(clean_fit)} clean_fit_C/C++={len(clean_fit_c)} "
          f"clean_test={len(clean_test)} C/C++_clean_test={sum(is_c_lang(e) for e in clean_test)}",
          file=sys.stderr)

    tok_by_eid = defaultdict(list)
    for i, e in enumerate(eids):
        tok_by_eid[int(e)].append(i)
    tok_by_eid = {e: np.asarray(ix, np.int64) for e, ix in tok_by_eid.items()}

    def toks(eid_list, code_only=False):
        if not eid_list:
            return np.array([], np.int64)
        t = np.concatenate([tok_by_eid[e] for e in eid_list])
        return t[is_code[t]] if code_only else t

    def fit_probe(pos_eids, neg_eids):
        ft = toks(list(pos_eids) + list(neg_eids))   # ALL tokens (no is_code gate) — exp-10 parity
        r = train_one_layer(X[ft], y_tok[ft], eids[ft], epochs=args.epochs,
                            device=device, verbose=False)
        return np.asarray(r["w"], np.float32), float(r["b"])

    pos_fit_of = {c: [e for e in fit_eids if cwe_of[e] == c] for c in cwes}
    pos_test_of = {c: [e for e in test_set if cwe_of[e] == c] for c in cwes}
    n_test_pos = {c: len(pos_test_of[c]) for c in cwes}

    # ---- train probe sets ----
    W_ac, b_ac, W_co, b_co = {}, {}, {}, {}
    for c in cwes:
        W_ac[c], b_ac[c] = fit_probe(pos_fit_of[c], clean_fit)       # all-clean (gate)
        W_co[c], b_co[c] = fit_probe(pos_fit_of[c], clean_fit_c)     # C-only-clean
        print(f"[dc] trained {c}: |w_ac|={np.linalg.norm(W_ac[c]):.2f} "
              f"|w_co|={np.linalg.norm(W_co[c]):.2f}", file=sys.stderr)
    # pooled memory probe: union of MEM_POOL vuln vs (allclean | conly)
    mem_pos_fit = [e for e in fit_eids if cwe_of[e] in MEM_POOL]
    Wp_ac, bp_ac = fit_probe(mem_pos_fit, clean_fit)
    Wp_co, bp_co = fit_probe(mem_pos_fit, clean_fit_c)

    Lg_ac = {c: (X @ W_ac[c] + b_ac[c]).astype(np.float32) for c in cwes}
    Lg_co = {c: (X @ W_co[c] + b_co[c]).astype(np.float32) for c in cwes}
    Lp_ac = (X @ Wp_ac + bp_ac).astype(np.float32)
    Lp_co = (X @ Wp_co + bp_co).astype(np.float32)

    # ---- negative pools (test, code-only) ----
    neg_all = toks(clean_test, code_only=True)
    neg_c = toks([e for e in clean_test if is_c_lang(e)], code_only=True)
    neg_py = toks([e for e in clean_test if not is_c_lang(e)], code_only=True)

    # honest "own" recipe (matches transfer_allclean exactly): positives = ALL code
    # tokens of CWE-Y vuln test examples, labelled by annotated y_tok (so each vuln
    # example's OWN non-vuln code tokens are negatives); + the regime's clean-neg pool.
    # Held identical across regimes — only the appended clean-negative pool changes.
    def pos_code(c):
        return toks(pos_test_of[c], code_only=True)

    def neg_matchedpatch(c):
        # negatives = the CWE's OWN paired patched safe-half. The persisted split is
        # pair-integral (pairs never straddle), so v in test_set => safe in test_set;
        # assert it so a broken split can't silently leak train-side safe halves.
        sh = []
        for v in pos_test_of[c]:
            s = safe_of_vuln.get(v)
            if s is None:
                continue
            assert s in test_set, f"matched-patch safe {s} of test vuln {v} not held-out"
            sh.append(s)
        return toks(sh, code_only=True)

    REGIMES = {"allclean": lambda c: neg_all, "conly": lambda c: neg_c,
               "pyonly": lambda c: neg_py, "matchedpatch": neg_matchedpatch}

    def eval_cell(logit, c, neg):
        pt = pos_code(c)
        if pt.size == 0 or neg.size == 0:
            return float("nan"), float("nan"), 0, 0
        ev = np.concatenate([pt, neg])
        lab = y_tok[ev]
        if len(np.unique(lab)) < 2:
            return float("nan"), float("nan"), int((y_tok[pt] == 1).sum()), int(neg.size)
        a = auc_or_nan(lab, logit[ev])
        lnull = auc_or_nan(lab, is_c_tok[ev].astype(np.float32))   # language indicator AUC
        return a, lnull, int((y_tok[pt] == 1).sum()), int(neg.size)

    # diagonal under every regime, both probe sets, with language-null
    def diagonal(Lg, Lp, tag):
        d = {}
        for c in cwes:
            row = {"family": "inj" if c in INJ else "mem",
                   "n_test_pos": n_test_pos[c], "trust": n_test_pos[c] >= MIN_TRUST_POS}
            for rn, nf in REGIMES.items():
                a, lnull, npos, nneg = eval_cell(Lg[c], c, nf(c))
                row[rn] = {"auc": a, "lang_null": lnull, "n_pos_tok": npos, "n_neg_tok": nneg}
            d[c] = row
        # pooled memory probe diagonal (eval on each mem CWE)
        pooled = {}
        for c in cwes:
            if c in INJ:
                continue
            pr = {}
            for rn, nf in REGIMES.items():
                a, lnull, npos, nneg = eval_cell(Lp, c, nf(c))
                pr[rn] = {"auc": a, "lang_null": lnull, "n_pos_tok": npos, "n_neg_tok": nneg}
            pooled[c] = pr
        return {"per_cwe": d, "pooled_memory": pooled, "probe_set": tag}

    # bootstrap CI on the diagonal cell (examples resampled), per regime, allclean-trained probes
    def diag_ci(Lg, c, regime):
        pe = pos_test_of[c]
        cl = clean_test
        cl_c = [e for e in clean_test if is_c_lang(e)]
        cl_py = [e for e in clean_test if not is_c_lang(e)]
        if not pe:
            return [float("nan"), float("nan"), 0]
        boots = []
        for _ in range(N_BOOT):
            ps = [pe[j] for j in rng.choice(len(pe), len(pe), replace=True)]
            pt = toks(ps, code_only=True)
            if regime == "matchedpatch":
                sh = [safe_of_vuln[v] for v in ps if v in safe_of_vuln]
                ng = toks(sh, code_only=True)
            else:
                pool = {"allclean": cl, "conly": cl_c, "pyonly": cl_py}[regime]
                if not pool:
                    return [float("nan"), float("nan"), 0]
                cs = [pool[j] for j in rng.choice(len(pool), len(pool), replace=True)]
                ng = toks(cs, code_only=True)
            if pt.size == 0 or ng.size == 0:
                continue
            ev = np.concatenate([pt, ng])
            lab = y_tok[ev]
            if len(np.unique(lab)) > 1:
                boots.append(auc_or_nan(lab, Lg[c][ev]))
        if not boots:
            return [float("nan"), float("nan"), 0]
        return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), len(boots)]

    if not args.cv_only:
        diag_ac = diagonal(Lg_ac, Lp_ac, "allclean_trained")
        diag_co = diagonal(Lg_co, Lp_co, "conly_trained")

        # CIs for the decisive cells: allclean-trained probes, conly & matchedpatch & allclean regimes
        cis = {}
        for c in cwes:
            cis[c] = {rn: diag_ci(Lg_ac, c, rn) for rn in ("allclean", "conly", "pyonly", "matchedpatch")}

        # reproduction gate JSON (allclean diagonal only -> compare to saved target)
        gate = {c: {"auc": diag_ac["per_cwe"][c]["allclean"]["auc"],
                    "n_test_pos": n_test_pos[c]} for c in cwes}
        (out / "repro_gate.json").write_text(json.dumps(
            {"model": args.model, "layer": L, "diagonal_allclean": gate}, indent=2,
            default=lambda x: None if x != x else x))

        res = dict(
            model=args.model, layer=L, metric="tokens_code_auc",
            cwes=cwes, inj=sorted(INJ), mem_pool=MEM_POOL, min_trust_pos=MIN_TRUST_POS,
            n_test_pos=n_test_pos,
            n_neg_tok={"allclean": int(neg_all.size), "conly": int(neg_c.size),
                       "pyonly": int(neg_py.size)},
            allclean_trained=diag_ac, conly_trained=diag_co,
            diag_ci_allclean_trained=cis,
            note="positives identical across regimes (annotated token_labels==1, code-only, "
                 "CWE-Y vuln test); only the negative pool changes. lang_null = ROC-AUC of a "
                 "pure C-vs-Python indicator on the SAME (pos,neg) token set.",
        )
        (out / "deconfound.json").write_text(json.dumps(res, indent=2,
                                            default=lambda x: None if x != x else x))
        np.savez_compressed(out / "probes_dc.npz", cwes=np.array(cwes),
                            W_ac=np.stack([W_ac[c] for c in cwes]),
                            b_ac=np.array([b_ac[c] for c in cwes], np.float32),
                            W_co=np.stack([W_co[c] for c in cwes]),
                            b_co=np.array([b_co[c] for c in cwes], np.float32),
                            Wp_ac=Wp_ac, bp_ac=np.float32(bp_ac),
                            Wp_co=Wp_co, bp_co=np.float32(bp_co), layer=np.int32(L))
        print(f"[dc] {args.model} diagonal (allclean-trained), auc | lang_null:", file=sys.stderr)
        for c in cwes:
            r = diag_ac["per_cwe"][c]
            print(f"  {c} {r['family']} nte={r['n_test_pos']}: "
                  f"all={r['allclean']['auc']:.3f}/{r['allclean']['lang_null']:.3f} "
                  f"conly={r['conly']['auc']:.3f}/{r['conly']['lang_null']:.3f} "
                  f"mp={r['matchedpatch']['auc']:.3f}/{r['matchedpatch']['lang_null']:.3f}",
                  file=sys.stderr)

    # ---- CV: grouped 5-fold x 3-seed for trusted CWEs, allclean & conly regimes (resumable) ----
    if args.do_cv or args.cv_only:
        run_cv(out, args, X, y_tok, is_code, is_c_tok, eids, rows, n_rows, tok_by_eid,
               cwes, cwe_of, is_clean, clean_fit, clean_fit_c, fit_eids, train_set, test_set,
               pair_group_key, device)
    print(f"[dc] {args.model} DONE", file=sys.stderr)


def run_cv(out, args, X, y_tok, is_code, is_c_tok, eids, rows, n_rows, tok_by_eid,
           cwes, cwe_of, is_clean, clean_fit, clean_fit_c, fit_eids, train_set, test_set,
           pair_group_key, device):
    """Grouped (pair-level) CV over the FULL labelled set per CWE, both regimes.
    For each (regime, cwe, fold, seed): split that CWE's pairs + clean pairs into
    5 grouped folds; train on 4/5, eval diagonal token-AUC on held-out 1/5
    (positives = held CWE vuln, negatives = held clean of the regime's language).
    Writes one JSON per cell (skip-if-exists) so a wall-hit just resumes."""
    from src.training.train_probe_spanmax import train_one_layer

    def toks(eid_list, code_only=False):
        if not eid_list:
            return np.array([], np.int64)
        t = np.concatenate([tok_by_eid[e] for e in eid_list])
        return t[is_code[t]] if code_only else t

    # use the union of train+test labelled examples for CV (more pairs -> tighter CV)
    all_eids = sorted(train_set | test_set)
    grp_of = {e: pair_group_key(rows[e]) for e in all_eids}
    clean_all = [e for e in all_eids if is_clean[e]]
    clean_all_c = [e for e in clean_all if (rows[e].get("lang") or "").lower() in ("c", "cpp", "c++")]

    cv_cwes = [c for c in cwes if c in TRUSTED_CV]
    for regime in ("allclean", "conly"):
        clean_pool = clean_all if regime == "allclean" else clean_all_c
        for c in cv_cwes:
            pos_all = [e for e in all_eids if cwe_of[e] == c]
            # ONE fold assignment over the UNION of pos+clean groups, then filter — so
            # if a group ever holds both a positive and a clean example it lands wholly
            # in one fold (no pair straddling train/test). (codex review CRITICAL.)
            all_groups = sorted(set(grp_of[e] for e in pos_all) |
                                set(grp_of[e] for e in clean_pool))
            for seed in CV_SEEDS:
                cell_path = out / "cv" / f"{regime}_{c}_{seed}.json"
                if cell_path.exists():
                    continue
                rng = np.random.default_rng(1000 * seed + 7)
                g = list(all_groups); rng.shuffle(g)
                folds = [set(f.tolist()) for f in np.array_split(g, CV_FOLDS)]
                fold_aucs, fold_lnull = [], []
                for k in range(CV_FOLDS):
                    held = folds[k]
                    tr_pos = [e for e in pos_all if grp_of[e] not in held]
                    te_pos = [e for e in pos_all if grp_of[e] in held]
                    tr_neg = [e for e in clean_pool if grp_of[e] not in held]
                    te_neg = [e for e in clean_pool if grp_of[e] in held]
                    ft = toks(tr_pos + tr_neg)
                    if ft.size == 0:
                        continue
                    r = train_one_layer(X[ft], y_tok[ft], eids[ft], epochs=args.epochs,
                                        device=device, verbose=False)
                    w = np.asarray(r["w"], np.float32); b = float(r["b"])
                    lg = X @ w + b
                    pt = toks(te_pos, code_only=True)
                    ng = toks(te_neg, code_only=True)
                    if pt.size == 0 or ng.size == 0:
                        continue
                    ev = np.concatenate([pt, ng])
                    lab = y_tok[ev]
                    if len(np.unique(lab)) < 2:
                        continue
                    fold_aucs.append(auc_or_nan(lab, lg[ev]))
                    fold_lnull.append(auc_or_nan(lab, is_c_tok[ev].astype(np.float32)))
                cell = {"model": args.model, "regime": regime, "cwe": c, "seed": seed,
                        "fold_aucs": fold_aucs, "fold_lang_null": fold_lnull,
                        "mean_auc": float(np.mean(fold_aucs)) if fold_aucs else float("nan"),
                        "std_auc": float(np.std(fold_aucs)) if fold_aucs else float("nan"),
                        "mean_lang_null": float(np.mean(fold_lnull)) if fold_lnull else float("nan"),
                        "n_pos_examples": len(pos_all)}
                cell_path.write_text(json.dumps(cell, indent=2,
                                     default=lambda x: None if x != x else x))
                print(f"[cv] {args.model} {regime} {c} seed{seed}: "
                      f"mean_auc={cell['mean_auc']:.3f} lnull={cell['mean_lang_null']:.3f} "
                      f"({len(fold_aucs)} folds)", file=sys.stderr)


if __name__ == "__main__":
    main()
