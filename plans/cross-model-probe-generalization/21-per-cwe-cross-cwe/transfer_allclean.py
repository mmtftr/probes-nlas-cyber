# [ai-generated]
"""exp-21 (corrected) — cross-CWE transfer matrix on tokens_code_auc, EXP-10 RECIPE.

The first re-exec rescored exp-21's saved probes, which were trained
one-vs-MATCHED-PATCH (subtractive, difflib spans) — a regime that handicaps
memory CWEs (additive fixes get an empty difflib span; near-zero negative
diversity). That under-counts memory and does NOT match exp-10, where per-CWE
memory probes trained vs ALL CLEAN reach tokens_code_auc ~0.73/0.77.

This script trains per-CWE probes the EXP-10 way and builds the full cross-CWE
transfer matrix:
  * positives = annotated `token_labels` spans (parse_spans, label==1) — the same
    label exp-10/extractor use (populated for additive memory fixes too), NOT
    difflib subtractive spans;
  * train CWE-X probe on {CWE-X vuln examples} ∪ {ALL cwe==null clean}, FULL SVEN,
    on ALL tokens of those examples (span-max pools per example; no is_code gate at
    train — matches exp-10 `_fit_probe`);
  * 15% group-aware VAL carve of TRAIN excluded from every fit (VAL_SEED=42), exact
    parity with exp-10 / train_all_layers;
  * Cell[train=X][test=Y] = tokens_code_auc of probe-X on {CWE-Y test positives} ∪
    {ALL clean test} — live-code tokens only (code_only_mask), label = annotated
    span. Shared clean-test negative pool across columns. Diagonal X==X MUST
    reproduce exp-10's specialized number (089~0.98, 125~0.73, 416~0.77).

Reuses exp-21's KEPT full-SVEN activations (token_activations_layer25.npz, X +
example_ids) + offsets.npz — NO re-extraction. Outputs --out/transfer_allclean.json.
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
VAL_FRAC, VAL_SEED = 0.15, 42
MIN_TRUST_POS = 10
N_BOOT = 1000


def auc_or_nan(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True, help="token_activations dir (X npz + offsets.npz)")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--min-train-pos", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    acts = Path(args.acts); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    L = args.layer
    rng = np.random.default_rng(args.seed)

    assert Path(args.split).exists(), f"split file missing: {args.split} (would mint a NEW split)"
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

    # per-token annotated label (exp-10 positives) + code mask (eval), aligned to X
    y_tok = np.zeros(len(eids), np.int8)
    is_code = np.zeros(len(eids), bool)
    cur = 0
    for e in range(n_rows):
        n = int((eids == e).sum())
        o = offsets_per_row[e]
        if o.shape[0] != n:
            raise SystemExit(f"offset/token mismatch eid {e}: {o.shape[0]} vs {n}")
        # tokens are concatenated contiguously in eid order — guard against silent
        # misalignment of labels/mask onto X (the cur-advancing assignment below
        # assumes this; verified by the diagonal reproducing exp-10).
        assert (eids[cur:cur + n] == e).all(), f"non-contiguous eid block at {e}"
        off_list = [(int(s), int(t)) for s, t in o]
        tok_spans = char_spans_to_token_spans(parse_spans(rows[e]), off_list)
        lab, _ = token_labels_array(n, tok_spans)
        y_tok[cur:cur + n] = lab
        is_code[cur:cur + n] = code_only_mask(rows[e].get("code", ""),
                                              rows[e].get("lang", "") or "", o).astype(bool)
        cur += n
    if cur != len(eids):
        raise SystemExit(f"consumed {cur} of {len(eids)}")

    cwe_of = {e: rows[e].get("cwe") for e in range(n_rows)}
    is_clean = {e: (rows[e].get("label") == 0 and not rows[e].get("cwe")) for e in range(n_rows)}

    # 15% group-aware VAL carve of TRAIN (exp-10 parity) — excluded from every fit
    tr_eid_grp = {e: pair_group_key(rows[e]) for e in train_set}
    tr_groups = sorted(set(tr_eid_grp.values()))
    vr = np.random.default_rng(VAL_SEED)
    vr.shuffle(tr_groups)
    n_val = max(1, int(round(VAL_FRAC * len(tr_groups))))
    val_groups = set(tr_groups[:n_val])
    val_eids = {e for e, g in tr_eid_grp.items() if g in val_groups}
    fit_eids = train_set - val_eids

    clean_fit = [e for e in fit_eids if is_clean[e]]
    clean_test = [e for e in test_set if is_clean[e]]

    train_pos = Counter(cwe_of[e] for e in fit_eids if cwe_of[e])
    cwes = sorted([c for c, n in train_pos.items() if n >= args.min_train_pos],
                  key=lambda c: (c not in INJ, c))
    print(f"[allclean] {args.model} CWEs (train_pos>={args.min_train_pos}): "
          f"{[(c, train_pos[c]) for c in cwes]}", file=sys.stderr)

    # token index lists per eid (fast masking + grouped bootstrap)
    tok_by_eid = defaultdict(list)
    for i, e in enumerate(eids):
        tok_by_eid[int(e)].append(i)
    tok_by_eid = {e: np.asarray(ix, np.int64) for e, ix in tok_by_eid.items()}

    def toks(eid_list, code_only=False):
        if not eid_list:
            return np.array([], np.int64)
        t = np.concatenate([tok_by_eid[e] for e in eid_list])
        return t[is_code[t]] if code_only else t

    # ---- train per-CWE probes (CWE-X vuln ∪ all clean), exp-10 recipe ----
    W, b = {}, {}
    n_train_pos = {}
    for c in cwes:
        pos_fit = [e for e in fit_eids if cwe_of[e] == c]
        n_train_pos[c] = len(pos_fit)
        fit_tok = toks(pos_fit + clean_fit)                 # ALL tokens (no is_code gate) — exp-10
        r = train_one_layer(X[fit_tok], y_tok[fit_tok], eids[fit_tok],
                            epochs=args.epochs, device=device, verbose=False)
        W[c] = np.asarray(r["w"], np.float32); b[c] = float(r["b"])
        print(f"[allclean] {args.model} trained {c}: n_pos={len(pos_fit)} "
              f"|w|={np.linalg.norm(W[c]):.2f}", file=sys.stderr)

    # precompute each probe's per-token logit ONCE (cells + both bootstraps just
    # index into these — avoids re-matmul per rep, which made block-bootstrap blow
    # past walltime). Lg[c] aligned to the global token axis.
    Lg = {c: (X @ W[c] + b[c]).astype(np.float32) for c in cwes}

    # shared clean-test negative tokens (code-only) + per-CWE test positives
    neg_test_tok = toks(clean_test, code_only=True)
    pos_test_eids = {c: [e for e in test_set if cwe_of[e] == c] for c in cwes}
    n_test_pos = {c: len(pos_test_eids[c]) for c in cwes}
    npos_tok = {}
    for c in cwes:
        pt = toks(pos_test_eids[c], code_only=True)
        npos_tok[c] = int((y_tok[pt] == 1).sum())

    def cell(c_x, c_y):
        pt = toks(pos_test_eids[c_y], code_only=True)
        if pt.size == 0:
            return float("nan")
        ev = np.concatenate([pt, neg_test_tok])
        lab = y_tok[ev]
        return auc_or_nan(lab, Lg[c_x][ev])

    M = {cx: {cy: cell(cx, cy) for cy in cwes} for cx in cwes}

    def block_means(MM):
        g = {"inj": [c for c in cwes if c in INJ], "mem": [c for c in cwes if c not in INJ]}
        o = {}
        for gtr, ctr in g.items():
            for gte, cte in g.items():
                num = den = 0.0
                for cx in ctr:
                    for cy in cte:
                        a = MM[cx][cy]; w = npos_tok.get(cy, 0)
                        if a == a and w:
                            num += a * w; den += w
                o[f"{gtr}->{gte}"] = (num / den) if den else float("nan")
        return o

    # diagonal bootstrap CIs (resample test pos + clean pools separately)
    diag = {}
    for c in cwes:
        pe = pos_test_eids[c]; a = M[c][c]
        boots = []
        if pe and a == a:
            for _ in range(N_BOOT):
                ps = [pe[j] for j in rng.choice(len(pe), len(pe), replace=True)]
                cs = [clean_test[j] for j in rng.choice(len(clean_test), len(clean_test), replace=True)]
                ev = np.concatenate([toks(ps, code_only=True), toks(cs, code_only=True)])
                lab = y_tok[ev]
                if len(np.unique(lab)) > 1:
                    boots.append(auc_or_nan(lab, Lg[c][ev]))
        lo, hi = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) if boots else (float("nan"),) * 2
        diag[c] = {"auc": a, "ci": [lo, hi], "n_train_pos": n_train_pos[c],
                   "n_test_pos": n_test_pos[c], "trust": n_test_pos[c] >= MIN_TRUST_POS,
                   "boot_reps": len(boots)}

    # ---- trusted block means (n_test_pos>=MIN_TRUST) + bootstrap CIs ----
    # exclude low-n columns (190/787) so block transfer is not driven by noise; this
    # also sidesteps the CWE-190 family ambiguity (it is n<10 -> excluded either way).
    trusted = [c for c in cwes if n_test_pos[c] >= MIN_TRUST_POS]
    inj_t = [c for c in trusted if c in INJ]; mem_t = [c for c in trusted if c not in INJ]

    def block4(get_auc, weight):
        g = {"inj": inj_t, "mem": mem_t}
        o = {}
        for gtr, ctr in g.items():
            for gte, cte in g.items():
                for tag, skip_self in (("", False), ("_offdiag", True)):
                    num = den = 0.0
                    for cx in ctr:
                        for cy in cte:
                            if skip_self and cx == cy:
                                continue
                            a = get_auc(cx, cy); w = weight[cy]
                            if a == a and w:
                                num += a * w; den += w
                    o[f"{gtr}->{gte}{tag}"] = (num / den) if den else float("nan")
        return o

    blk_trusted = block4(lambda cx, cy: M[cx][cy], npos_tok)
    # bootstrap: resample each trusted column's test positives + the clean pool
    boot = {k: [] for k in blk_trusted}
    for _ in range(N_BOOT // 2):
        cs = [clean_test[j] for j in rng.choice(len(clean_test), len(clean_test), replace=True)]
        negb = toks(cs, code_only=True)
        col_tok, col_w = {}, {}
        for cy in trusted:
            pe = pos_test_eids[cy]
            ps = [pe[j] for j in rng.choice(len(pe), len(pe), replace=True)]
            col_tok[cy] = toks(ps, code_only=True); col_w[cy] = int((y_tok[col_tok[cy]] == 1).sum())

        def bcell(cx, cy):
            ev = np.concatenate([col_tok[cy], negb]); lab = y_tok[ev]
            return auc_or_nan(lab, Lg[cx][ev])
        bb = block4(bcell, col_w)
        for k, v in bb.items():
            if v == v:
                boot[k].append(v)
    blk_ci = {k: ([float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), len(v)]
                  if v else [float("nan")] * 3) for k, v in boot.items()}

    res = dict(
        model=args.model, layer=L, metric="tokens_code_auc",
        recipe="exp-10: annotated token_labels positives; train CWE-X vuln ∪ ALL clean (full SVEN, all tokens); honest code-only eval; shared clean-test negatives",
        cwes=cwes, inj=sorted(INJ), min_trust_pos=MIN_TRUST_POS,
        trusted_cwes=trusted, trusted_inj=inj_t, trusted_mem=mem_t,
        n_train_pos=n_train_pos, n_test_pos=n_test_pos, n_pos_tokens=npos_tok,
        n_clean_test_examples=len(clean_test), n_neg_test_tokens=int(neg_test_tok.size),
        auc=M, block_auc=block_means(M),
        block_auc_trusted=blk_trusted, block_auc_trusted_ci=blk_ci,
        diagonal=diag,
    )
    np.savez_compressed(out / "probes_allclean.npz", cwes=np.array(cwes),
                        W=np.stack([W[c] for c in cwes]),
                        b=np.array([b[c] for c in cwes], np.float32), layer=np.int32(L))
    (out / "transfer_allclean.json").write_text(json.dumps(res, indent=2,
                                               default=lambda x: None if x != x else x))
    print(f"[allclean] {args.model} diagonal (token-AUC, exp-10 recipe):", file=sys.stderr)
    for c in cwes:
        d = diag[c]; fam = "INJ" if c in INJ else "mem"
        print(f"  {c} {fam}: {d['auc']:.3f} CI{[round(x,3) for x in d['ci']]} "
              f"trust={d['trust']} (ntr={d['n_train_pos']}, nte={d['n_test_pos']})", file=sys.stderr)
    print(f"[allclean] blocks = { {k: round(v,3) for k,v in res['block_auc'].items()} }", file=sys.stderr)


if __name__ == "__main__":
    main()
