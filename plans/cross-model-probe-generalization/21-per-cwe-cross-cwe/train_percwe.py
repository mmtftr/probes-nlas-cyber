# [ai-generated]
"""exp-21 — per-CWE probes + cross-CWE detection matrix on SVEN-subtractive.

Trains one span-max linear probe per CWE (one-vs-paired-safe, honest tight∩is_code
labels) at a model's operating layer, then evaluates every probe on every CWE's
held-out test pairs → a CWE×CWE transfer matrix. Imbalance is handled by training
each probe at BOTH its natural size and a balanced cap (--balanced-n pairs).

Reuses the on-scratch repo libraries (extractor output, span-max trainer,
tree-sitter code mask) — no new deps. Pure eval functions (`score_examples`,
`pair_matrix`) are importable for local validation with a stand-in scorer.

Outputs (--out):
  matrix.json            — pairacc/auc/det matrices (natural+balanced), n_train/n_test, thr
  probes_percwe.npz      — W_natural[cwe], b_natural, W_balanced, b_balanced (persisted)
  logits_percwe.npz      — per-token logit of every natural probe over ALL tokens
                           (KEEP on scratch for follow-up; not downloaded)
"""
from __future__ import annotations
import argparse, difflib, json, os, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

REPO = Path(os.environ.get("REPO", Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(REPO))
from sklearn.metrics import roc_auc_score, precision_recall_curve   # noqa: E402
from src.training.train_probe_spanmax import train_one_layer        # noqa: E402
from src.remotes.train_eval import load_or_make_split      # noqa: E402
from src.eval.code_mask import code_only_mask, live_code_char_ranges  # noqa: E402

INJ = {"CWE-089", "CWE-078", "CWE-022", "CWE-079"}


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


def f1max_threshold(y, s):
    if len(np.unique(y)) < 2:
        return float("inf"), 0.0
    p, r, t = precision_recall_curve(y, s)
    f1 = np.where((p + r) > 0, 2 * p * r / (p + r + 1e-12), 0.0)
    k = int(np.argmax(f1[:-1])) if len(t) else 0
    return float(t[k]), float(f1[k])


# ── pure eval helpers (testable locally with any scorer) ─────────────────────
def example_maxcode(logit, eids, is_code, target_eids):
    """For each eid in target_eids, max logit over its is_code tokens (or -inf)."""
    out = {}
    for e in target_eids:
        m = (eids == e) & is_code
        out[e] = float(logit[m].max()) if m.any() else float("-inf")
    return out


def pair_matrix(score_fn, sub_pairs, cwe_of, test_set, cwes):
    """score_fn(eid)->float (example max-code logit under some probe).
    Returns (pairacc[c'], auc[c'], n[c']) over c' test pairs."""
    pairacc, auc, npairs = {}, {}, {}
    for cp in cwes:
        tp = [(v, s) for (v, s) in sub_pairs if cwe_of[v] == cp and v in test_set]
        if not tp:
            pairacc[cp] = float("nan"); auc[cp] = float("nan"); npairs[cp] = 0
            continue
        sv = np.array([score_fn(v) for v, s in tp])
        ss = np.array([score_fn(s) for v, s in tp])
        pairacc[cp] = float(np.mean(sv > ss))
        labels = np.r_[np.ones(len(tp)), np.zeros(len(tp))]
        scores = np.r_[sv, ss]
        auc[cp] = (float(roc_auc_score(labels, scores))
                   if len(np.unique(labels)) > 1 else float("nan"))
        npairs[cp] = len(tp)
    return pairacc, auc, npairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--acts", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--balanced-n", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    acts = Path(args.acts); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    L = args.layer

    rows, train_eids, test_eids = load_or_make_split(Path(args.dataset), Path(args.split))
    train_set, test_set = set(int(e) for e in train_eids), set(int(e) for e in test_eids)

    offs = np.load(acts / "offsets.npz")
    n_rows = len(offs.files)
    offsets_per_row = [offs[f"offsets_row_{i:04d}"] for i in range(n_rows)]

    cand = [p for p in acts.glob("token_activations_layer*.npz")
            if int(p.stem.replace("token_activations_layer", "")) == L]
    if not cand:
        raise SystemExit(f"no token_activations for layer {L} under {acts}")
    npz = np.load(cand[0])
    X = npz["X"].astype(np.float32)
    eids = npz["example_ids"].astype(np.int32)

    # align flat token axis -> char spans + is_code (mirror dump_logits.py)
    char_s = np.empty(len(eids), np.int32); char_e = np.empty(len(eids), np.int32)
    is_code = np.zeros(len(eids), bool)
    cur = 0
    for e in range(n_rows):
        n = int((eids == e).sum())
        o = offsets_per_row[e]
        if o.shape[0] != n:
            raise SystemExit(f"offset/token mismatch eid {e}: {o.shape[0]} vs {n}")
        char_s[cur:cur + n] = o[:, 0]; char_e[cur:cur + n] = o[:, 1]
        m = code_only_mask(rows[e].get("code", ""), rows[e].get("lang", "") or "", o)
        is_code[cur:cur + n] = m.astype(bool)
        cur += n
    if cur != len(eids):
        raise SystemExit(f"consumed {cur} of {len(eids)}")

    # pairs + subtractive membership + tight∩is_code labels
    grp = defaultdict(list)
    for eid, r in enumerate(rows):
        grp[(r.get("_file_name"), r.get("_func_name"))].append(eid)
    sub_pairs = []
    for es in grp.values():
        vs = [e for e in es if rows[e]["label"] == 1]
        ss = [e for e in es if rows[e]["label"] == 0]
        for i in range(min(len(vs), len(ss))):
            v, s = vs[i], ss[i]
            sp = tight_spans(rows[v]["code"], rows[s]["code"])
            if overlaps_live(sp, live_code_char_ranges(rows[v]["code"], rows[v].get("lang") or "")):
                sub_pairs.append((v, s))
    tspans = {v: tight_spans(rows[v]["code"], rows[s]["code"]) for v, s in sub_pairs}
    cwe_of = {v: rows[v].get("cwe") for v, s in sub_pairs}

    y_tok = np.zeros(len(eids), int)
    for v, sp in tspans.items():
        if not sp:
            continue
        idx = np.where(eids == v)[0]
        s_, e_ = char_s[idx], char_e[idx]
        ov = np.zeros(len(idx), bool)
        for i1, i2 in sp:
            ov |= (s_ < i2) & (e_ > i1)
        y_tok[idx] = (ov & is_code[idx]).astype(int)

    cwes = sorted(set(cwe_of.values()), key=lambda c: (c not in INJ, c))  # injection first

    def train_for(pairs_list):
        ve = [v for v, s in pairs_list]; se = [s for v, s in pairs_list]
        keep = np.isin(eids, ve + se) & is_code
        idx = np.where(keep)[0]
        r = train_one_layer(X[idx], y_tok[idx], eids[idx],
                            epochs=args.epochs, device=device, verbose=False)
        return np.asarray(r["w"], np.float32), float(r["b"])

    rng = np.random.default_rng(args.seed)
    W_nat, b_nat, W_bal, b_bal, thr_nat = {}, {}, {}, {}, {}
    n_train_pairs, n_train_bal = {}, {}
    for c in cwes:
        tp = [(v, s) for (v, s) in sub_pairs if cwe_of[v] == c and v in train_set]
        n_train_pairs[c] = len(tp)
        w, b = train_for(tp); W_nat[c], b_nat[c] = w, b
        # natural threshold (F1-max on this CWE's train code tokens)
        ve = [v for v, s in tp]; se = [s for v, s in tp]
        idx = np.where(np.isin(eids, ve + se) & is_code)[0]
        lg = X[idx] @ w + b
        thr_nat[c], _ = f1max_threshold(y_tok[idx], lg)
        # balanced
        pick = tp if len(tp) <= args.balanced_n else [tp[i] for i in
               rng.choice(len(tp), args.balanced_n, replace=False)]
        n_train_bal[c] = len(pick)
        wb, bb = train_for(pick); W_bal[c], b_bal[c] = wb, bb
        print(f"[percwe] {args.model} {c}: n_train={len(tp)} bal={len(pick)} "
              f"|w|={np.linalg.norm(w):.2f} thr={thr_nat[c]:.2f}", file=sys.stderr)

    # cross-CWE matrices (natural + balanced) over held-out test pairs
    test_eids_all = sorted({v for v, s in sub_pairs if v in test_set}
                           | {s for v, s in sub_pairs if v in test_set})

    def score_cache(W, b):
        """precompute eid -> maxcode logit for every test example, per probe c."""
        cache = {}
        for c in cwes:
            lg = X @ W[c] + b[c]
            cache[c] = example_maxcode(lg, eids, is_code, test_eids_all)
        return cache

    def build(W, b):
        cache = score_cache(W, b)
        pa, au, nn = {}, {}, {}
        det = {}
        for c in cwes:
            sc = cache[c]
            pacc, auc, npr = pair_matrix(lambda e: sc[e], sub_pairs, cwe_of, test_set, cwes)
            pa[c], au[c], nn[c] = pacc, auc, npr
            # detection rate at thr_c: frac of c' test vuln with maxcode >= thr
            dd = {}
            for cp in cwes:
                tv = [v for (v, s) in sub_pairs if cwe_of[v] == cp and v in test_set]
                dd[cp] = (float(np.mean([sc[v] >= thr_nat[c] for v in tv]))
                          if tv else float("nan"))
            det[c] = dd
        return pa, au, nn, det

    pa_n, au_n, nn_n, det_n = build(W_nat, b_nat)
    pa_b, au_b, nn_b, _ = build(W_bal, b_bal)

    def block(mat):
        """2x2 injection/memory block means, pooled (weighted by n_test pairs)."""
        groups = {"inj": [c for c in cwes if c in INJ], "mem": [c for c in cwes if c not in INJ]}
        b = {}
        for gtr, ctr in groups.items():
            for gte, cte in groups.items():
                num = den = 0.0
                for c in ctr:
                    for cp in cte:
                        v = mat[c][cp]; n = nn_n[c][cp]
                        if v == v and n:           # not nan
                            num += v * n; den += n
                b[f"{gtr}->{gte}"] = (num / den) if den else float("nan")
        return b

    res = dict(
        model=args.model, layer=L,
        cwes=cwes, inj=sorted(INJ),
        n_train_pairs=n_train_pairs, n_train_balanced=n_train_bal,
        n_test_pairs={c: nn_n[cwes[0]].get(c, 0) for c in cwes},
        thr_natural={c: (None if thr_nat[c] == float("inf") else round(thr_nat[c], 4)) for c in cwes},
        pairacc_natural={c: pa_n[c] for c in cwes},
        auc_natural={c: au_n[c] for c in cwes},
        pairacc_balanced={c: pa_b[c] for c in cwes},
        auc_balanced={c: au_b[c] for c in cwes},
        detrate_natural={c: det_n[c] for c in cwes},
        block_pairacc_natural=block(pa_n),
        block_pairacc_balanced=block(pa_b),
        block_auc_natural=block(au_n),
    )
    (out / "matrix.json").write_text(json.dumps(res, indent=2, default=lambda x: None if x != x else x))

    # persist probes + per-token logits of natural probes (KEEP on scratch)
    np.savez_compressed(out / "probes_percwe.npz",
                        cwes=np.array(cwes),
                        W_natural=np.stack([W_nat[c] for c in cwes]),
                        b_natural=np.array([b_nat[c] for c in cwes], np.float32),
                        W_balanced=np.stack([W_bal[c] for c in cwes]),
                        b_balanced=np.array([b_bal[c] for c in cwes], np.float32),
                        layer=np.int32(L))
    np.savez_compressed(out / "logits_percwe.npz",
                        cwes=np.array(cwes), example_id=eids,
                        char_start=char_s, char_end=char_e, is_code=is_code,
                        y_tok=y_tok.astype(np.int8),
                        logit=np.stack([(X @ W_nat[c] + b_nat[c]).astype(np.float32) for c in cwes]))
    print(f"[percwe] {args.model}: blocks(nat) = {res['block_pairacc_natural']}", file=sys.stderr)
    print(f"[percwe] {args.model}: blocks(bal) = {res['block_pairacc_balanced']}", file=sys.stderr)


if __name__ == "__main__":
    main()
