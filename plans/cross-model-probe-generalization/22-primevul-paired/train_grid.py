# [ai-generated]
"""Train the subtractive-regime config grid from CACHED activations (no re-extract
beyond the one-time acts dump) and cross-evaluate every probe on both subsets.

Grid per (model, layer):
  subset      ∈ {base (all 1430), subtractive (956, additive pairs dropped)}
  granularity ∈ {line (whole-line evidence), token (tight difflib changed chars)}
  negatives   ∈ {Y = "none" (non-code tokens are negatives),
                 X = "code_only" (non-code negatives dropped)}
  => 2×2×2 = 8 probes.

Standing rule (both X and Y): a token is POSITIVE only if (granularity-positive
AND is_code). Non-code tokens are never positive. Loss is left UNCHANGED
(span-max); the subtractive subset guarantees every kept vuln example has a
positive token, so no example-level-positivity hack is needed. SVEN-base keeps
old logic but here we also retrain it under the grid for cross-comparison.

Common honest EVAL label (so all 8 probes are compared identically):
  truth = (tight-token positive AND is_code); AUC over code tokens (code-only),
  on each of {subtractive-test, base-test, additive-test} held-out sets.

Self-test (no GPU/acts): `python train_grid.py --selftest <logits_layer.npz>`
validates the label computation against an exp-16 dump (which carries
char_start/char_end/is_code/y/example_id).
"""
from __future__ import annotations
import argparse, difflib, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))


# ---------------------------------------------------------------------------
# labels (pure, unit-testable)
# ---------------------------------------------------------------------------
def _line_spans(before: str, after: str, line_cap: int = 4000):
    """Fast line-level delete/replace spans -> `before` char ranges (huge-func
    fallback; char-level difflib is O(n*m)). Mirrors exp-22 build_primevul.py."""
    a = before.splitlines(keepends=True)
    b = after.splitlines(keepends=True)
    if len(a) > line_cap or len(b) > line_cap:
        return [(0, len(before))] if before else []
    starts, off = [], 0
    for ln in a:
        starts.append(off); off += len(ln)
    starts.append(off)
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    return [(starts[i1], starts[i2]) for t, i1, i2, j1, j2 in sm.get_opcodes()
            if t in ("replace", "delete") and i2 > i1]


def _tight_spans(before: str, after: str, cap: int = 6000):
    # Length-guarded: PrimeVul funcs reach ~480 KB; char-level difflib would hang.
    # cap=6000 keeps ALL SVEN funcs (max 5833) char-level (matches exp-19);
    # only PrimeVul's larger funcs use the fast line-level fallback.
    if len(before) > cap or len(after) > cap:
        return _line_spans(before, after)
    sm = difflib.SequenceMatcher(a=before, b=after, autojunk=False)
    return [(i1, i2) for t, i1, i2, j1, j2 in sm.get_opcodes()
            if t in ("replace", "delete") and i2 > i1]


def _overlap_mask(cs, ce, spans):
    out = np.zeros(len(cs), dtype=bool)
    for (i1, i2) in spans:
        out |= (cs < i2) & (ce > i1)
    return out


def compute_labels(eids, char_start, char_end, is_code, ds, vuln_to_safe):
    """Return dict of per-token boolean arrays:
       line_pos, token_pos  (granularity positives, NOT yet is_code-gated).
    line_pos = token overlaps any whole-line evidence span of its example.
    token_pos = token overlaps a tight difflib(before, after) changed span."""
    n = len(eids)
    line_pos = np.zeros(n, dtype=bool)
    token_pos = np.zeros(n, dtype=bool)
    for eid in np.unique(eids):
        idx = np.where(eids == eid)[0]
        row = ds[int(eid)]
        if row.get("label") != 1:
            continue
        cs, ce = char_start[idx], char_end[idx]
        # line: whole-line evidence spans from the base dataset token_labels
        ev = (row.get("token_labels") or {}).get("evidence") or []
        if ev:
            line_pos[idx] = _overlap_mask(cs, ce, [(s, e) for s, e in ev])
        # token: tight diff vs the paired safe example
        s_eid = vuln_to_safe.get(int(eid))
        if s_eid is not None:
            spans = _tight_spans(row["code"], ds[s_eid]["code"])
            if spans:
                token_pos[idx] = _overlap_mask(cs, ce, spans)
    return {"line": line_pos, "token": token_pos}


def build_pairs(ds):
    grp = defaultdict(list)
    for eid, r in enumerate(ds):
        grp[(r.get("_file_name"), r.get("_func_name"))].append(eid)
    v2s = {}
    for eids in grp.values():
        vs = [e for e in eids if ds[e]["label"] == 1]
        ss = [e for e in eids if ds[e]["label"] == 0]
        for i in range(min(len(vs), len(ss))):
            v2s[vs[i]] = ss[i]
    return v2s


def _auc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")


# ---------------------------------------------------------------------------
# acts loading + per-token char offsets / is_code (cluster path)
# ---------------------------------------------------------------------------
def load_layer(acts_dir: Path, layer: int, ds):
    from src.eval.code_mask import code_only_mask
    npz = np.load(acts_dir / f"token_activations_layer{layer:02d}.npz")
    X = npz["X"]
    eids = npz["example_ids"].astype(np.int32)
    offs = np.load(acts_dir / "offsets.npz")
    n_rows = len(offs.files)
    cs = np.empty(len(eids), np.int32); ce = np.empty(len(eids), np.int32)
    is_code = np.zeros(len(eids), bool)
    cur = 0
    for eid in range(n_rows):
        n_tok = int((eids == eid).sum())
        o = offs[f"offsets_row_{eid:04d}"]
        if o.shape[0] != n_tok:
            raise SystemExit(f"layer {layer}: offset/token mismatch eid {eid}: {o.shape[0]} vs {n_tok}")
        cs[cur:cur + n_tok] = o[:, 0]; ce[cur:cur + n_tok] = o[:, 1]
        m = code_only_mask(ds[eid].get("code", ""), ds[eid].get("lang", "") or "", o)
        is_code[cur:cur + n_tok] = m.astype(bool)
        cur += n_tok
    if cur != len(eids):
        raise SystemExit(f"layer {layer}: consumed {cur} of {len(eids)}")
    return X, eids, cs, ce, is_code


# ---------------------------------------------------------------------------
def run(args):
    from src.training.train_probe_spanmax import train_one_layer
    from src.remotes.the cluster.train_eval import load_or_make_split
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = [json.loads(l) for l in open(args.dataset)]
    _, train_eids, test_eids = load_or_make_split(Path(args.dataset), Path(args.split))
    test_set = set(int(e) for e in test_eids)
    v2s = build_pairs(ds)
    member = json.loads(Path(args.membership).read_text())
    sub_set = set(member["kept_eids"])

    acts_dir = Path(args.acts)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    layer_files = sorted(acts_dir.glob("token_activations_layer*.npz"))
    if args.layers:
        want = {int(x) for x in args.layers.split(",") if x.strip()}
        layer_files = [f for f in layer_files
                       if int(f.stem.replace("token_activations_layer", "")) in want]

    results = []
    for f in layer_files:
        L = int(f.stem.replace("token_activations_layer", ""))
        X, eids, cs, ce, isc = load_layer(acts_dir, L, ds)
        labs = compute_labels(eids, cs, ce, isc, ds, v2s)
        in_sub = np.fromiter((int(e) in sub_set for e in eids), bool, len(eids))
        is_test = np.fromiter((int(e) in test_set for e in eids), bool, len(eids))
        # common honest eval truth: tight-token ∩ is_code
        truth = labs["token"] & isc
        # held-out pairs for example-level transfer (vuln vs its safe pair),
        # split by regime. Additive vuln have no positive token, so example-level
        # pair-ranking is the only honest transfer signal for them.
        sub_pairs = [(v, s) for v, s in v2s.items()
                     if v in test_set and s in test_set and v in sub_set]
        add_pairs = [(v, s) for v, s in v2s.items()
                     if v in test_set and s in test_set and v not in sub_set]
        for subset in ("base", "subtractive"):
            smask = np.ones(len(eids), bool) if subset == "base" else in_sub
            for gran in ("line", "token"):
                ypos = labs[gran] & isc            # is_code-gated positives
                for neg, mn in (("Y", "none"), ("X", "code_only")):
                    trmask = smask & ~is_test
                    if len(np.unique(ypos[trmask])) < 2:
                        continue
                    r = train_one_layer(
                        X[trmask], ypos[trmask].astype(np.int8), eids[trmask],
                        epochs=args.epochs, device=device, verbose=False,
                        mask_negatives=mn,
                        code_mask=isc[trmask] if mn == "code_only" else None,
                    )
                    w, b = np.asarray(r["w"], np.float32), float(r["b"])
                    logit = X @ w + b
                    prob = 1.0 / (1.0 + np.exp(-logit))
                    code = isc
                    # token-level code-only AUC (common honest truth) per test set
                    def code_auc(setmask):
                        m = setmask & code & is_test
                        return _auc(truth[m], logit[m]), int(truth[m].sum()), int(m.sum())
                    sa, sp, sn = code_auc(in_sub)
                    ba, bp, bn = code_auc(np.ones(len(eids), bool))
                    # per-example max code-prob (one vectorised pass)
                    cidx = np.where(code)[0]
                    ec, pc = eids[cidx], prob[cidx]
                    o = np.argsort(ec, kind="stable"); ec, pc = ec[o], pc[o]
                    uq, st = np.unique(ec, return_index=True)
                    mx = np.maximum.reduceat(pc, st)
                    ex_max = dict(zip(uq.tolist(), mx.tolist()))

                    def pair_stats(pairs):
                        if not pairs:
                            return float("nan"), float("nan"), 0
                        sv = np.array([ex_max.get(v, np.nan) for v, _ in pairs])
                        ss = np.array([ex_max.get(s, np.nan) for _, s in pairs])
                        ok = np.isfinite(sv) & np.isfinite(ss)
                        if ok.sum() == 0:
                            return float("nan"), float("nan"), 0
                        acc = float((sv[ok] > ss[ok]).mean())
                        ys = np.r_[np.ones(ok.sum()), np.zeros(ok.sum())]
                        auc = _auc(ys, np.r_[sv[ok], ss[ok]])
                        return acc, auc, int(ok.sum())
                    spa, spauc, spn = pair_stats(sub_pairs)   # subtractive transfer
                    apa, apauc, apn = pair_stats(add_pairs)   # additive transfer
                    rec = {
                        "layer": L, "train_subset": subset, "granularity": gran, "negatives": neg,
                        "eval_subtractive_test_code_auc": sa, "n_sub_pos": sp, "n_sub_tok": sn,
                        "eval_base_test_code_auc": ba, "n_base_pos": bp, "n_base_tok": bn,
                        "pair_acc_subtractive": spa, "pair_auc_subtractive": spauc, "n_sub_pairs": spn,
                        "pair_acc_additive": apa, "pair_auc_additive": apauc, "n_add_pairs": apn,
                        "n_train_pos": int(ypos[trmask].sum()), "n_train_tok": int(trmask.sum()),
                    }
                    results.append(rec)
                    np.savez_compressed(out / f"probe_L{L:02d}_{subset}_{gran}_{neg}.npz",
                                        w=w, b=np.float32(b), layer=np.int32(L))
                    print(f"[grid] L{L} {subset:11s} {gran:5s} {neg}  "
                          f"subTokAUC={sa:.3f} baseTokAUC={ba:.3f} "
                          f"pairAcc[sub={spa:.2f} add={apa:.2f}] "
                          f"(train_pos={rec['n_train_pos']})", file=sys.stderr)
        del X
    (out / "metrics_grid.json").write_text(json.dumps(
        {"model": args.model, "results": results}, indent=2))
    print(f"[grid] wrote {out/'metrics_grid.json'}  ({len(results)} configs)", file=sys.stderr)


def selftest(npz_path):
    """Validate label logic against an exp-16 logits dump (no acts needed)."""
    ds = [json.loads(l) for l in open(REPO / "data" / "dataset.jsonl")]
    v2s = build_pairs(ds)
    z = np.load(npz_path)
    eids, cs, ce, isc, y_old = z["example_id"], z["char_start"], z["char_end"], z["is_code"], z["y"].astype(int)
    labs = compute_labels(eids, cs, ce, isc, ds, v2s)
    line_isc = (labs["line"] & isc)
    tok_isc = (labs["token"] & isc)
    # the dumped y is whole-line evidence; on code tokens it should match our line∩is_code
    agree = int((line_isc == (y_old.astype(bool) & isc)).sum())
    print(f"selftest {Path(npz_path).name}:")
    print(f"  tokens={len(eids)} is_code={int(isc.sum())}")
    print(f"  old-y positives={int(y_old.sum())}  line∩code positives={int(line_isc.sum())}  "
          f"(old-y ∩ code)={int((y_old.astype(bool)&isc).sum())}")
    print(f"  line∩code == old-y∩code agreement: {agree}/{len(eids)} "
          f"({'MATCH' if agree==len(eids) else 'MISMATCH'})")
    print(f"  token∩code positives={int(tok_isc.sum())}  (tight diff, is_code-gated)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest")
    ap.add_argument("--model"); ap.add_argument("--acts"); ap.add_argument("--out")
    ap.add_argument("--dataset", default=str(REPO / "data" / "dataset.jsonl"))
    ap.add_argument("--split", default=str(REPO / "data" / "sven_split_meta.json"))
    ap.add_argument("--membership", default=str(HERE / "subtractive_membership.json"))
    ap.add_argument("--layers"); ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    if args.selftest:
        selftest(args.selftest)
    else:
        run(args)


if __name__ == "__main__":
    main()
