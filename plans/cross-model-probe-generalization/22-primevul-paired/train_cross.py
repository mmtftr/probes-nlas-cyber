# [ai-generated]
"""Cross-dataset probe train/eval: SVEN <-> PrimeVul, one model, operating layer.

Trains the canonical regime (token granularity, X = code-only negatives,
is_code-gated positives, SUBTRACTIVE subset) on three train sources and
cross-evaluates every probe on every eval set:

  train sources : sven_cpp (SVEN C/C++ slice), sven_full (all SVEN), pv (PrimeVul)
  eval sets     : sven_cpp_test, sven_full_test, pv_test

Per (probe, eval-set) we report the HONEST metrics, identical definition to
exp-19 so numbers are directly comparable:
  - token-code-AUC  (truth = tight-token ∩ is_code, over test code tokens)
  - g-mean^2        (TPR*TNR at the g-mean-max threshold, same tokens)
  - pairAcc sub/add (example-level max code-prob, vuln vs safe, test pairs)

A probe trained on dataset A is applied to dataset B's activations
(logit_B = X_B @ w + b) — that IS the cross-dataset transfer measurement.

Reuses exp-19 train_grid helpers (compute_labels/build_pairs/load_layer) and
src.eval.metrics.max_gmean. Run inside the container container after env.sh.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))                 # local self-contained train_grid copy
import train_grid as tg                       # noqa: E402  compute_labels/build_pairs/load_layer/_auc
from src.eval.metrics import max_gmean        # noqa: E402

CPP = {"c", "cpp", "c++"}


def sven_test_eids(ds_path, split_path):
    from src.remotes.the cluster.train_eval import load_or_make_split
    _, _train, test = load_or_make_split(Path(ds_path), Path(split_path))
    return set(int(e) for e in test)


def make_state(name, ds_path, acts_dir, layer, membership_path, test_eids):
    ds = [json.loads(l) for l in open(ds_path)]
    v2s = tg.build_pairs(ds)
    X, eids, cs, ce, isc = tg.load_layer(Path(acts_dir), layer, ds)
    labs = tg.compute_labels(eids, cs, ce, isc, ds, v2s)
    truth = labs["token"] & isc                       # honest token positive
    ypos = (labs["token"] & isc).astype(np.int8)      # train positives (token, is_code)
    sub_set = set(json.loads(Path(membership_path).read_text())["kept_eids"])
    test_set = set(int(e) for e in test_eids)
    is_test = np.fromiter((int(e) in test_set for e in eids), bool, len(eids))
    in_sub = np.fromiter((int(e) in sub_set for e in eids), bool, len(eids))
    lang_per = np.array([ (ds[int(e)].get("lang") or "").lower() in CPP for e in eids ], bool)
    return dict(name=name, ds=ds, X=X, eids=eids, isc=isc, truth=truth, ypos=ypos,
                v2s=v2s, sub_set=sub_set, test_set=test_set,
                is_test=is_test, in_sub=in_sub, cpp_tok=lang_per)


def train_probe(st, train_mask, epochs, device):
    from src.training.train_probe_spanmax import train_one_layer
    if len(np.unique(st["ypos"][train_mask])) < 2:
        return None
    r = train_one_layer(
        st["X"][train_mask], st["ypos"][train_mask], st["eids"][train_mask],
        epochs=epochs, device=device, verbose=False,
        mask_negatives="code_only", code_mask=st["isc"][train_mask])
    return np.asarray(r["w"], np.float32), float(r["b"])


def ex_max_prob(st, score):
    """max SCORE over each example's CODE tokens -> {eid: max}. Rank on LOGIT,
    not sigmoid(prob): large logits saturate prob to exactly 1.0 and tie, which
    collapses pairAcc to 0 (the bug seen in exp-16 gmean.py)."""
    cidx = np.where(st["isc"])[0]
    ec, pc = st["eids"][cidx], score[cidx]
    o = np.argsort(ec, kind="stable"); ec, pc = ec[o], pc[o]
    uq, start = np.unique(ec, return_index=True)
    mx = np.maximum.reduceat(pc, start)
    return dict(zip(uq.tolist(), mx.tolist()))


def pair_stats(pairs, exmax):
    if not pairs:
        return float("nan"), 0
    sv = np.array([exmax.get(v, np.nan) for v, _ in pairs])
    ss = np.array([exmax.get(s, np.nan) for _, s in pairs])
    ok = np.isfinite(sv) & np.isfinite(ss)
    if ok.sum() == 0:
        return float("nan"), 0
    return float((sv[ok] > ss[ok]).mean()), int(ok.sum())


def eval_set_pairs(st, cpp_only):
    """(sub_pairs, add_pairs) among this state's held-out test pairs."""
    sub, add = [], []
    for v, s in st["v2s"].items():
        if v not in st["test_set"] or s not in st["test_set"]:
            continue
        if cpp_only and not (st["ds"][v].get("lang", "").lower() in CPP):
            continue
        (sub if v in st["sub_set"] else add).append((v, s))
    return sub, add


def eval_on(w, b, st, token_mask, sub_pairs, add_pairs):
    logit = st["X"] @ w + b
    m = token_mask & st["isc"]
    auc = tg._auc(st["truth"][m], logit[m])
    g2 = max_gmean(st["truth"][m].astype(int), logit[m])["gmean_squared"]
    exmax = ex_max_prob(st, logit)   # rank on logit (tie-free), not saturated prob
    pa_sub, n_sub = pair_stats(sub_pairs, exmax)
    pa_add, n_add = pair_stats(add_pairs, exmax)
    return dict(token_code_auc=auc, gmean_sq=g2,
                pair_acc_sub=pa_sub, n_sub_pairs=n_sub,
                pair_acc_add=pa_add, n_add_pairs=n_add,
                n_eval_tok=int(m.sum()), n_eval_pos=int(st["truth"][m].sum()))


def run(args):
    # Probe training is a linear head on CACHED activations (trivial FLOPs) — run
    # it on CPU. The GPUs are still busy with the OTHER model's extraction in the
    # same job, and a 95 GB GPU can show ~0 free mid-run; CPU (820 GB cgroup)
    # decouples training from that contention. (Repo rule: probes are cheap.)
    device = "cpu"

    sven = make_state("sven", args.sven_dataset, args.sven_acts, args.layer_sven,
                      args.sven_membership,
                      sven_test_eids(args.sven_dataset, args.sven_split))
    pv_member = json.loads(Path(args.pv_membership).read_text())
    pv = make_state("pv", args.pv_dataset, args.pv_acts, args.layer_pv,
                    args.pv_membership, set(pv_member["test_eids"]))

    # train masks (subtractive ∩ train-split, token/X/is_code)
    sven_train = sven["in_sub"] & ~sven["is_test"]
    train_masks = {
        "sven_cpp":  sven_train & sven["cpp_tok"],
        "sven_full": sven_train,
        "pv":        pv["in_sub"] & ~pv["is_test"],
    }
    train_state = {"sven_cpp": sven, "sven_full": sven, "pv": pv}

    # eval sets: (state, token_mask, cpp_only)
    eval_sets = {
        "sven_cpp_test":  (sven, sven["is_test"] & sven["cpp_tok"], True),
        "sven_full_test": (sven, sven["is_test"], False),
        "pv_test":        (pv,   pv["is_test"], False),
    }
    eval_pairs = {k: eval_set_pairs(st, cpp) for k, (st, _, cpp) in eval_sets.items()}

    probes = {}
    for name, mask in train_masks.items():
        wb = train_probe(train_state[name], mask, args.epochs, device)
        probes[name] = wb
        print(f"[train] {name:9s} train_pos={int(train_state[name]['ypos'][mask].sum())} "
              f"train_tok={int(mask.sum())} {'OK' if wb else 'SKIP(<2 classes)'}", file=sys.stderr)

    results = []
    for pname, wb in probes.items():
        if wb is None:
            continue
        w, b = wb
        for ename, (st, tmask, _cpp) in eval_sets.items():
            sub_p, add_p = eval_pairs[ename]
            m = eval_on(w, b, st, tmask, sub_p, add_p)
            rec = {"model": args.model, "train": pname, "eval": ename,
                   "layer_sven": args.layer_sven, "layer_pv": args.layer_pv, **m}
            results.append(rec)
            print(f"[eval] {pname:9s} -> {ename:14s} tokAUC={m['token_code_auc']:.3f} "
                  f"g2={m['gmean_sq']:.3f} pairAcc[sub={m['pair_acc_sub']:.2f} "
                  f"add={m['pair_acc_add']:.2f}]", file=sys.stderr)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "metrics_cross.json").write_text(json.dumps(
        {"model": args.model, "results": results}, indent=2))
    print(f"[cross] wrote {out/'metrics_cross.json'} ({len(results)} records)", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sven-acts", required=True); ap.add_argument("--pv-acts", required=True)
    ap.add_argument("--sven-dataset", required=True)
    ap.add_argument("--sven-membership", default=str(HERE / "subtractive_membership.json"))
    ap.add_argument("--sven-split", default=str(REPO / "data" / "sven_split_meta.json"))
    ap.add_argument("--pv-dataset", required=True); ap.add_argument("--pv-membership", required=True)
    ap.add_argument("--layer-sven", type=int, required=True)
    ap.add_argument("--layer-pv", type=int, required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--epochs", type=int, default=30)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
