# [ai-generated]
"""5-fold × N-seed cross-validation variance pass for the subtractive grid.

Reuses the exp-19 harness (compute_labels/load_layer/build_pairs/_auc from
train_grid.py) but replaces the single fixed seed-42 split with grouped K-fold
CV: func-pair GROUPS are partitioned into K folds (so before/after pairs never
straddle train/test — no leakage), and each seed reshuffles the partition AND
seeds probe init. Reports mean±std per config over the K×seeds train/evals.

Runs at each model's operating layer (1 layer) — variance pass, not a layer
sweep. Resumable: per-(seed,fold,config) records appended to records.jsonl and
skipped on resubmit.

Self-test (no acts): `python train_grid_cv.py --selftest-folds <logits.npz>`
checks group-clean folds (no eid in train+test; each eid tested once per seed).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
from train_grid import compute_labels, load_layer, build_pairs, _auc  # noqa: E402

OP = {
    "Qwen/Qwen2.5-Coder-32B-Instruct": 25, "Qwen/Qwen2.5-Coder-7B-Instruct": 16,
    "google/gemma-3-1b-it": 25, "google/gemma-3-4b-it": 7,
    "google/gemma-3-12b-it": 15, "google/gemma-3-27b-it": 19, "google/gemma-3-12b-pt": 13,
}


def group_folds(ds, all_eids, n_folds, seed):
    """Partition func-pair groups into n_folds; return eid->fold dict."""
    groups = sorted(set((ds[e].get("_file_name"), ds[e].get("_func_name")) for e in all_eids))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(groups))
    fold_of_group = {}
    for fold, idxs in enumerate(np.array_split(perm, n_folds)):
        for gi in idxs:
            fold_of_group[groups[gi]] = fold
    return {e: fold_of_group[(ds[e].get("_file_name"), ds[e].get("_func_name"))] for e in all_eids}


def pair_stats(pairs, ex_max):
    if not pairs:
        return float("nan"), float("nan"), 0
    sv = np.array([ex_max.get(v, np.nan) for v, _ in pairs])
    ss = np.array([ex_max.get(s, np.nan) for _, s in pairs])
    ok = np.isfinite(sv) & np.isfinite(ss)
    if ok.sum() == 0:
        return float("nan"), float("nan"), 0
    acc = float((sv[ok] > ss[ok]).mean())
    auc = _auc(np.r_[np.ones(ok.sum()), np.zeros(ok.sum())], np.r_[sv[ok], ss[ok]])
    return acc, auc, int(ok.sum())


def run(args):
    from src.training.train_probe_spanmax import train_one_layer
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = [json.loads(l) for l in open(args.dataset)]
    v2s = build_pairs(ds)
    member = json.loads(Path(args.membership).read_text())
    sub_set = set(member["kept_eids"])
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    n_folds = args.folds

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rec_path = out / "cv_records.jsonl"
    done = set()
    records = []
    if rec_path.exists():
        for line in rec_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line); records.append(r)
                done.add((r["layer"], r["seed"], r["fold"], r["train_subset"], r["granularity"], r["negatives"]))
    recf = rec_path.open("a")

    layer = OP.get(args.model)
    acts_dir = Path(args.acts)
    lf = list(acts_dir.glob(f"token_activations_layer{layer:02d}.npz"))
    if not lf:
        lf = sorted(acts_dir.glob("token_activations_layer*.npz"))  # fallback: whatever's there
    for f in lf:
        L = int(f.stem.replace("token_activations_layer", ""))
        X, eids, cs, ce, isc = load_layer(acts_dir, L, ds)
        labs = compute_labels(eids, cs, ce, isc, ds, v2s)
        truth = labs["token"] & isc
        in_sub = np.fromiter((int(e) in sub_set for e in eids), bool, len(eids))
        all_eids = sorted(set(int(e) for e in eids))
        for seed in seeds:
            efold = group_folds(ds, all_eids, n_folds, seed)
            tok_fold = np.fromiter((efold[int(e)] for e in eids), int, len(eids))
            for k in range(n_folds):
                is_test = tok_fold == k
                test_eset = set(e for e in all_eids if efold[e] == k)
                sub_pairs = [(v, s) for v, s in v2s.items() if v in test_eset and s in test_eset and v in sub_set]
                add_pairs = [(v, s) for v, s in v2s.items() if v in test_eset and s in test_eset and v not in sub_set]
                for subset in ("base", "subtractive"):
                    smask = np.ones(len(eids), bool) if subset == "base" else in_sub
                    for gran in ("line", "token"):
                        ypos = labs[gran] & isc
                        for neg, mn in (("Y", "none"), ("X", "code_only")):
                            key = (L, seed, k, subset, gran, neg)
                            if key in done:
                                continue
                            trmask = smask & ~is_test
                            if len(np.unique(ypos[trmask])) < 2:
                                continue
                            r = train_one_layer(
                                X[trmask], ypos[trmask].astype(np.int8), eids[trmask],
                                epochs=args.epochs, device=device, verbose=False, seed=seed,
                                mask_negatives=mn, code_mask=isc[trmask] if mn == "code_only" else None,
                            )
                            w, b = np.asarray(r["w"], np.float32), float(r["b"])
                            logit = X @ w + b
                            with np.errstate(over="ignore"):
                                prob = 1.0 / (1.0 + np.exp(-logit))

                            def cauc(setmask):
                                m = setmask & isc & is_test
                                return _auc(truth[m], logit[m])
                            sa = cauc(in_sub); ba = cauc(np.ones(len(eids), bool))
                            cidx = np.where(isc)[0]
                            ec, pc = eids[cidx], prob[cidx]
                            o = np.argsort(ec, kind="stable"); ec, pc = ec[o], pc[o]
                            uq, st = np.unique(ec, return_index=True)
                            ex_max = dict(zip(uq.tolist(), np.maximum.reduceat(pc, st).tolist()))
                            spa, _, spn = pair_stats(sub_pairs, ex_max)
                            apa, _, apn = pair_stats(add_pairs, ex_max)
                            rec = {"layer": L, "seed": seed, "fold": k,
                                   "train_subset": subset, "granularity": gran, "negatives": neg,
                                   "sub_test_code_auc": sa, "base_test_code_auc": ba,
                                   "pair_acc_sub": spa, "pair_acc_add": apa,
                                   "n_sub_pairs": spn, "n_add_pairs": apn}
                            records.append(rec); recf.write(json.dumps(rec) + "\n"); recf.flush()
                            print(f"[cv] L{L} s{seed} f{k} {subset:11s} {gran:5s} {neg} "
                                  f"sub={sa:.3f} base={ba:.3f} pAcc[s={spa:.2f} a={apa:.2f}]", file=sys.stderr)
        del X
    recf.close()
    aggregate(records, out, args.model)


def aggregate(records, out, model):
    from collections import defaultdict
    groups = defaultdict(list)
    for r in records:
        groups[(r["layer"], r["train_subset"], r["granularity"], r["negatives"])].append(r)
    agg = []
    for (L, subset, gran, neg), rs in sorted(groups.items()):
        def ms(key):
            v = np.array([r[key] for r in rs], float); v = v[np.isfinite(v)]
            return (float(v.mean()), float(v.std()), int(len(v))) if len(v) else (float("nan"), float("nan"), 0)
        m_sub = ms("sub_test_code_auc"); m_base = ms("base_test_code_auc")
        m_psub = ms("pair_acc_sub"); m_padd = ms("pair_acc_add")
        agg.append({"layer": L, "train_subset": subset, "granularity": gran, "negatives": neg,
                    "n": len(rs),
                    "sub_test_code_auc_mean": m_sub[0], "sub_test_code_auc_std": m_sub[1],
                    "base_test_code_auc_mean": m_base[0], "base_test_code_auc_std": m_base[1],
                    "pair_acc_sub_mean": m_psub[0], "pair_acc_sub_std": m_psub[1],
                    "pair_acc_add_mean": m_padd[0], "pair_acc_add_std": m_padd[1]})
    (out / "metrics_cv.json").write_text(json.dumps({"model": model, "aggregate": agg, "records": records}, indent=2))
    print(f"[cv] wrote {out/'metrics_cv.json'} ({len(records)} records, {len(agg)} configs)", file=sys.stderr)


def selftest_folds(npz_path, n_folds=5, seeds=(1, 2, 3)):
    ds = [json.loads(l) for l in open(REPO / "data" / "dataset.jsonl")]
    z = np.load(npz_path); eids = z["example_id"]
    all_eids = sorted(set(int(e) for e in eids))
    print(f"selftest folds: {len(all_eids)} eids")
    for seed in seeds:
        efold = group_folds(ds, all_eids, n_folds, seed)
        # each group entirely in one fold
        from collections import defaultdict
        gfold = defaultdict(set)
        for e in all_eids:
            gfold[(ds[e].get("_file_name"), ds[e].get("_func_name"))].add(efold[e])
        clean = all(len(v) == 1 for v in gfold.values())
        sizes = [sum(1 for e in all_eids if efold[e] == k) for k in range(n_folds)]
        # each eid tested exactly once across folds (trivially true: one fold/eid)
        print(f"  seed {seed}: group-clean={clean}  fold sizes={sizes}  sum={sum(sizes)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest-folds")
    ap.add_argument("--model"); ap.add_argument("--acts"); ap.add_argument("--out")
    ap.add_argument("--dataset", default=str(REPO / "data" / "dataset.jsonl"))
    ap.add_argument("--membership", default=str(HERE / "subtractive_membership.json"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    if args.selftest_folds:
        selftest_folds(args.selftest_folds)
    else:
        run(args)


if __name__ == "__main__":
    main()
