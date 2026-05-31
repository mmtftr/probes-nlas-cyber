# [ai-generated]
"""Probe vs. the model's own verbalized judgment, on the SAME held-out examples.

No model here — uses the cached per-layer activation memmaps (probe side) and
the precomputed verbalized P(yes) scores from verbalized_judge.py. For each of K
group-clean splits (seeds), train a linear span-max probe on the TRAIN tokens at
the model's best single layer, max-pool to a per-example probe score on the test
split, and compare its example-AUC to the verbalized example-AUC over the SAME
test example ids.

    probe ex-AUC      = roc_auc_score(ex_label, probe_score)        # internal
    verbalized ex-AUC = roc_auc_score(ex_label, [p_yes[e] ...])     # stated
    delta             = probe - verbalized                          # the gap

delta > +1 std  ⇒ the probe reads an internal vulnerability belief the model does
not state when asked (the introspection gap; validates probing as more than
"just ask the LLM"). delta ≈ 0 ⇒ asking suffices. delta < 0 ⇒ asking beats it.

Mirrors the cached-acts reuse / split logic of exp 03 (loss_alpha_sweep.py):
acts dir holds layer_{NN}.npy, y.npy, example_ids.npy; the split is the
group-clean make_split_for_seed (copied verbatim from exp-02 splits_variance.py);
pair_group_key / example_scores come from src/remotes/the cluster/train_eval.py.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from src.training.train_probe_spanmax import train_one_layer  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

LENGTH_BASELINE = 0.575  # SVEN length-baseline example-AUC (exp 02/03 reference)


def _load_train_eval():
    p = REPO / "src" / "remotes" / "the cluster" / "train_eval.py"
    spec = importlib.util.spec_from_file_location("remote_train_eval", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_split_for_seed(eid_to_group, seed, frac_heldout=0.2):
    """Group-clean held-out split for a seed (mirrors load_or_make_split exactly).
    Copied verbatim from exp-02 splits_variance.py / exp-03 loss_alpha_sweep.py."""
    groups = sorted(set(eid_to_group.values()))
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n_held = max(1, int(round(frac_heldout * len(groups))))
    heldout = set(groups[:n_held])
    train_eids = {e for e, g in eid_to_group.items() if g not in heldout}
    return train_eids, {e for e, g in eid_to_group.items() if g in heldout}


def merge_verbalized(scores_dir: Path):
    """Concatenate all verbalized_scores.gpu*.json -> {eid: p_yes}, {eid: label}."""
    p_yes, lab = {}, {}
    files = sorted(scores_dir.glob("verbalized_scores.gpu*.json"))
    if not files:
        raise SystemExit(f"no verbalized_scores.gpu*.json under {scores_dir}")
    for f in files:
        for rec in json.loads(f.read_text()):
            e = int(rec["eid"])
            p_yes[e] = float(rec["p_yes"])
            lab[e] = int(rec["label"])
    return p_yes, lab


def _ms(vals):
    a = np.array([v for v in vals if v is not None and v == v], dtype=float)
    return (float(a.mean()), float(a.std(ddof=0))) if a.size else (None, None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--scores-glob", required=True,
                    help="directory holding verbalized_scores.gpu*.json")
    ap.add_argument("--layer", type=int, required=True,
                    help="model's best single layer (Gemma 19, Qwen 41)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    layer = args.layer
    te_mod = _load_train_eval()
    acts = Path(args.acts_dir)

    p_yes, vlab = merge_verbalized(Path(args.scores_glob))
    print(f"[compare] merged {len(p_yes)} verbalized scores", file=sys.stderr)

    Xfull = np.asarray(np.load(acts / f"layer_{layer:02d}.npy", mmap_mode="r"),
                       dtype=np.float32)
    y = np.load(acts / "y.npy")
    eids = np.load(acts / "example_ids.npy")
    if not np.isfinite(Xfull).all():
        raise SystemExit(f"non-finite activations at layer {layer}")

    rows = [json.loads(l) for l in Path(args.dataset).open() if l.strip()]
    eid_to_group = {i: te_mod.pair_group_key(r) for i, r in enumerate(rows)}

    per_seed = []
    seed42 = {}  # stored arrays for the disagreement analysis + plotting
    for seed in seeds:
        tr_eids, te_eids = make_split_for_seed(eid_to_group, seed)
        tr = np.fromiter((int(e) in tr_eids for e in eids), bool, len(eids))
        te = ~tr
        ytr, etr = y[tr], eids[tr]
        if len(np.unique(ytr)) < 2 or te.sum() == 0:
            per_seed.append({"seed": seed, "skipped": "degenerate"})
            continue
        r = train_one_layer(Xfull[tr], ytr, etr, epochs=args.epochs, device=device,
                             verbose=False, alpha=args.alpha, neg_incl=False)
        w, b = np.asarray(r["w"], np.float32), float(r["b"])
        Xte = Xfull[te]
        tok_p = 1.0 / (1.0 + np.exp(-(Xte @ w + b)))
        tok_y, te_tok_eids = y[te], eids[te]
        ex_ids, ex_p = te_mod.example_scores(tok_p, te_tok_eids)
        ex_y = np.array([int(y[(eids == e)].max() > 0) for e in ex_ids])

        # Verbalized scores over the SAME test example ids (intersect with scored).
        keep = np.array([int(e) in p_yes for e in ex_ids], dtype=bool)
        ex_ids_k = ex_ids[keep]
        probe_score = ex_p[keep]
        ex_y_k = ex_y[keep]
        verb_score = np.array([p_yes[int(e)] for e in ex_ids_k], dtype=float)

        probe_auc = (float(roc_auc_score(ex_y_k, probe_score))
                     if len(np.unique(ex_y_k)) > 1 else float("nan"))
        verb_auc = (float(roc_auc_score(ex_y_k, verb_score))
                    if len(np.unique(ex_y_k)) > 1 else float("nan"))
        per_seed.append({
            "seed": seed, "probe_auc": probe_auc, "verbalized_auc": verb_auc,
            "delta": probe_auc - verb_auc, "n_test_ex": int(len(ex_ids_k)),
        })
        if seed == 42:
            seed42 = {
                "ex_ids": [int(e) for e in ex_ids_k],
                "ex_label": [int(v) for v in ex_y_k],
                "probe_score": [float(v) for v in probe_score],
                "p_yes": [float(v) for v in verb_score],
            }
        print(f"[compare] seed {seed}: probe={probe_auc:.3f} verbalized={verb_auc:.3f} "
              f"delta={probe_auc - verb_auc:+.3f} (n={len(ex_ids_k)})", file=sys.stderr)

    valid = [s for s in per_seed if "probe_auc" in s]
    probe_m, probe_s = _ms([s["probe_auc"] for s in valid])
    verb_m, verb_s = _ms([s["verbalized_auc"] for s in valid])
    delta_m, delta_s = _ms([s["delta"] for s in valid])

    # Disagreement analysis on seed=42 test set.
    spearman = None
    probe_catches_model_denies = []
    model_catches_probe_misses = []
    if seed42:
        ps = np.array(seed42["probe_score"])
        py = np.array(seed42["p_yes"])
        lab = np.array(seed42["ex_label"])
        ids = np.array(seed42["ex_ids"])
        if len(ps) > 2 and np.std(ps) > 0 and np.std(py) > 0:
            rho, _ = spearmanr(ps, py)
            spearman = float(rho)
        # Top tercile of probe scores on this test set.
        top_thresh = np.quantile(ps, 2.0 / 3.0)
        bot_thresh = np.quantile(ps, 1.0 / 3.0)
        # Probe catches (top tercile), model verbally denies (p_yes < 0.5), true positive.
        m1 = (lab == 1) & (ps >= top_thresh) & (py < 0.5)
        probe_catches_model_denies = [int(e) for e in ids[m1]]
        # Reverse: model says yes (p_yes >= 0.5) on a true positive the probe ranks low.
        m2 = (lab == 1) & (py >= 0.5) & (ps <= bot_thresh)
        model_catches_probe_misses = [int(e) for e in ids[m2]]

    record = {
        "model_layer": layer,
        "alpha": args.alpha,
        "epochs": args.epochs,
        "seeds": seeds,
        "n_examples_scored": len(p_yes),
        "probe_auc_mean": probe_m, "probe_auc_std": probe_s,
        "verbalized_auc_mean": verb_m, "verbalized_auc_std": verb_s,
        "delta_mean": delta_m, "delta_std": delta_s,
        "length_baseline": LENGTH_BASELINE,
        "spearman": spearman,
        "n_probe_catches_model_denies": len(probe_catches_model_denies),
        "probe_catches_model_denies": probe_catches_model_denies[:10],
        "n_model_catches_probe_misses": len(model_catches_probe_misses),
        "model_catches_probe_misses": model_catches_probe_misses[:10],
        "per_seed": per_seed,
        "seed42_arrays": seed42,
    }
    Path(args.out).write_text(json.dumps(record, indent=2))

    print(f"[compare] layer L{layer:02d}  probe_auc={probe_m:.3f}±{probe_s:.3f}  "
          f"verbalized_auc={verb_m:.3f}±{verb_s:.3f}  "
          f"delta={delta_m:+.3f}±{delta_s:.3f}", file=sys.stderr)
    print(f"[compare] spearman(probe,p_yes)={spearman}  "
          f"probe-catches-model-denies={len(probe_catches_model_denies)} "
          f"{probe_catches_model_denies[:5]}  "
          f"model-catches-probe-misses={len(model_catches_probe_misses)} "
          f"{model_catches_probe_misses[:5]}", file=sys.stderr)
    if delta_m is not None and delta_s is not None:
        verdict = ("INTROSPECTION GAP (probe > verbalized by >1 std)"
                   if delta_m > delta_s else
                   "verbalized beats probe" if delta_m < -delta_s else
                   "no clear gap (within ±1 std)")
        print(f"[compare] verdict: {verdict}", file=sys.stderr)


if __name__ == "__main__":
    main()
