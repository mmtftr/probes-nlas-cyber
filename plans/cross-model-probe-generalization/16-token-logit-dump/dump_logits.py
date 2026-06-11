# [ai-generated]
"""Dump EVERY per-token + per-example probe logit over the full SVEN dataset.

Re-creates the canonical span-max vulnerability probe (deterministic: seed-42
group-clean split, identical trainer) from freshly extracted token activations,
then materialises the per-token logits that prior runs computed for AUC and threw
away. Fixes the gap recorded in memory `persist-token-level-predictions`.

For each captured layer under --acts (token_activations_layer{NN}.npz with
X/y/example_ids, plus offsets.npz):
  1. group-clean train/test split (seed=42, 20% held-out), loaded VERBATIM from
     the shared train_eval helper so the split matches every prior experiment;
  2. train a span-max linear probe on TRAIN tokens (train_one_layer);
  3. logit = X·w + b and prob = sigmoid(logit) for ALL tokens (train+test);
  4. attach per-token char offsets (from offsets.npz) and the tree-sitter
     live-code mask (code_only_mask) so each token row is self-describing;
  5. max-pool prob per example for the example-level score;
  6. verify on the HELD-OUT split: tokens_auc (all), tokens_code_auc (code-only,
     the historical headline), example_auc.

Outputs (under --out, one set per layer):
  - logits_layer{NN}.npz   — flat, token-aligned columns:
        logit, prob (float32); y (int8); example_id (int32);
        char_start, char_end (int32); is_test, is_code (bool)
  - example_scores_layer{NN}.json — [{eid, score, logit_max, label, cwe, lang, is_test}]
  - probe_layer{NN}.npz    — w, b, layer  (the probe itself, finally persisted)
  - metrics_logitdump.json — per-layer + best-layer AUCs, for the correctness gate
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from sklearn.metrics import roc_auc_score  # noqa: E402
from src.training.train_probe_spanmax import train_one_layer  # noqa: E402
from src.remotes.train_eval import load_or_make_split  # noqa: E402
from src.eval.code_mask import code_only_mask  # noqa: E402


def _auc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")


def _example_maxpool(prob, eids):
    """(eids_sorted_unique, max_prob, argmax_logit_idx) via stable group max."""
    order = np.argsort(eids, kind="stable")
    e_sorted, p_sorted = eids[order], prob[order]
    uniq, idx = np.unique(e_sorted, return_index=True)
    scores = np.maximum.reduceat(p_sorted, idx)
    return uniq, scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--acts", required=True, help="dir with token_activations_layer*.npz + offsets.npz")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default=None, help="comma-sep subset to dump; default = all present")
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    acts_dir = Path(args.acts)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows, train_eids, test_eids = load_or_make_split(Path(args.dataset), Path(args.split))
    test_set = set(int(e) for e in test_eids)

    # Per-row char offsets, in dataset (eid) order, to align with the flat token axis.
    offs_npz = np.load(acts_dir / "offsets.npz")
    n_rows = len(offs_npz.files)
    offsets_per_row = [offs_npz[f"offsets_row_{i:04d}"] for i in range(n_rows)]

    layer_files = sorted(acts_dir.glob("token_activations_layer*.npz"))
    if args.layers:
        want = {int(x) for x in args.layers.split(",") if x.strip()}
        layer_files = [f for f in layer_files
                       if int(f.stem.replace("token_activations_layer", "")) in want]
    if not layer_files:
        raise SystemExit(f"no layer npz under {acts_dir} (filter={args.layers})")

    per_layer_metrics = []
    for f in layer_files:
        li = int(f.stem.replace("token_activations_layer", ""))
        npz = np.load(f)
        X, y, eids = npz["X"], npz["y"].astype(np.int8), npz["example_ids"].astype(np.int32)

        # --- alignment guard: flat token axis must match concatenated row offsets ---
        char_start = np.empty(len(eids), dtype=np.int32)
        char_end = np.empty(len(eids), dtype=np.int32)
        is_code = np.zeros(len(eids), dtype=bool)
        cur = 0
        for eid in range(n_rows):
            n_tok = int((eids == eid).sum())
            o = offsets_per_row[eid]
            if o.shape[0] != n_tok:
                raise SystemExit(
                    f"layer {li}: offset/token mismatch at eid {eid}: "
                    f"offsets={o.shape[0]} tokens={n_tok} (truncation skew?)")
            char_start[cur:cur + n_tok] = o[:, 0]
            char_end[cur:cur + n_tok] = o[:, 1]
            m = code_only_mask(rows[eid].get("code", ""), rows[eid].get("lang", "") or "", o)
            is_code[cur:cur + n_tok] = m.astype(bool)
            cur += n_tok
        if cur != len(eids):
            raise SystemExit(f"layer {li}: consumed {cur} of {len(eids)} tokens")

        tr = np.fromiter((int(e) not in test_set for e in eids), bool, len(eids))
        te = ~tr
        if len(np.unique(y[tr])) < 2:
            print(f"[dump] layer {li}: train has one class, skipping", file=sys.stderr)
            continue

        r = train_one_layer(X[tr], y[tr], eids[tr], epochs=args.epochs, device=device, verbose=False)
        w, b = np.asarray(r["w"], np.float32), float(r["b"])

        logit = (X @ w + b).astype(np.float32)
        prob = (1.0 / (1.0 + np.exp(-logit))).astype(np.float32)

        # verification metrics on the held-out split
        tok_auc = _auc(y[te], prob[te])
        code_te = te & is_code
        tok_code_auc = _auc(y[code_te], prob[code_te])
        ex_ids, ex_p = _example_maxpool(prob[te], eids[te])
        ex_y = np.array([int(y[eids == e].max() > 0) for e in ex_ids])
        ex_auc = _auc(ex_y, ex_p)

        # --- dump every logit ---
        np.savez_compressed(
            out / f"logits_layer{li:02d}.npz",
            logit=logit, prob=prob, y=y, example_id=eids,
            char_start=char_start, char_end=char_end, is_test=te, is_code=is_code)
        np.savez_compressed(out / f"probe_layer{li:02d}.npz",
                            w=w, b=np.float32(b), layer=np.int32(li))

        # example-level over ALL examples (not just test)
        all_eids, all_scores = _example_maxpool(prob, eids)
        ex_records = []
        for e, sc in zip(all_eids.tolist(), all_scores.tolist()):
            row = rows[e]
            ex_records.append({
                "eid": int(e), "score": float(sc),
                "logit_max": float(logit[eids == e].max()),
                "label": int(np.asarray(y[eids == e]).max() > 0),
                "cwe": row.get("cwe"), "lang": row.get("lang"),
                "is_test": int(e) in test_set})
        (out / f"example_scores_layer{li:02d}.json").write_text(json.dumps(ex_records))

        m = {"layer": li, "n_tokens": int(len(eids)),
             "n_tokens_code": int(is_code.sum()),
             "test_tokens_auc": tok_auc, "test_tokens_code_auc": tok_code_auc,
             "test_example_auc": ex_auc, "val_ex_auc": float(r["ex_auc"]),
             "n_test_examples": int(len(ex_ids))}
        per_layer_metrics.append(m)
        print(f"[dump] {args.model} L{li}: tok_auc={tok_auc:.3f} "
              f"tok_code_auc={tok_code_auc:.3f} ex_auc={ex_auc:.3f}", file=sys.stderr)

    if not per_layer_metrics:
        raise SystemExit("no layer produced metrics")
    best = max(per_layer_metrics,
               key=lambda d: (d["test_example_auc"] if d["test_example_auc"] == d["test_example_auc"] else -1))
    summary = {"model": args.model, "acts": str(acts_dir),
               "n_examples": len(rows), "n_train_ex": len(train_eids),
               "n_test_ex": len(test_eids), "best_layer": best["layer"],
               "best": best, "layers": per_layer_metrics}
    (out / "metrics_logitdump.json").write_text(json.dumps(summary, indent=2))
    print(f"[dump] {args.model}: best L{best['layer']} "
          f"tok_code_auc={best['test_tokens_code_auc']:.3f} "
          f"ex_auc={best['test_example_auc']:.3f}", file=sys.stderr)


if __name__ == "__main__":
    main()
