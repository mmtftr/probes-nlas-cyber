# [ai-generated]
"""Sweep 6: per-language / per-CWE tokens_code breakdown at one model's best layer.

Trains the span-max probe on the train split at --layer, scores TEST tokens, then
reports tokens_code_auc restricted to subsets of the test set:
  - overall (all test tokens)
  - by language: rows with lang == {python, c, cpp} (pos+neg of that lang)
  - by CWE: positives with cwe == X  vs  ALL negatives (cwe is null on the
    `func_src_after` negatives) — "can the probe find CWE-X vulns among real code".

One model per invocation (cheap; cached acts, single layer). Reuses
src/eval/honest_scoring.honest_token_aucs on the filtered token subset.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from src.training.train_probe_spanmax import train_one_layer  # noqa: E402
from src.eval.honest_scoring import (  # noqa: E402
    honest_token_aucs, load_dataset_rows, load_offsets_npz,
)


def _load_train_eval():
    p = REPO / "src" / "remotes" / "train_eval.py"
    spec = importlib.util.spec_from_file_location("remote_train_eval", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--offsets", default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--model", default="")
    ap.add_argument("--min-cwe-pos", type=int, default=10,
                    help="skip CWEs with fewer than this many positive rows in test")
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    te_mod = _load_train_eval()
    acts = Path(args.acts_dir)
    offsets_by_eid = load_offsets_npz(Path(args.offsets) if args.offsets else acts / "offsets.npz")
    rows_by_eid = load_dataset_rows(Path(args.dataset))

    y = np.load(acts / "y.npy")
    eids = np.load(acts / "example_ids.npy")
    rows, train_eids, test_eids = te_mod.load_or_make_split(Path(args.dataset), Path(args.split))

    tr = np.fromiter((int(e) in train_eids for e in eids), bool, len(eids))
    te = np.fromiter((int(e) in test_eids for e in eids), bool, len(eids))

    Xmm = np.load(acts / f"layer_{args.layer:02d}.npy", mmap_mode="r")
    r = train_one_layer(np.asarray(Xmm[tr], np.float32), y[tr], eids[tr],
                        epochs=args.epochs, device=device, verbose=False)
    w, b = np.asarray(r["w"], np.float32), float(r["b"])

    Xte = np.asarray(Xmm[te], np.float32)
    tok_p, tok_y, te_e = 1.0 / (1.0 + np.exp(-(Xte @ w + b))), y[te], eids[te]

    def subset(eid_set):
        if not eid_set:
            return None
        m = np.isin(te_e, np.fromiter(eid_set, dtype=te_e.dtype))
        if m.sum() == 0:
            return None
        h = honest_token_aucs(tok_p[m], tok_y[m], te_e[m], offsets_by_eid, rows_by_eid)
        return {"tokens_code_auc": h["tokens_code_auc"], "tokens_auc": h["tokens_auc"],
                "n_pos_code": h["n_pos_code"], "n_total_code": h["n_total_code"],
                "n_examples": int(len(eid_set))}

    test_list = sorted(test_eids)
    out = {"model": args.model, "layer": args.layer,
           "overall": subset(set(test_list)),
           "by_lang": {}, "by_cwe": {}}
    for lang in ("python", "c", "cpp"):
        es = {e for e in test_list if (rows[e].get("lang") or "").lower() == lang}
        s = subset(es)
        if s:
            out["by_lang"][lang] = s

    neg_eids = {e for e in test_list if not rows[e].get("cwe")}
    cwe_counts = Counter(rows[e].get("cwe") for e in test_list if rows[e].get("cwe"))
    for cwe, n in cwe_counts.most_common():
        if n < args.min_cwe_pos:
            continue
        pos_eids = {e for e in test_list if rows[e].get("cwe") == cwe}
        s = subset(pos_eids | neg_eids)
        if s:
            s["n_pos_examples"] = len(pos_eids)
            out["by_cwe"][cwe] = s

    Path(args.out).write_text(json.dumps(out, indent=2))
    ov = out["overall"]["tokens_code_auc"] if out["overall"] else float("nan")
    print(f"[breakdown] {args.model} L{args.layer} overall tc={ov:.3f} "
          f"langs={list(out['by_lang'])} cwes={list(out['by_cwe'])}", file=sys.stderr)


if __name__ == "__main__":
    main()
