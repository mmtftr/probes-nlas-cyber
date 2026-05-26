"""Retrain span-max probe on the 80% SVEN train split (group-clean).

Filters the existing data/token_activations to keep only rows whose
pair_group_key is in the SVEN training split (sven_split_meta.json),
then runs train_probe_spanmax.train_one_layer per layer. Best layer
by example-AUC is saved to data/probe_spanmax_sven_train.npz.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.train_probe_spanmax import train_one_layer  # type: ignore


def pair_group_key(row: dict) -> str:
    repo = row.get("_origin_repo")
    if repo:
        return f"repo::{repo}"
    fn = row.get("_file_name") or ""
    func = row.get("_func_name") or ""
    if fn or func:
        return f"func::{fn}::{func}"
    return f"row::{hashlib.sha1((row.get('code') or '').encode('utf-8')).hexdigest()[:12]}"


def main() -> None:
    meta = json.loads((ROOT / "data" / "sven_split_meta.json").read_text())
    heldout_groups = set(meta["heldout_groups"])
    print(f"[retrain] heldout groups: {len(heldout_groups)}", file=sys.stderr)

    # Map example_id (= row index in dataset.jsonl) -> group_key
    eid_to_group: dict[int, str] = {}
    with (ROOT / "data" / "dataset.jsonl").open() as f:
        for i, line in enumerate(f):
            eid_to_group[i] = pair_group_key(json.loads(line))
    keep_eids = {
        eid for eid, g in eid_to_group.items() if g not in heldout_groups
    }
    print(
        f"[retrain] dataset.jsonl rows: {len(eid_to_group)}  "
        f"train_eids: {len(keep_eids)}  heldout_eids: {len(eid_to_group)-len(keep_eids)}",
        file=sys.stderr,
    )

    out_npz = ROOT / "data" / "probe_spanmax_sven_train.npz"
    out_card = ROOT / "data" / "probe_spanmax_sven_train_card.json"

    layers = [8, 17, 26, 34]
    results: list[dict] = []
    for li in layers:
        path = ROOT / "data" / "token_activations" / f"token_activations_layer{li:02d}.npz"
        t0 = time.time()
        npz = np.load(path)
        X_all, y_all, eids_all = npz["X"], npz["y"], npz["example_ids"]
        keep_mask = np.fromiter((int(e) in keep_eids for e in eids_all), dtype=bool, count=len(eids_all))
        X, y, eids = X_all[keep_mask], y_all[keep_mask], eids_all[keep_mask]
        n_ex = int(np.unique(eids).size)
        print(
            f"[retrain] layer {li:02d}  filtered tokens {keep_mask.sum()}/{len(keep_mask)}  "
            f"examples={n_ex}  load+filter {time.time()-t0:.1f}s",
            file=sys.stderr,
        )
        r = train_one_layer(
            X, y, eids,
            epochs=30, lr=1e-3, batch_examples=8,
            device="cuda" if __import__("torch").cuda.is_available() else "cpu",
            verbose=False,
        )
        r["layer"] = li
        results.append(r)
        print(
            f"[retrain] layer {li:02d}  tok_AUC={r['tok_auc']:.3f}  ex_AUC={r['ex_auc']:.3f}",
            file=sys.stderr,
        )

    def _score(r: dict) -> float:
        v = r["ex_auc"]
        return v if not np.isnan(v) else -1.0

    best = max(results, key=_score)
    print(
        f"[retrain] best layer = {best['layer']}  ex_AUC={best['ex_auc']:.3f}  tok_AUC={best['tok_auc']:.3f}",
        file=sys.stderr,
    )

    np.savez_compressed(
        out_npz,
        w=best["w"].astype(np.float32),
        b=np.float32(best["b"]),
        layer=np.int32(best["layer"]),
    )
    card = {
        "loss": "span-max (Obeso, Arditi et al. 2025)",
        "best_layer": best["layer"],
        "best_token_auc": float(best["tok_auc"]),
        "best_example_auc": float(best["ex_auc"]),
        "all_layers": [
            {
                "layer": r["layer"],
                "token_auc": float(r["tok_auc"]),
                "example_auc": float(r["ex_auc"]),
            }
            for r in results
        ],
        "split": "80% of pair_group_keys (sven_split_meta.json seed=42); 20% groups held out",
        "n_train_examples": int(len(keep_eids)),
        "dataset": "dataset.jsonl (SVEN-before + SVEN-after), filtered to train groups",
    }
    out_card.write_text(json.dumps(card, indent=2))
    print(f"[retrain] saved {out_npz} + {out_card}", file=sys.stderr)


if __name__ == "__main__":
    main()
