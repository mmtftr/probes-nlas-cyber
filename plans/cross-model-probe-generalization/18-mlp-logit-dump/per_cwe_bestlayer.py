# [ai-generated]
"""Per-CWE token-code AUC for the MLP head AT ITS OWN BEST LAYER (2026-06-14).

exp-18 dumped, per (model, head), a full per-token table `logits_mlp.npz` at the
head's val-swept best layer (cols: prob, y, example_id, is_test, is_code, ...).
`metrics_mlp.json` reports only the OVERALL test tokens_code_auc; it has no
per-CWE breakdown. The blog fig-4 memory panel needs the memory-CWE token AUC at
the SAME best layer the overall number uses (so overall and memory dashed lines
describe one MLP, not two layer policies).

This recomputes the exact exp-09 `by_cwe` contrast from the dump:
  per-CWE subset = {test examples with cwe==C}  ∪  {clean (cwe=None) test examples}
  tokens_code_auc = roc_auc(y, prob) over that subset's is_code tokens.
`is_code` in the npz IS the honest_scoring code mask — verified here by
reproducing each `metrics_mlp.json` test_tokens_code_auc bit-for-bit (assert).
eid->cwe / is_test come from the committed `example_scores_mlp.json`.

Best head per model = max test_tokens_code_auc (the selection fig-3's MLP_AUC
uses). The npz live OUTSIDE the public repo (large); point MLP_NPZ_ROOT at them
(default: the local scp mirror). Output `results/mlp_memory_bestlayer.json` IS
committed — the figure reads that, so a re-render needs no npz.

Run: MLP_NPZ_ROOT=<dir> python per_cwe_bestlayer.py
"""
import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
NPZ_ROOT = Path(os.environ.get(
    "MLP_NPZ_ROOT", HERE / "_mlp_npz"))
MEM = ["CWE-125", "CWE-416", "CWE-476"]


def best_head(slug):
    """(head, layer, test_tokens_code_auc) = the head with max test code-AUC."""
    cands = []
    for h in ("mlp256", "mlp512"):
        mp = RES / slug / h / "metrics_mlp.json"
        if mp.exists():
            m = json.loads(mp.read_text())
            cands.append((h, m["layer"], m["test_tokens_code_auc"]))
    return max(cands, key=lambda t: t[2])


def per_cwe(slug, head):
    z = np.load(NPZ_ROOT / slug / head / "logits_mlp.npz", allow_pickle=True)
    eid, y, p = z["example_id"], z["y"].astype(int), z["prob"].astype(float)
    is_test, is_code = z["is_test"], z["is_code"]
    ref = json.loads((RES / slug / head / "metrics_mlp.json").read_text())["test_tokens_code_auc"]
    overall = float(roc_auc_score(y[is_test & is_code], p[is_test & is_code]))
    assert abs(overall - ref) < 1e-9, f"{slug}: overall {overall} != ref {ref}"

    ex = json.loads((RES / slug / head / "example_scores_mlp.json").read_text())
    test_eids = {int(e["eid"]) for e in ex if e.get("is_test")}
    cwe_of = {int(e["eid"]): e["cwe"] for e in ex if e.get("is_test")}
    neg = {e for e in test_eids if not cwe_of.get(e)}
    out = {}
    for c in MEM:
        pos = {e for e in test_eids if cwe_of.get(e) == c}
        m = np.isin(eid, np.fromiter(pos | neg, dtype=eid.dtype)) & is_code & is_test
        yy, pp = y[m], p[m]
        out[c] = float(roc_auc_score(yy, pp)) if len(set(yy.tolist())) == 2 else None
    vals = [out[c] for c in MEM if out[c] is not None]
    return overall, out, (float(np.mean(vals)) if vals else None)


def main():
    slugs = [d.name for d in sorted(RES.iterdir())
             if d.is_dir() and (d / "mlp512" / "metrics_mlp.json").exists()]
    res = {}
    for slug in slugs:
        head, layer, _ = best_head(slug)
        overall, by_cwe, mem = per_cwe(slug, head)
        res[slug] = {"head": head, "layer": layer, "overall": overall,
                     "by_cwe": by_cwe, "mem_mean": mem}
        print(f"{slug:38s} {head} L{layer}  overall={overall:.4f}  mem={mem:.4f}  "
              + "  ".join(f"{c}={by_cwe[c]:.3f}" for c in MEM if by_cwe[c]))
    (RES / "mlp_memory_bestlayer.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {RES / 'mlp_memory_bestlayer.json'}")


if __name__ == "__main__":
    main()
