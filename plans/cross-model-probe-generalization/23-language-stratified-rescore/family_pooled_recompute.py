# [ai-generated]
"""Standalone, numpy-only recompute of the TRUE pooled family-vs-rest AUC for
FIG-H, with the exp-16 format gate. Runs locally (models whose npz is present) or
on the cluster (all 7) — identical code, paths via env:

  LOGIT_ROOT  dir holding <slug>/logits_layer<LL>.npz   (default: repo exp-16 results)
  DS_PATH     dataset.jsonl                              (default: repo data/dataset.jsonl)
  OUT_JSON    where to write the result                 (default: results/family_pooled_cluster.json)

AUC is the tie-corrected Mann-Whitney rank statistic == sklearn roc_auc_score, so
no sklearn dependency (the NGC image only ships numpy). Each model first reproduces
its historical tokens_code_auc anchor (gate, |Δ|<=0.001) before its family_pooled
is trusted. family_pooled mirrors rescore_language.py's field exactly.
"""
import json
import os
from pathlib import Path

import numpy as np

# Defaults are LAZY: only resolve the repo root (parents[3]) when the env var is
# absent (local runs). On the cluster the script lives at /tmp (shallow path) and
# all three are passed via env, so parents[3] is never evaluated.
_HERE = Path(__file__).resolve()
LOGIT_ROOT = Path(os.environ["LOGIT_ROOT"]) if "LOGIT_ROOT" in os.environ else (
    _HERE.parents[3] / "plans/cross-model-probe-generalization/16-token-logit-dump/results")
DS_PATH = Path(os.environ["DS_PATH"]) if "DS_PATH" in os.environ else (
    _HERE.parents[3] / "data" / "dataset.jsonl")
OUT_JSON = Path(os.environ["OUT_JSON"]) if "OUT_JSON" in os.environ else (
    _HERE.parent / "results" / "family_pooled_cluster.json")

# (slug dir, operating layer, historical tokens_code_auc anchor) — from rescore_language.py
MODELS = {
    "Qwen2.5-Coder-32B-Instruct": ("logitdump_Qwen_Qwen2.5-Coder-32B-Instruct", 25, 0.776),
    "Qwen2.5-Coder-7B-Instruct":  ("logitdump_Qwen_Qwen2.5-Coder-7B-Instruct", 16, 0.813),
    "gemma-3-1b-it":  ("logitdump_google_gemma-3-1b-it", 25, 0.744),
    "gemma-3-4b-it":  ("logitdump_google_gemma-3-4b-it", 7, 0.775),
    "gemma-3-12b-it": ("logitdump_google_gemma-3-12b-it", 15, 0.763),
    "gemma-3-27b-it": ("logitdump_google_gemma-3-27b-it", 19, 0.759),
    "gemma-3-12b-pt": ("logitdump_google_gemma-3-12b-pt", 13, 0.782),
}
FAM_FIG = {"injection": ["CWE-089", "CWE-078", "CWE-022", "CWE-079"],
           "memory": ["CWE-125", "CWE-416", "CWE-476"]}


def auc(y, s):
    """Tie-corrected Mann-Whitney AUC (== sklearn.metrics.roc_auc_score)."""
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    _, inv, counts = np.unique(s[order], return_counts=True, return_inverse=True)
    cum = np.cumsum(counts); start = cum - counts
    avg_rank = (start + cum + 1) / 2.0      # 1-based average rank within each tie group
    ranks = np.empty(len(s)); ranks[order] = avg_rank[inv]
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def main():
    ds = [json.loads(l) for l in open(DS_PATH)]
    out = {}
    for model, (slug, L, anchor) in MODELS.items():
        npz_path = LOGIT_ROOT / slug / f"logits_layer{L:02d}.npz"
        if not npz_path.exists():
            print(f"{model:28s} SKIP (no npz at {npz_path})")
            continue
        npz = np.load(npz_path)
        prob = npz["prob"]; y_line = npz["y"].astype(int); eid = npz["example_id"]
        base = npz["is_test"].astype(bool) & npz["is_code"].astype(bool)
        cwe_tok = np.array([ds[int(e)].get("cwe") for e in eid], dtype=object)
        label_tok = np.array([ds[int(e)]["label"] for e in eid])

        gate = auc(y_line[base], prob[base])
        gate_pass = abs(gate - anchor) <= 0.001

        fam_pooled = {}
        for fam, cwes in FAM_FIG.items():
            # trusted = the per-CWE untrusted flag rescore uses: n_pos_examples >= 10
            # (counted WITHOUT the y_line gate, exactly as rescore_language.py does)
            keep = [c for c in cwes
                    if len(np.unique(eid[base & (label_tok == 1) & (cwe_tok == c)])) >= 10]
            in_fam = np.isin(cwe_tok.astype(str), keep)
            fam_pos = base & (label_tok == 1) & (y_line == 1) & in_fam
            fam_pooled[fam] = {
                "pooled_auc": auc(fam_pos[base].astype(int), prob[base]),
                "trusted_cwes": keep,
                "n_pos_tokens": int(fam_pos.sum()),
                "n_tokens": int(base.sum()),
            }
        out[model] = {"layer": L, "anchor_tokens_code_auc": anchor,
                      "gate_line_code_auc": gate, "gate_delta": gate - anchor,
                      "gate_pass": bool(gate_pass), "family_pooled": fam_pooled}
        ip = fam_pooled["injection"]["pooled_auc"]; mp = fam_pooled["memory"]["pooled_auc"]
        print(f"{model:28s} L{L:>2} gate={gate:.4f}(Δ{gate-anchor:+.4f},"
              f"{'OK' if gate_pass else 'FAIL'}) inj={ip:.4f} mem={mp:.4f}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {len(out)} models -> {OUT_JSON}")


if __name__ == "__main__":
    main()
