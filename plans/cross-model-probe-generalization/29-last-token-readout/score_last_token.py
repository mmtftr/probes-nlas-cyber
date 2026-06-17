# [ai-generated]
"""Last-code-token readout of the saved span-max probe, vs verbalized P(yes).

Pure CPU rescore (no torch, no GPU, no retraining). For each model, at its
canonical operating layer, read the EXISTING per-token probe logits (exp-16
logits_layer{NN}.npz) and turn them into example-level scores three ways, plus
the model's verbalized read:

  last_code_token : the probe LOGIT at each example's FINAL live-code token
  max_pool        : max probe LOGIT over the example's live-code tokens
  mean_pool       : mean probe LOGIT over the example's live-code tokens
  verbalized      : exp-17 P(yes) at the assistant turn-boundary (separate fwd)

All four are EXAMPLE-LEVEL (one score/function), scored with the SAME true
function-vuln label (row['label']) on the SAME test pool, so they compare
like-with-like. This is a SECONDARY (example-level) metric per project-log §3 —
it does NOT touch the token-level headline (tokens_code_auc 0.75-0.82). Reads use
the raw LOGIT (not prob): AUC(max logit) == AUC(max prob) by monotonicity, but
the logit avoids float32 sigmoid saturation ties that collapse prob-max to ~0.5.

Gates (hard-fail):
  - token gate: recomputed tokens_code_auc from the npz must equal exp-16's
    stored test_tokens_code_auc (proves logit/is_code/is_test/eid alignment).
  - verbalized gate: recomputed verbalized test AUC must match exp-17's
    verbalized_auc_test.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
E16 = REPO / "plans/cross-model-probe-generalization/16-token-logit-dump/results"
E17 = REPO / "plans/cross-model-probe-generalization/17-verbalized-logit-dump/results"
SUBSET = REPO / "plans/cross-model-probe-generalization/19-subtractive-regime/subtractive_membership.json"
DATASET = REPO / "data/dataset.jsonl"
SPLIT = REPO / "data/sven_split_meta.json"

# slug -> (operating layer = blog headline layer, exp-17 verbalized dir or None)
MODELS = {
    "logitdump_Qwen_Qwen2.5-Coder-32B-Instruct": (25, "Qwen_Qwen2.5-Coder-32B-Instruct"),
    "logitdump_Qwen_Qwen2.5-Coder-7B-Instruct": (16, "Qwen_Qwen2.5-Coder-7B-Instruct"),
    "logitdump_google_gemma-3-1b-it": (25, "google_gemma-3-1b-it"),
    "logitdump_google_gemma-3-4b-it": (7, "google_gemma-3-4b-it"),
    "logitdump_google_gemma-3-12b-it": (15, "google_gemma-3-12b-it"),
    "logitdump_google_gemma-3-12b-pt": (13, None),
    "logitdump_google_gemma-3-27b-it": (19, "google_gemma-3-27b-it"),
}

RNG_SEED = 42
N_BOOT = 1000
GATE_TOL = 1e-6          # same data + same sklearn -> bit-exact-ish
VERB_GATE_TOL = 5e-3     # label sourced differently (json vs npz); allow tiny drift


# --- inlined from src/remotes/train_eval.py (model-independent, no torch) ---
def pair_group_key(row: dict) -> str:
    repo = row.get("_origin_repo")
    if repo:
        return f"repo::{repo}"
    fn = row.get("_file_name") or ""
    func = row.get("_func_name") or ""
    if fn or func:
        return f"func::{fn}::{func}"
    return f"row::{hashlib.sha1((row.get('code') or '').encode()).hexdigest()[:12]}"


def load_split(dataset_path: Path, split_path: Path):
    rows = [json.loads(l) for l in dataset_path.open()]
    eid_to_group = {i: pair_group_key(r) for i, r in enumerate(rows)}
    heldout = set(json.loads(split_path.read_text())["heldout_groups"])
    test_eids = {e for e, g in eid_to_group.items() if g in heldout}
    return rows, test_eids


def _auc(y, s):
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")


def _boot_ci(y, s, n=N_BOOT):
    y, s = np.asarray(y), np.asarray(s)
    rng = np.random.default_rng(RNG_SEED)
    idx = np.arange(len(y))
    vals = [roc_auc_score(y[b], s[b]) for b in (rng.choice(idx, len(idx), replace=True) for _ in range(n))
            if len(np.unique(y[b])) > 1]
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"),) * 2


def _paired_delta_ci(y, sa, sb, n=N_BOOT):
    """CI of AUC(sa) - AUC(sb) on shared (paired) resamples."""
    y, sa, sb = np.asarray(y), np.asarray(sa), np.asarray(sb)
    rng = np.random.default_rng(RNG_SEED)
    idx = np.arange(len(y))
    d = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        d.append(roc_auc_score(y[b], sa[b]) - roc_auc_score(y[b], sb[b]))
    if not d:
        return {"delta_mean": float("nan"), "ci": [float("nan")] * 2}
    return {"delta_mean": float(np.mean(d)),
            "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


def per_example_reads(npz):
    """eid -> {last_code_token, max_pool, mean_pool} (all LOGIT, code-only)."""
    logit = npz["logit"]; y = npz["y"]
    eids = npz["example_id"]; is_code = npz["is_code"].astype(bool)
    out = {}
    for e in np.unique(eids):
        idx = np.where(eids == e)[0]
        assert idx[-1] - idx[0] + 1 == len(idx), f"eid {e}: token rows not contiguous"  # codex M5
        cm = is_code[idx]
        sel = idx[cm] if cm.any() else idx          # fall back to all tokens if no code token
        out[int(e)] = dict(
            last_code_token=float(logit[sel[-1]]),  # final live-code token (token order)
            max_pool=float(logit[sel].max()),
            mean_pool=float(logit[sel].mean()),
            n_pos_code=int((y[sel] > 0).sum()),
            has_code=bool(cm.any()),
        )
    return out


def stored_tok_code_auc(slug, layer):
    mp = E16 / slug / "metrics_logitdump.json"
    if not mp.exists():
        return None
    hit = [x for x in json.loads(mp.read_text())["layers"] if x["layer"] == layer]
    return hit[0]["test_tokens_code_auc"] if hit else None


def reads_from_reduced(red_path, layer):
    """Cluster-reduced per-example JSON (for models with no local npz)."""
    d = json.loads(Path(red_path).read_text())
    assert d["layer"] == layer, f"{red_path}: layer {d['layer']} != {layer}"
    reads = {int(e): {"last_code_token": v["last"], "max_pool": v["max"],
                      "mean_pool": v["mean"], "n_pos_code": v["npos"], "has_code": v["hc"]}
             for e, v in d["per_example"].items()}
    return reads, float(d["tokens_code_auc"])


def load_reads_and_gate(slug, layer):
    """Return (reads, gate) from the local npz if present, else the cluster-reduced
    JSON. Hard-fail if the recomputed tokens_code_auc disagrees with exp-16's stored
    value (proves the read is faithful)."""
    npz_p = E16 / slug / f"logits_layer{layer:02d}.npz"
    red_p = E16 / slug / "lasttok_reduced.json"
    stored = stored_tok_code_auc(slug, layer)
    if stored is None:
        raise SystemExit(f"[gate] {slug}: no stored tokens_code_auc for L{layer}")
    if npz_p.exists():
        npz = np.load(npz_p)
        prob, logit = npz["prob"], npz["logit"]
        assert np.max(np.abs(prob - 1.0 / (1.0 + np.exp(-logit)))) < 1e-4, "prob != sigmoid(logit)"
        m = npz["is_test"].astype(bool) & npz["is_code"].astype(bool)
        recomputed, reads, src, tol = _auc(npz["y"][m], prob[m]), per_example_reads(npz), "npz", GATE_TOL
    elif red_p.exists():
        reads, recomputed = reads_from_reduced(red_p, layer)
        src, tol = "reduced", 1e-4   # cluster numpy rank-AUC vs sklearn: allow tiny drift
    else:
        return None
    if abs(recomputed - stored) > tol:
        raise SystemExit(f"[gate] {slug} L{layer} ({src}): tokens_code_auc {recomputed:.6f} != "
                         f"stored {stored:.6f} (diff {abs(recomputed-stored):.2e})")
    return reads, {"tokens_code_auc_recomputed": recomputed, "tokens_code_auc_stored": stored, "source": src}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-partial", action="store_true",
                    help="score only models whose logit npz is present (else hard-fail listing missing)")
    args = ap.parse_args()

    rows, test_eids = load_split(DATASET, SPLIT)
    true_label = {i: int(rows[i]["label"]) for i in range(len(rows))}
    test = sorted(test_eids)
    sub_kept = set(json.loads(SUBSET.read_text())["kept_eids"])
    subtractive = [e for e in test if e in sub_kept]   # canonical ADR-0004 subset ∩ test

    # presence check (Opus C2): a model is available via local npz OR cluster-reduced json
    def available(slug, layer):
        return (E16 / slug / f"logits_layer{layer:02d}.npz").exists() or \
               (E16 / slug / "lasttok_reduced.json").exists()
    missing = [slug for slug, (layer, _) in MODELS.items() if not available(slug, layer)]
    if missing and not args.allow_partial:
        raise SystemExit("[fatal] no logit npz or reduced json for: " + ", ".join(missing) +
                         "\n  fetch from cluster scratch or rerun with --allow-partial.")

    out_all = {"n_test": len(test), "n_test_subtractive": len(subtractive),
               "n_boot": N_BOOT, "missing": missing, "models": {}}
    for slug, (layer, verb_dir) in MODELS.items():
        loaded = load_reads_and_gate(slug, layer)
        if loaded is None:
            continue
        reads, gate = loaded

        # verbalized P(yes) on test, true label, + hard gate vs exp-17
        verb, verb_gate = {}, None
        if verb_dir is not None:
            vp = E17 / verb_dir / "example_scores_verbalized.json"
            if vp.exists():
                vd = {r["eid"]: float(r["p_yes"]) for r in json.loads(vp.read_text())}
                verb = {e: vd[e] for e in test if e in vd}
                if len(verb) != len(test):
                    raise SystemExit(f"[gate] {slug}: verbalized covers {len(verb)}/{len(test)} test eids")
                vrec = _auc([true_label[e] for e in test], [verb[e] for e in test])
                stored = json.loads((E17 / verb_dir / "metrics_verbalized_logits.json").read_text()).get("verbalized_auc_test")
                if stored is not None and abs(vrec - stored) > VERB_GATE_TOL:
                    raise SystemExit(f"[gate] {slug}: verbalized AUC {vrec:.4f} != stored {stored:.4f}")
                verb_gate = {"verbalized_auc_recomputed": vrec, "verbalized_auc_stored": stored}

        def eval_pool(pool):
            y = np.array([true_label[e] for e in pool])
            res = {"n": len(pool), "n_pos": int(y.sum())}
            for k in ("last_code_token", "max_pool", "mean_pool"):
                s = np.array([reads[e][k] for e in pool])
                res[k] = {"auc": _auc(y, s), "ci": _boot_ci(y, s)}
            if verb:                                   # probe & verbalized on identical eids (codex M4)
                sv = np.array([verb[e] for e in pool])
                res["verbalized"] = {"auc": _auc(y, sv), "ci": _boot_ci(y, sv)}
                res["paired_vs_verbalized"] = {
                    k: _paired_delta_ci(y, np.array([reads[e][k] for e in pool]), sv)
                    for k in ("last_code_token", "max_pool")}
            return res

        out_all["models"][slug] = {"layer": layer, "token_gate": gate, "verbalized_gate": verb_gate,
                                   "full_test": eval_pool(test),
                                   "subtractive_test": eval_pool(subtractive)}
        ft = out_all["models"][slug]["full_test"]
        vb = ft.get("verbalized", {}).get("auc", float("nan"))
        print(f"[{slug}] L{layer} full(n={ft['n']},pos={ft['n_pos']}): "
              f"last_tok={ft['last_code_token']['auc']:.3f} max={ft['max_pool']['auc']:.3f} "
              f"mean={ft['mean_pool']['auc']:.3f} verb={vb:.3f}  "
              f"[gate tok_auc={gate['tokens_code_auc_recomputed']:.4f}=={gate['tokens_code_auc_stored']:.4f}]",
              file=sys.stderr)

    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results/last_token_readout.json").write_text(json.dumps(out_all, indent=2))
    print(f"[done] {len(out_all['models'])} models -> results/last_token_readout.json"
          + (f"  (MISSING: {missing})" if missing else ""), file=sys.stderr)


if __name__ == "__main__":
    main()
