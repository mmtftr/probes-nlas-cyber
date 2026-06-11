# [ai-generated]
"""Analyze the prompt-sensitivity sweep: per-variant memory + injection example-AUC.

CPU-ONLY (no model, no cached acts). For each model's promptsweep run dir and each
prompt variant, merge the per-example p_yes shards
(variant_<id>.gpu*.json) and compute:
  * MEMORY    example-AUC over (memory-family positives ∪ all negatives)
  * INJECTION example-AUC over (injection-family positives ∪ all negatives)
both averaged over the SAME 5-seed group-clean splits the belief audit uses.

WHAT IS REUSED (loaded by file path from the sibling 05-.../compare_belief_audit.py
so this NEVER drifts from the belief audit's definitions):
  * FAMILY map               — CWE -> {memory, injection}
  * MIN_TRUST_POS            — n_pos trust threshold
  * make_split_for_seed      — group-clean per-seed train/test split

and from src/remotes/train_eval.py:
  * pair_group_key           — vuln/fix pairs never straddle the boundary

Crucially this is the SAME per-family example-AUC the belief audit computes for
its `verbalized_auc` column — except there is NO probe here. Verbalized P(yes) is
already one judgment per example, so we score the (family-pos ∪ all-neg) TEST set
directly. We restrict the eval set to negatives + family-positives present in the
SCORED eids (a shard gap can never silently change the denominator), exactly as
compare_belief_audit does with `e in p_yes`.

V0_generic is the reference column (== belief-audit prompt). The printed table
shows every variant's memory/injection AUC AND the delta vs V0, per model.

Output: runs/promptsweep_<slug>_aucs.json =
  {model, run_dir, question_by_variant, seeds, variants: {
      <id>: {question, memory_auc_mean/std, injection_auc_mean/std,
             n_mem_pos (median over seeds), n_inj_pos (median over seeds),
             memory_trust, injection_trust, per_seed: {memory: [...], injection: [...]}}}}
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


def _load_compare_belief_audit():
    """[ai-generated] Load the sibling 05-.../compare_belief_audit.py by file path
    to REUSE its FAMILY map, MIN_TRUST_POS, FAMILIES and make_split_for_seed
    verbatim. Loading by path (not import) keeps definitions in lockstep with the
    belief audit; if its family assignment changes, this analysis inherits it."""
    p = (REPO / "plans" / "cross-model-probe-generalization"
         / "05-probe-vs-verbalized" / "compare_belief_audit.py")
    if not p.exists():
        raise SystemExit(f"[promptsweep] cannot find compare_belief_audit.py at {p}")
    spec = importlib.util.spec_from_file_location("compare_belief_audit", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["compare_belief_audit"] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_train_eval():
    p = REPO / "src" / "remotes" / "train_eval.py"
    spec = importlib.util.spec_from_file_location("remote_train_eval", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def merge_variant(scores_dir: Path, variant_id: str):
    """[ai-generated] Concatenate variant_<id>.gpu*.json -> {eid: p_yes}, {eid: label}.
    Mirrors compare_belief_audit.merge_verbalized but for a single variant's shard
    set. Errors loudly if no shard is present for the variant."""
    p_yes, lab = {}, {}
    files = sorted(scores_dir.glob(f"variant_{variant_id}.gpu*.json"))
    if not files:
        raise FileNotFoundError(
            f"no variant_{variant_id}.gpu*.json under {scores_dir}")
    for f in files:
        for rec in json.loads(f.read_text()):
            e = int(rec["eid"])
            p_yes[e] = float(rec["p_yes"])
            lab[e] = int(rec["label"])
    return p_yes, lab


def _ms(vals):
    a = np.array([v for v in vals if v is not None and v == v], dtype=float)
    return (float(a.mean()), float(a.std(ddof=0))) if a.size else (None, None)


def _auc(y, s):
    return (float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan"))


def variant_family_aucs(p_yes, rows, eid_to_group, family_of, cwe_of,
                        seeds, families, make_split, min_trust_pos):
    """[ai-generated] Per-family verbalized example-AUC over the 5-seed splits.

    For each seed, the per-family eval set is (family positives ∪ all negatives)
    on the TEST side, restricted to eids present in `p_yes` — IDENTICAL to
    compare_belief_audit's per-family eval-set construction (just without the
    probe judges). Label is the row's example label (max>0 collapses to the row's
    own label since verbalized is one judgment per row).

    Returns {family: {auc_mean, auc_std, n_pos_median, trust, per_seed: [...]}}.
    """
    scored = set(p_yes)
    out = {f: {"per_seed": [], "_n_pos": []} for f in families}
    for seed in seeds:
        _tr_eids, te_eids = make_split(eid_to_group, seed)
        # All negatives (cwe == null/empty) on the test side, restricted to scored.
        neg_te = {e for e in te_eids if not cwe_of(e) and e in scored}
        for fam in families:
            pos_te = {e for e in te_eids
                      if family_of(e) == fam and e in scored}
            eval_eids = sorted(pos_te | neg_te)
            n_pos = len(pos_te)
            n_neg = len(neg_te)
            rec = {"seed": seed, "n_test_pos": n_pos, "n_neg_test": n_neg,
                   "trust": n_pos >= min_trust_pos}
            out[fam]["_n_pos"].append(n_pos)
            if n_pos == 0 or n_neg == 0:
                rec["error"] = "empty positive or negative pool"
                out[fam]["per_seed"].append(rec)
                continue
            y = np.array([1 if e in pos_te else 0 for e in eval_eids])
            s = np.array([p_yes[e] for e in eval_eids], dtype=float)
            rec["auc"] = _auc(y, s)
            rec["n_eval_ex"] = len(eval_eids)
            out[fam]["per_seed"].append(rec)

    res = {}
    for fam in families:
        aucs = [r["auc"] for r in out[fam]["per_seed"] if "auc" in r]
        m, sd = _ms(aucs)
        n_pos_vals = out[fam]["_n_pos"]
        n_pos_med = int(np.median(n_pos_vals)) if n_pos_vals else 0
        res[fam] = {
            "auc_mean": m, "auc_std": sd,
            "n_pos_median": n_pos_med,
            "trust": n_pos_med >= min_trust_pos,
            "per_seed": out[fam]["per_seed"],
        }
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True,
                    help="directory holding variant_<id>.gpu*.json shards")
    ap.add_argument("--dataset", required=True, help="dataset.jsonl")
    ap.add_argument("--out", required=True, help="output aucs json path")
    ap.add_argument("--model", default="")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    args = ap.parse_args()

    cba = _load_compare_belief_audit()
    te_mod = _load_train_eval()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    rows = [json.loads(l) for l in Path(args.dataset).open() if l.strip()]
    eid_to_group = {i: te_mod.pair_group_key(r) for i, r in enumerate(rows)}

    def cwe_of(e):
        return rows[int(e)].get("cwe")

    def family_of(e):
        return cba.FAMILY.get(cwe_of(e))

    run_dir = Path(args.run_dir)
    # Discover variant ids present as shards (order-preserved from the judge's
    # canonical list if it is importable; else lexical from the filesystem).
    try:
        pvj = importlib.util.spec_from_file_location(
            "prompt_variants_judge", Path(__file__).parent / "prompt_variants_judge.py")
        pmod = importlib.util.module_from_spec(pvj)
        sys.modules["prompt_variants_judge_for_ids"] = pmod
        pvj.loader.exec_module(pmod)
        variant_ids = [v["id"] for v in pmod.PROMPT_VARIANTS]
        question_by_variant = {v["id"]: v["question"] for v in pmod.PROMPT_VARIANTS}
    except Exception:  # noqa: BLE001 — fall back to filesystem discovery
        variant_ids, question_by_variant = [], {}
        for f in sorted(run_dir.glob("variant_*.gpu*.json")):
            vid = f.stem.split(".gpu")[0].replace("variant_", "")
            if vid not in variant_ids:
                variant_ids.append(vid)

    if not variant_ids:
        raise SystemExit(f"[promptsweep] no variant_*.gpu*.json under {run_dir}")

    variants_out = {}
    for vid in variant_ids:
        try:
            p_yes, _lab = merge_variant(run_dir, vid)
        except FileNotFoundError as e:
            print(f"[promptsweep] {vid}: SKIP — {e}", file=sys.stderr)
            continue
        fam_aucs = variant_family_aucs(
            p_yes, rows, eid_to_group, family_of, cwe_of,
            seeds, cba.FAMILIES, cba.make_split_for_seed, cba.MIN_TRUST_POS)
        mem, inj = fam_aucs["memory"], fam_aucs["injection"]
        variants_out[vid] = {
            "question": question_by_variant.get(vid),
            "n_examples_scored": len(p_yes),
            "memory_auc_mean": mem["auc_mean"], "memory_auc_std": mem["auc_std"],
            "injection_auc_mean": inj["auc_mean"], "injection_auc_std": inj["auc_std"],
            "n_mem_pos": mem["n_pos_median"], "n_inj_pos": inj["n_pos_median"],
            "memory_trust": mem["trust"], "injection_trust": inj["trust"],
            "per_seed": {"memory": mem["per_seed"], "injection": inj["per_seed"]},
        }
        print(f"[promptsweep] {vid}: memory={_fmt(mem['auc_mean'], mem['auc_std'])} "
              f"(n_pos~{mem['n_pos_median']} trust={mem['trust']})  "
              f"injection={_fmt(inj['auc_mean'], inj['auc_std'])} "
              f"(n_pos~{inj['n_pos_median']} trust={inj['trust']})", file=sys.stderr)

    record = {
        "model": args.model,
        "run_dir": str(run_dir),
        "seeds": seeds,
        "min_trust_pos": cba.MIN_TRUST_POS,
        "family_map": cba.FAMILY,
        "question_by_variant": question_by_variant,
        "variants": variants_out,
    }
    Path(args.out).write_text(json.dumps(record, indent=2))
    print(f"[promptsweep] wrote {args.out}", file=sys.stderr)

    _print_table(args.model, variants_out)


def _fmt(m, s):
    if m is None:
        return "  n/a "
    return f"{m:.3f}±{s:.3f}" if s is not None else f"{m:.3f}"


def _print_table(model, variants_out):
    """[ai-generated] variant × {memory, injection} AUC, vs V0_generic reference."""
    print("", file=sys.stderr)
    print(f"=== prompt sweep: {model or '(unnamed model)'} ===", file=sys.stderr)
    print(f"{'variant':<20} {'memory AUC':>14} {'Δmem vs V0':>12} "
          f"{'injection AUC':>16} {'Δinj vs V0':>12}", file=sys.stderr)
    ref = variants_out.get("V0_generic", {})
    ref_mem = ref.get("memory_auc_mean")
    ref_inj = ref.get("injection_auc_mean")
    for vid, v in variants_out.items():
        m = v["memory_auc_mean"]
        i = v["injection_auc_mean"]
        dm = (f"{m - ref_mem:+.3f}" if (m is not None and ref_mem is not None
                                        and vid != "V0_generic") else " (ref)"
              if vid == "V0_generic" else "   n/a")
        di = (f"{i - ref_inj:+.3f}" if (i is not None and ref_inj is not None
                                        and vid != "V0_generic") else " (ref)"
              if vid == "V0_generic" else "   n/a")
        print(f"{vid:<20} {_fmt(m, v['memory_auc_std']):>14} {dm:>12} "
              f"{_fmt(i, v['injection_auc_std']):>16} {di:>12}", file=sys.stderr)
    print("(memory ↑ from ~0.4 toward ~0.7+ under a memory-specific prompt => the "
          "verbalized blind spot is a framing artifact; check if injection ↓.)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
