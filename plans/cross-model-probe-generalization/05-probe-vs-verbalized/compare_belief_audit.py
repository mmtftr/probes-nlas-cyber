# [ai-generated]
"""Belief audit: per-CWE-family THREE-WAY comparison at the EXAMPLE level.

Extends exp-05 (compare_probe_vs_verbalized.py) from a single overall
probe-vs-verbalized number to a per-family three-way contrast. No model here:
reuses the cached per-layer activation memmaps (probe side, runs/layersweep_<slug>/
acts) and the precomputed verbalized P(yes) scores from verbalized_judge.py.

THE QUESTION (see BELIEF-AUDIT.md). Our headline finding is that the GENERAL
probe misses memory-safety vulns while a FAMILY-pooled probe RECOVERS them from
the activations — so the model REPRESENTS memory-vuln but the general probe
under-allocates. This audit asks whether the model's OWN verbalized judgment
("is this code vulnerable? yes/no") also misses memory-safety. Per CWE family we
compute the EXAMPLE-level AUC of three judges over (family positives ∪ all
negatives):

  1. GENERAL linear probe @ best layer   — misses memory (the under-allocation)
  2. FAMILY-pooled probe @ best layer     — recovers memory (signal IS in acts)
  3. verbalized P(yes) per example         — does the model REPORT it?

  family probe RECOVERS memory AND verbalized ALSO misses memory
      => genuine introspection gap (the probe reads what the model won't say).
  verbalized CATCHES memory
      => belief is promptable; the general probe's miss is pure capacity-
         allocation, not an introspection gap.

EXAMPLE-LEVEL EVERYWHERE. Verbalized is inherently one judgment per function, so
the probe sides max-pool code-token sigmoids to a per-example score (exactly like
exp-05). All AUCs are example-AUC over the SAME example ids — apples to apples.

WHAT IS COPIED FROM WHERE
  * make_split_for_seed              — verbatim from exp-05 / exp-02 splits.
  * FAMILY map + pooled-family fit   — verbatim from exp-10 per_cwe_probe.py
                                       (one pooled probe over ALL family positives,
                                       NOT a loop over member CWEs).
  * train_one_layer / example_scores / pair_group_key — the shared pipeline.

DATA SCARCITY. ~54 memory test positives at the EXAMPLE level give a WIDE CI on
the memory family AUC. We surface n_pos and a `trust` flag (n_pos >= MIN_TRUST_POS)
and DO NOT break memory down by individual CWE here — the pooled family is the
trustworthy unit.
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

LENGTH_BASELINE = 0.49  # SVEN before/after example-AUC length baseline (~chance).

# [ai-generated] FAMILY map — copied VERBATIM from exp-10 per_cwe_probe.py. The
# pooled-family probe trains on every positive whose CWE maps to the family, ∪ a
# negative pool. Keep this in lockstep with exp-10; if exp-10's map changes
# (e.g. the CWE-190 family call), mirror it here.
# TODO(adhoc-decision): CWE-190 family is debatable (see exp-10 docstring); the
# lead owns the assignment. This file does not re-decide it — it copies exp-10.
FAMILY = {
    "CWE-089": "injection", "CWE-078": "injection", "CWE-022": "injection",
    "CWE-079": "injection", "CWE-190": "injection",
    "CWE-125": "memory", "CWE-476": "memory", "CWE-416": "memory",
    "CWE-787": "memory",
}

# A family example-AUC below this many test positives is flagged untrustworthy.
# TODO(adhoc-decision): threshold is the lead's call; 10 mirrors exp-10.
MIN_TRUST_POS = 10

FAMILIES = ("memory", "injection")


def _load_train_eval():
    p = REPO / "src" / "remotes" / "train_eval.py"
    spec = importlib.util.spec_from_file_location("remote_train_eval", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _verbalized_question() -> "str | None":
    """[ai-generated] Read the QUESTION constant from the sibling
    verbalized_judge.py for provenance, robust to cwd (load by file path, no
    model deps). Returns None if it can't be read."""
    p = Path(__file__).parent / "verbalized_judge.py"
    if not p.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location("_vj_for_question", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return getattr(mod, "QUESTION", None)
    except Exception:  # noqa: BLE001 — provenance only; never sink the compare
        return None


def make_split_for_seed(eid_to_group, seed, frac_heldout=0.2):
    """Group-clean held-out split for a seed. Copied VERBATIM from exp-05 /
    exp-02 splits_variance.py — a pair never straddles the train/test boundary."""
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


def _auc(y, s):
    return (float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan"))


def _tok_probs(probe_result, X):
    """Per-token sigmoid prob from a train_one_layer result (linear head)."""
    w = np.asarray(probe_result["w"], np.float32)
    b = float(probe_result["b"])
    return 1.0 / (1.0 + np.exp(-(X @ w + b)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--scores-glob", required=True,
                    help="directory holding verbalized_scores.gpu*.json")
    ap.add_argument("--layer", type=int, required=True,
                    help="model's honest val_tokens_code best layer")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="")
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

    p_yes, _vlab = merge_verbalized(Path(args.scores_glob))
    print(f"[belief] merged {len(p_yes)} verbalized scores", file=sys.stderr)

    Xfull = np.asarray(np.load(acts / f"layer_{layer:02d}.npy", mmap_mode="r"),
                       dtype=np.float32)
    y = np.load(acts / "y.npy")
    eids = np.load(acts / "example_ids.npy")
    if not np.isfinite(Xfull).all():
        raise SystemExit(f"non-finite activations at layer {layer}")

    rows = [json.loads(l) for l in Path(args.dataset).open() if l.strip()]
    eid_to_group = {i: te_mod.pair_group_key(r) for i, r in enumerate(rows)}

    def cwe_of(e):  # row index == eid (same convention as exp-10)
        return rows[int(e)].get("cwe")

    def family_of(e):
        return FAMILY.get(cwe_of(e))

    # eids present in the activation arrays (a row may be missing if extraction
    # truncated/skipped it). Restrict everything to these.
    acts_eids = set(int(e) for e in np.unique(eids))

    # --- per-seed accumulation ---
    # overall_per_seed: the exp-05 overall probe-vs-verbalized number, preserved.
    # fam_per_seed[fam]: list of {general_auc, family_auc, verbalized_auc, n_pos, ...}
    overall_per_seed = []
    fam_per_seed = {f: [] for f in FAMILIES}
    seed42_arrays = {}

    for seed in seeds:
        tr_eids, te_eids = make_split_for_seed(eid_to_group, seed)
        tr = np.fromiter((int(e) in tr_eids for e in eids), bool, len(eids))
        te = ~tr
        ytr, etr = y[tr], eids[tr]
        if len(np.unique(ytr)) < 2 or te.sum() == 0:
            overall_per_seed.append({"seed": seed, "skipped": "degenerate"})
            continue

        # --- GENERAL probe: trained ONCE on ALL train tokens (the exp-05 / 06 baseline) ---
        gen_res = train_one_layer(Xfull[tr], ytr, etr, epochs=args.epochs,
                                  device=device, verbose=False, alpha=args.alpha,
                                  neg_incl=False)
        Xte = Xfull[te]
        tok_y, te_tok_eids = y[te], eids[te]
        gen_tok_p = _tok_probs(gen_res, Xte)
        gen_ex_ids, gen_ex_p = te_mod.example_scores(gen_tok_p, te_tok_eids)
        ex_y = np.array([int(y[(eids == e)].max() > 0) for e in gen_ex_ids])
        gen_score_by_eid = {int(e): float(p) for e, p in zip(gen_ex_ids, gen_ex_p)}
        exy_by_eid = {int(e): int(v) for e, v in zip(gen_ex_ids, ex_y)}

        # --- OVERALL probe vs verbalized (exp-05 number, intersection of scored) ---
        keep = np.array([int(e) in p_yes for e in gen_ex_ids], dtype=bool)
        ov_ids = gen_ex_ids[keep]
        ov_probe = gen_ex_p[keep]
        ov_y = ex_y[keep]
        ov_verb = np.array([p_yes[int(e)] for e in ov_ids], dtype=float)
        probe_auc = _auc(ov_y, ov_probe)
        verb_auc = _auc(ov_y, ov_verb)
        overall_per_seed.append({
            "seed": seed, "probe_auc": probe_auc, "verbalized_auc": verb_auc,
            "delta": probe_auc - verb_auc, "n_test_ex": int(len(ov_ids)),
        })
        print(f"[belief] seed {seed} OVERALL: probe={probe_auc:.3f} "
              f"verbalized={verb_auc:.3f} delta={probe_auc - verb_auc:+.3f} "
              f"(n={len(ov_ids)})", file=sys.stderr)

        # --- per-family three-way ---
        # negatives in train/test (cwe == null), restricted to acts + scored eids.
        neg_tr = {e for e in tr_eids if not cwe_of(e) and e in acts_eids}
        neg_te = {e for e in te_eids if not cwe_of(e) and e in acts_eids}

        for fam in FAMILIES:
            # Family-pooled probe FIT pool: every train positive in the family ∪
            # all negatives (exactly exp-10's --cwe memory|injection with
            # --neg-pool all). max-pool philosophy carries to example level here.
            pos_tr = {e for e in tr_eids if family_of(e) == fam and e in acts_eids}
            pos_te = {e for e in te_eids if family_of(e) == fam and e in acts_eids}
            # Eval example set: family positives ∪ all negatives (test side).
            eval_eids = pos_te | neg_te
            # Intersect with eids the verbalized side actually scored, so all three
            # judges run on the IDENTICAL example set.
            eval_eids = {e for e in eval_eids if e in p_yes}
            n_pos = sum(1 for e in eval_eids if e in pos_te)
            rec = {
                "seed": seed,
                "n_train_pos": len(pos_tr),
                "n_test_pos": n_pos,
                "n_neg_test": sum(1 for e in eval_eids if e in neg_te),
                "trust": n_pos >= MIN_TRUST_POS,
            }
            if not pos_tr or n_pos == 0 or rec["n_neg_test"] == 0:
                rec["error"] = "empty positive or negative pool in fit/eval"
                fam_per_seed[fam].append(rec)
                continue

            # --- family-pooled probe ---
            spec_fit_eids = pos_tr | neg_tr
            spec_fit_mask = np.fromiter((int(e) in spec_fit_eids for e in eids),
                                        bool, len(eids))
            spec_res = train_one_layer(
                np.asarray(Xfull[spec_fit_mask], np.float32),
                y[spec_fit_mask], eids[spec_fit_mask],
                epochs=args.epochs, device=device, verbose=False,
                alpha=args.alpha, neg_incl=False)
            # Score the family eval subset (tokens) and max-pool to example level.
            eval_mask = np.fromiter((int(e) in eval_eids for e in eids),
                                    bool, len(eids))
            Xev = np.asarray(Xfull[eval_mask], np.float32)
            ev_tok_eids = eids[eval_mask]
            spec_tok_p = _tok_probs(spec_res, Xev)
            spec_ex_ids, spec_ex_p = te_mod.example_scores(spec_tok_p, ev_tok_eids)

            # Align the three judges on the SAME ordered eval example ids.
            ev_ids = spec_ex_ids
            ev_y = np.array([exy_by_eid.get(int(e),
                             int(y[(eids == e)].max() > 0)) for e in ev_ids])
            fam_probe_score = spec_ex_p
            # GENERAL probe was trained on all tokens; reuse its per-example score
            # over this family subset (same general probe, restricted eval set).
            gen_fam_score = np.array(
                [gen_score_by_eid[int(e)] for e in ev_ids], dtype=float)
            verb_fam_score = np.array(
                [p_yes[int(e)] for e in ev_ids], dtype=float)

            g_auc = _auc(ev_y, gen_fam_score)
            f_auc = _auc(ev_y, fam_probe_score)
            v_auc = _auc(ev_y, verb_fam_score)
            rec.update({
                "general_auc": g_auc,
                "family_auc": f_auc,
                "verbalized_auc": v_auc,
                "n_eval_ex": int(len(ev_ids)),
            })
            fam_per_seed[fam].append(rec)
            print(f"[belief] seed {seed} {fam}: general={g_auc:.3f} "
                  f"family={f_auc:.3f} verbalized={v_auc:.3f} "
                  f"(n_pos={n_pos} trust={rec['trust']})", file=sys.stderr)

            if seed == 42:
                seed42_arrays.setdefault(fam, {
                    "ex_ids": [int(e) for e in ev_ids],
                    "ex_label": [int(v) for v in ev_y],
                    "general_score": [float(v) for v in gen_fam_score],
                    "family_score": [float(v) for v in fam_probe_score],
                    "p_yes": [float(v) for v in verb_fam_score],
                })

    # --- aggregate overall ---
    valid_ov = [s for s in overall_per_seed if "probe_auc" in s]
    probe_m, probe_s = _ms([s["probe_auc"] for s in valid_ov])
    verb_m, verb_s = _ms([s["verbalized_auc"] for s in valid_ov])
    delta_m, delta_s = _ms([s["delta"] for s in valid_ov])

    # --- aggregate per family (mean/std across seeds) ---
    families_out = {}
    for fam in FAMILIES:
        seeds_ok = [s for s in fam_per_seed[fam] if "general_auc" in s]
        g_m, g_s = _ms([s["general_auc"] for s in seeds_ok])
        f_m, f_s = _ms([s["family_auc"] for s in seeds_ok])
        v_m, v_s = _ms([s["verbalized_auc"] for s in seeds_ok])
        n_pos_vals = [s["n_test_pos"] for s in fam_per_seed[fam] if "n_test_pos" in s]
        n_pos_med = int(np.median(n_pos_vals)) if n_pos_vals else 0
        families_out[fam] = {
            "general_auc_mean": g_m, "general_auc_std": g_s,
            "family_auc_mean": f_m, "family_auc_std": f_s,
            "verbalized_auc_mean": v_m, "verbalized_auc_std": v_s,
            "delta_family_minus_general":
                (f_m - g_m) if (f_m is not None and g_m is not None) else None,
            "delta_family_minus_verbalized":
                (f_m - v_m) if (f_m is not None and v_m is not None) else None,
            "n_test_pos_median": n_pos_med,
            "trust": n_pos_med >= MIN_TRUST_POS,
            "per_seed": fam_per_seed[fam],
        }

    record = {
        "model": args.model,
        "layer": layer,
        "alpha": args.alpha,
        "epochs": args.epochs,
        "seeds": seeds,
        "question": _verbalized_question(),
        "min_trust_pos": MIN_TRUST_POS,
        "length_baseline": LENGTH_BASELINE,
        "n_examples_scored": len(p_yes),
        # Overall exp-05 probe-vs-verbalized (preserved).
        "overall": {
            "probe_auc_mean": probe_m, "probe_auc_std": probe_s,
            "verbalized_auc_mean": verb_m, "verbalized_auc_std": verb_s,
            "delta_mean": delta_m, "delta_std": delta_s,
            "per_seed": overall_per_seed,
        },
        # Per-family three-way (the belief audit).
        "families": families_out,
        "seed42_arrays": seed42_arrays,
    }
    Path(args.out).write_text(json.dumps(record, indent=2))

    print(f"[belief] OVERALL L{layer:02d}  probe={probe_m:.3f}±{probe_s:.3f}  "
          f"verbalized={verb_m:.3f}±{verb_s:.3f}  "
          f"delta={delta_m:+.3f}±{delta_s:.3f}", file=sys.stderr)
    for fam in FAMILIES:
        fo = families_out[fam]
        if fo["general_auc_mean"] is None:
            print(f"[belief] {fam}: no valid seed (empty pool?)", file=sys.stderr)
            continue
        g, f, v = fo["general_auc_mean"], fo["family_auc_mean"], fo["verbalized_auc_mean"]
        # Verdict: did the family probe recover signal the general probe + the
        # verbalized self-report both miss? (the introspection gap).
        recovers = (f is not None and g is not None and f - g > fo["family_auc_std"])
        verb_misses = (v is not None and v < 0.55)  # ~chance-ish self-report
        if recovers and verb_misses:
            verdict = "INTROSPECTION GAP (family recovers; verbalized misses)"
        elif v is not None and v >= 0.60:
            verdict = "belief PROMPTABLE (verbalized catches it)"
        else:
            verdict = "inconclusive"
        print(f"[belief] {fam}: general={g:.3f} family={f:.3f} verbalized={v:.3f} "
              f"n_pos~{fo['n_test_pos_median']} trust={fo['trust']} -> {verdict}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
