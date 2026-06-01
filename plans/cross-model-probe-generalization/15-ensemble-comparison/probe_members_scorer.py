# [ai-generated]
"""Exp-15: PROBE-side member scores for the symmetric ensemble-comparison matrix.

For one model at its single best layer on CACHED acts, this trains the PROBE
members of the symmetric matrix (the probe side; the verbalized side is the
14-.../prompt_variants_judge.py P(yes) per prompt) and dumps, FOR EACH TEST
EXAMPLE, the per-example MAX-POOLED sigmoid score of every probe member, per
seed. The 15-.../build_matrix.py combiner then assembles:

  general      : probe = pooled-ALL-positives probe            (member "general")
  memory       : probe = pooled-MEMORY-category probe          (member "memory")
  injection    : probe = pooled-INJECTION-category probe       (member "injection")
  ind-ensemble : MAX over the per-INDIVIDUAL-CWE probes        (members "cwe:<NNN>")
  cat-ensemble : MAX(memory, injection)                        (from the two above)

so this script only needs to emit the ATOMIC member scores; the ensembling (MAX
over members) lives in build_matrix.py where it mirrors the verbalized side
exactly.

RENAME NOTE (lead-confirmed): what exp-05/compare_belief_audit called the "family
probe" is here the "memory probe" (the pooled MEMORY category). We add the
"injection probe" (pooled INJECTION category). Collectively memory+injection are
the "category" probes. The per-CWE probes are the "individual" members.

COMPARABILITY CONTRACT — mirrors exp-05 compare_belief_audit.py EXACTLY so the
probe member AUCs are apples-to-apples with the belief audit's `family_auc` /
`verbalized_auc` columns and with the verbalized matrix:

  * SPLIT: 5 group-clean seeds via make_split_for_seed (loaded by file path from
    compare_belief_audit.py — same shuffle, same frac_heldout=0.2). A pair never
    straddles the train/test boundary (pair_group_key).
  * PER-SEED FIT (the choice — see TODO below): like compare_belief_audit, each
    probe member is FIT on the seed's FULL train pool (NO 15% VAL carve). This
    departs from exp-10 posthoc_ensemble (which carves a 15% VAL for a learned
    combiner) because the belief-audit reference does NOT carve, and the matrix
    must line up with THAT reference. We store per-seed example scores; the matrix
    averages example-AUC over seeds.
  * POOLING: per-token sigmoid -> example_scores (MAX-pool) -> one score per
    example. Verbalized is already one judgment per example, so both sides are
    example-level.
  * LOSS: alpha (default 1.0 to match compare_belief_audit's CLI default) +
    neg_incl=False. TODO(adhoc-decision) below.

MEMBER FIT POOLS (test side is restricted to scored eids in build_matrix.py):
  general      : ALL train positives    u all train negatives.
  memory       : MEMORY train positives u all train negatives.
  injection    : INJECTION train pos    u all train negatives.
  cwe:<NNN>    : that CWE's train pos    u all train negatives  (one per CWE).
all with --neg-pool all (every cwe==null negative), matching exp-10's default so
the per-CWE / category probes are comparable to the existing per-CWE numbers.

DATA SCARCITY: per-CWE TEST positives are tiny for some CWEs (e.g. CWE-787 ~5,
CWE-190 ~4). Their individual probes are noisy and may HURT the MAX ind-ensemble.
We still TRAIN and DUMP every member; the low-n flagging happens in build_matrix
(n_test_pos < MIN_TRUST_POS), surfaced not hidden. A CWE with an empty train-pos
pool for a given seed is recorded with score=None for that seed's examples.

OUTPUT (resumable; skip if it exists):
  $WORK/runs/ensemble15_<slug>/probe_member_scores.json = {
    "model", "layer", "alpha", "seeds", "members": [...ordered...],
    "family_map": {CWE: family}, "min_trust_pos",
    "labels":  {eid: 0/1 example label},        # union over seeds' test sets
    "cwe":     {eid: "CWE-NNN" | null},
    "per_seed": {<seed>: {
        "test_eids": [...],
        "scores": {<member>: {eid: score | null}},  # MAX-pooled sigmoid, test eids
        "n_train_pos": {<member>: int},
        "n_test_pos":  {<member>: int},  # member's own positives among test eids
    }},
  }
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

# CWE -> family. Loaded VERBATIM (below) from compare_belief_audit.py so it never
# drifts; this literal is only a fallback/reference. memory = CWE-416/476/125/787;
# injection = CWE-089/078/022/079/190. (verbatim from 10-.../per_cwe_probe.py.)
FAMILY_FALLBACK = {
    "CWE-089": "injection", "CWE-078": "injection", "CWE-022": "injection",
    "CWE-079": "injection", "CWE-190": "injection",
    "CWE-125": "memory", "CWE-476": "memory", "CWE-416": "memory",
    "CWE-787": "memory",
}
FAMILIES = ("memory", "injection")
# Fixed per-CWE order (memory first, then injection) for deterministic member
# columns. Mirrors the verbalized V_cwe* order in prompt_variants_judge.py.
CWE_ORDER = ["CWE-416", "CWE-476", "CWE-125", "CWE-787",
             "CWE-089", "CWE-078", "CWE-022", "CWE-079", "CWE-190"]


def _load_compare_belief_audit():
    """[ai-generated] Load 05-.../compare_belief_audit.py by file path to REUSE its
    FAMILY map, MIN_TRUST_POS, FAMILIES and make_split_for_seed VERBATIM. Loading
    by path (not import) keeps definitions in lockstep with the belief audit; if
    its family assignment or split shuffle changes, this scorer inherits it."""
    p = (REPO / "plans" / "cross-model-probe-generalization"
         / "05-probe-vs-verbalized" / "compare_belief_audit.py")
    if not p.exists():
        raise SystemExit(f"[ens15] cannot find compare_belief_audit.py at {p}")
    spec = importlib.util.spec_from_file_location("compare_belief_audit", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["compare_belief_audit"] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_train_eval():
    p = REPO / "src" / "remotes" / "the cluster" / "train_eval.py"
    spec = importlib.util.spec_from_file_location("remote_train_eval", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tok_probs(probe_result, X):
    """Per-token sigmoid prob from a train_one_layer LINEAR result (w, b). Never
    mutates X. Mirrors compare_belief_audit._tok_probs."""
    w = np.asarray(probe_result["w"], np.float32)
    b = float(probe_result["b"])
    return 1.0 / (1.0 + np.exp(-(np.asarray(X, np.float32) @ w + b)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--best-layer", type=int, required=True)
    ap.add_argument("--out", required=True,
                    help="OUTPUT FILE probe_member_scores.json")
    ap.add_argument("--model", default="")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    # TODO(adhoc-decision): alpha default 1.0 mirrors compare_belief_audit's CLI
    # default so the probe member example-AUCs line up with the belief-audit
    # reference. exp-10 posthoc uses train_one_layer's default 10.0; the lead picks
    # which the matrix should match. We match the EXAMPLE-level belief reference.
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        print(f"[ens15] {out} exists, skipping", file=sys.stderr)
        return

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cba = _load_compare_belief_audit()
    te_mod = _load_train_eval()
    family_map = dict(getattr(cba, "FAMILY", FAMILY_FALLBACK))
    min_trust_pos = int(getattr(cba, "MIN_TRUST_POS", 10))
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    layer = args.best_layer

    acts = Path(args.acts_dir)
    Xfull = np.asarray(np.load(acts / f"layer_{layer:02d}.npy", mmap_mode="r"),
                       dtype=np.float32)
    y = np.load(acts / "y.npy")
    eids = np.load(acts / "example_ids.npy")
    if not np.isfinite(Xfull).all():
        raise SystemExit(f"[ens15] non-finite activations at layer {layer}")
    acts_eids = set(int(e) for e in np.unique(eids))

    rows = [json.loads(l) for l in Path(args.dataset).open() if l.strip()]
    eid_to_group = {i: te_mod.pair_group_key(r) for i, r in enumerate(rows)}

    def cwe_of(e):
        return rows[int(e)].get("cwe")

    def family_of(e):
        return family_map.get(cwe_of(e))

    # CWEs that actually appear (intersection of the fixed order with present CWEs).
    present_cwes = {cwe_of(e) for e in acts_eids if cwe_of(e)}
    cwe_members = [c for c in CWE_ORDER if c in present_cwes]
    missing_order = [c for c in CWE_ORDER if c not in present_cwes]
    if missing_order:
        print(f"[ens15] WARNING: CWE(s) in fixed order not present in data: "
              f"{missing_order}", file=sys.stderr)

    # Ordered member list: category members first, then per-CWE (memory then inj).
    members = ["general", "memory", "injection"] + [f"cwe:{c}" for c in cwe_members]
    print(f"[ens15] {args.model} L{layer} members={members}", file=sys.stderr)

    def fit_pool_for(member, tr_eids, neg_tr):
        """Train positives for a member u all train negatives. Returns
        (pos_tr_set, fit_eids_set)."""
        if member == "general":
            pos = {e for e in tr_eids if cwe_of(e) and e in acts_eids}
        elif member in FAMILIES:
            pos = {e for e in tr_eids if family_of(e) == member and e in acts_eids}
        elif member.startswith("cwe:"):
            c = member.split("cwe:", 1)[1]
            pos = {e for e in tr_eids if cwe_of(e) == c and e in acts_eids}
        else:
            raise ValueError(f"unknown member {member!r}")
        return pos, (pos | neg_tr)

    def member_test_pos(member, te_eids):
        """The member's OWN positives among the test eids (for n_test_pos / low-n
        flagging). general -> all positives; memory/injection -> family positives;
        cwe:X -> that CWE's positives."""
        if member == "general":
            return {e for e in te_eids if cwe_of(e) and e in acts_eids}
        if member in FAMILIES:
            return {e for e in te_eids if family_of(e) == member and e in acts_eids}
        c = member.split("cwe:", 1)[1]
        return {e for e in te_eids if cwe_of(e) == c and e in acts_eids}

    labels: dict[int, int] = {}
    cwe_by_eid: dict[int, "str | None"] = {}
    per_seed: dict[str, dict] = {}

    for seed in seeds:
        tr_eids, te_eids = cba.make_split_for_seed(eid_to_group, seed)
        tr_eids = {e for e in tr_eids if e in acts_eids}
        te_eids = {e for e in te_eids if e in acts_eids}
        neg_tr = {e for e in tr_eids if not cwe_of(e)}

        # Test-side flat token mask (scored once; every member is scored on it so
        # the per-example scores share the SAME example set per seed).
        te_mask = np.fromiter((int(e) in te_eids for e in eids), bool, len(eids))
        Xte = np.asarray(Xfull[te_mask], np.float32)
        te_tok_eids = eids[te_mask]

        # Example-level labels for this seed's test eids (max>0 over the eid's tokens).
        ex_ids_ref, _ = te_mod.example_scores(
            np.zeros(te_mask.sum(), np.float32), te_tok_eids)
        for e in ex_ids_ref:
            ei = int(e)
            labels[ei] = int(y[(eids == e)].max() > 0)
            cwe_by_eid[ei] = cwe_of(ei)

        seed_scores: dict[str, dict] = {}
        n_train_pos: dict[str, int] = {}
        n_test_pos: dict[str, int] = {}

        for member in members:
            pos_tr, fit_eids = fit_pool_for(member, tr_eids, neg_tr)
            n_train_pos[member] = len(pos_tr)
            n_test_pos[member] = len(member_test_pos(member, te_eids))

            # Empty train-pos pool -> cannot fit; record None for every test eid.
            if not pos_tr or not neg_tr:
                seed_scores[member] = {str(int(e)): None for e in ex_ids_ref}
                print(f"[ens15] seed {seed} {member}: empty pos/neg train pool "
                      f"-> scores=None", file=sys.stderr)
                continue

            fit_mask = np.fromiter((int(e) in fit_eids for e in eids),
                                   bool, len(eids))
            res = train_one_layer(
                np.asarray(Xfull[fit_mask], np.float32),
                y[fit_mask], eids[fit_mask],
                epochs=args.epochs, device=device, verbose=False,
                alpha=args.alpha, neg_incl=False)
            tok_p = _tok_probs(res, Xte)
            ex_ids, ex_p = te_mod.example_scores(tok_p, te_tok_eids)
            seed_scores[member] = {str(int(e)): float(p)
                                   for e, p in zip(ex_ids, ex_p)}
            print(f"[ens15] seed {seed} {member}: n_train_pos={len(pos_tr)} "
                  f"n_test_pos={n_test_pos[member]} scored={len(ex_ids)}",
                  file=sys.stderr)

        per_seed[str(seed)] = {
            "test_eids": [int(e) for e in ex_ids_ref],
            "scores": seed_scores,
            "n_train_pos": n_train_pos,
            "n_test_pos": n_test_pos,
        }

    record = {
        "model": args.model,
        "layer": layer,
        "alpha": args.alpha,
        "epochs": args.epochs,
        "seeds": seeds,
        "members": members,
        "family_map": family_map,
        "min_trust_pos": min_trust_pos,
        "labels": {str(k): v for k, v in labels.items()},
        "cwe": {str(k): v for k, v in cwe_by_eid.items()},
        "per_seed": per_seed,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record))
    print(f"[ens15] wrote {out} ({len(members)} members x {len(seeds)} seeds)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
