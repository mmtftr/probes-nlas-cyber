# [ai-generated]
"""Exp-15: build the SYMMETRIC probe-vs-verbalized specialization matrix (CPU-only).

NO model, NO cached acts. Reads two precomputed per-example score sources and
assembles a symmetric matrix comparing PROBE-side specialization against
VERBALIZED-side specialization at the EXAMPLE level (one score per function):

  side    ∈ {probe, verbalized}
  member  ∈ {general, memory, injection, ind-ensemble, cat-ensemble}
  cell    ∈ {memory, injection, overall}

The verbalized analogue of a specialized PROBE is a specialized PROMPT:

  member       | probe                              | verbalized
  -------------|------------------------------------|------------------------------
  general      | pooled-ALL-positives probe         | V0_generic prompt
  memory       | pooled-MEMORY-category probe       | V1_memory prompt
  injection    | pooled-INJECTION-category probe    | V_injection prompt
  ind-ensemble | MAX over per-INDIVIDUAL-CWE probes | MAX over per-CWE prompts
  cat-ensemble | MAX(memory probe, injection probe) | MAX(V1_memory, V_injection)

COMBINE RULE = MAX over members (lead-confirmed), parameterised via --combine
{max,mean} (default max). TODO(adhoc-decision): MAX is the lead's call; 'mean' is
offered for a robustness check.

ind-ensemble is FAMILY-AWARE per cell (lead-confirmed):
  * memory cell    -> MAX over the 4 MEMORY CWE members only.
  * injection cell -> MAX over the 5 INJECTION CWE members only.
  * overall cell   -> MAX over ALL 9 CWE members.
This holds for BOTH sides (per-CWE probes and per-CWE prompts).

EVAL (per cell), mirrors 05-/compare_belief_audit + 14-/analyze_prompt_sweep:
  * memory cell    -> (MEMORY positives ∪ ALL negatives) on the TEST side.
  * injection cell -> (INJECTION positives ∪ ALL negatives) on the TEST side.
  * overall cell   -> (ALL positives ∪ ALL negatives) on the TEST side.
restricted to eids present on BOTH score sources (so a missing verbalized shard
or an unscored probe eid can never silently change the denominator). Example-AUC,
averaged over the 5 group-clean seeds (make_split_for_seed, loaded from
compare_belief_audit so the split shuffle is IDENTICAL to the probe-member scorer
and the belief audit).

DATA SCARCITY: per-CWE test positives are tiny for some CWEs (CWE-787 ~5, CWE-190
~4). The matrix reports, per cell, n_pos (median over seeds) and a `trust` flag
(n_pos >= MIN_TRUST_POS); for ind-ensemble it ALSO lists the per-CWE member
n_test_pos with a `low_n` flag (n_test_pos < MIN_TRUST_POS) — flagged, NOT hidden.
A low-n noisy member can DRAG DOWN the MAX, so the flags explain a weak
ind-ensemble cell.

Output: ensemble15_<slug>_matrix.json (means±std, n_pos, low-n flags) + a readable
table printed to stderr.
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

FAMILIES = ("memory", "injection")
CELLS = ("memory", "injection", "overall")
MEMBERS = ("general", "memory", "injection", "ind-ensemble", "cat-ensemble")

# member -> verbalized variant id (single-prompt members). ind-ensemble and
# cat-ensemble are COMPOSED below from these + the per-CWE prompts.
VERB_SINGLE = {
    "general": "V0_generic",
    "memory": "V1_memory",
    "injection": "V_injection",
}
# CWE -> verbalized per-CWE variant id (the ind-ensemble members, verbalized side).
VERB_CWE = {
    "CWE-416": "V_cwe416", "CWE-476": "V_cwe476", "CWE-125": "V_cwe125",
    "CWE-787": "V_cwe787", "CWE-089": "V_cwe089", "CWE-078": "V_cwe078",
    "CWE-022": "V_cwe022", "CWE-079": "V_cwe079", "CWE-190": "V_cwe190",
}


def _load_compare_belief_audit():
    """[ai-generated] Load 05-.../compare_belief_audit.py by file path to REUSE
    FAMILY, MIN_TRUST_POS, FAMILIES, make_split_for_seed VERBATIM — same split
    shuffle as probe_members_scorer + the belief audit."""
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


def merge_variant(scores_dir: Path, variant_id: str):
    """[ai-generated] Concatenate variant_<id>.gpu*.json -> {eid: p_yes}. Mirrors
    14-/analyze_prompt_sweep.merge_variant. Returns {} if no shard exists (a member
    whose prompt was never scored is simply unavailable; surfaced, not crashed)."""
    p_yes: dict[int, float] = {}
    files = sorted(scores_dir.glob(f"variant_{variant_id}.gpu*.json"))
    for f in files:
        for rec in json.loads(f.read_text()):
            p_yes[int(rec["eid"])] = float(rec["p_yes"])
    return p_yes


def combine_scores(per_member_score_maps, eids, how):
    """[ai-generated] Combine several {eid: score} maps into one {eid: score} by
    rowwise MAX (or MEAN) over members, for the eids where AT LEAST ONE member has
    a score. A member missing an eid (None or absent) does not contribute; if no
    member scores an eid it is omitted (caller restricts to scored eids anyway).
    Never mutates inputs."""
    out: dict[int, float] = {}
    for e in eids:
        vals = []
        for m in per_member_score_maps:
            v = m.get(e)
            if v is not None and v == v:  # not None, not NaN
                vals.append(float(v))
        if not vals:
            continue
        out[e] = max(vals) if how == "max" else float(np.mean(vals))
    return out


def _ms(vals):
    a = np.array([v for v in vals if v is not None and v == v], dtype=float)
    return (float(a.mean()), float(a.std(ddof=0))) if a.size else (None, None)


def _auc(y, s):
    return (float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-scores", required=True,
                    help="probe_member_scores.json from probe_members_scorer.py")
    ap.add_argument("--promptsweep-dir", required=True,
                    help="dir with variant_<id>.gpu*.json (14-.../promptsweep_<slug>)")
    ap.add_argument("--dataset", required=True, help="dataset.jsonl")
    ap.add_argument("--out", required=True, help="ensemble15_<slug>_matrix.json")
    ap.add_argument("--model", default="")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    # TODO(adhoc-decision): MAX over members is lead-confirmed; 'mean' is a
    # robustness alternative. Held as a parameter so neither is silently baked in.
    ap.add_argument("--combine", choices=("max", "mean"), default="max")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        print(f"[ens15] {out} exists, skipping", file=sys.stderr)
        return

    cba = _load_compare_belief_audit()
    te_mod = _load_train_eval()
    family_map = dict(getattr(cba, "FAMILY"))
    min_trust_pos = int(getattr(cba, "MIN_TRUST_POS", 10))
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    rows = [json.loads(l) for l in Path(args.dataset).open() if l.strip()]
    eid_to_group = {i: te_mod.pair_group_key(r) for i, r in enumerate(rows)}

    def cwe_of(e):
        return rows[int(e)].get("cwe")

    def family_of(e):
        return family_map.get(cwe_of(e))

    # --- PROBE side: per-seed per-member {eid: score} ---
    pdoc = json.loads(Path(args.probe_scores).read_text())
    probe_members = pdoc["members"]            # ["general","memory","injection","cwe:...",...]
    probe_cwes = [m.split("cwe:", 1)[1] for m in probe_members if m.startswith("cwe:")]
    # probe_seed_scores[seed][member] = {eid:int -> score|None}
    probe_seed_scores: dict[int, dict] = {}
    for s, sd in pdoc["per_seed"].items():
        seed = int(s)
        probe_seed_scores[seed] = {
            m: {int(e): v for e, v in mp.items()}
            for m, mp in sd["scores"].items()
        }
    probe_n_test_pos = {int(s): sd["n_test_pos"] for s, sd in pdoc["per_seed"].items()}

    # --- VERBALIZED side: per-variant {eid: p_yes} (seed-independent; one read per row) ---
    psweep = Path(args.promptsweep_dir)
    verb_single = {m: merge_variant(psweep, vid) for m, vid in VERB_SINGLE.items()}
    verb_cwe = {c: merge_variant(psweep, vid) for c, vid in VERB_CWE.items()}
    verb_cwes_present = [c for c in VERB_CWE if verb_cwe.get(c)]
    missing_verb = [m for m, mp in verb_single.items() if not mp]
    if missing_verb:
        print(f"[ens15] WARNING: missing verbalized single-prompt members: "
              f"{missing_verb} (their cells will be n/a)", file=sys.stderr)
    missing_verb_cwe = [c for c in VERB_CWE if not verb_cwe.get(c)]
    if missing_verb_cwe:
        print(f"[ens15] WARNING: missing verbalized per-CWE prompts: "
              f"{missing_verb_cwe}", file=sys.stderr)

    def cwes_for_cell(cell):
        """CWEs whose members contribute to ind-ensemble for this cell."""
        if cell == "overall":
            return list(VERB_CWE.keys())
        return [c for c, f in family_map.items() if f == cell]

    def probe_member_map(member, seed, cell):
        """{eid: score} for a probe member on a given seed/cell (combining for the
        ensemble members)."""
        ss = probe_seed_scores[seed]
        if member in ("general", "memory", "injection"):
            return ss.get(member, {})
        if member == "cat-ensemble":
            maps = [ss.get("memory", {}), ss.get("injection", {})]
            eids = set().union(*[set(m) for m in maps]) if maps else set()
            return combine_scores(maps, eids, args.combine)
        if member == "ind-ensemble":
            cwes = [c for c in cwes_for_cell(cell) if f"cwe:{c}" in ss]
            maps = [ss[f"cwe:{c}"] for c in cwes]
            eids = set().union(*[set(m) for m in maps]) if maps else set()
            return combine_scores(maps, eids, args.combine)
        raise ValueError(member)

    def verb_member_map(member, cell):
        """{eid: p_yes} for a verbalized member on a given cell (seed-independent)."""
        if member in ("general", "memory", "injection"):
            return verb_single.get(member, {})
        if member == "cat-ensemble":
            maps = [verb_single.get("memory", {}), verb_single.get("injection", {})]
            eids = set().union(*[set(m) for m in maps]) if maps else set()
            return combine_scores(maps, eids, args.combine)
        if member == "ind-ensemble":
            cwes = [c for c in cwes_for_cell(cell) if verb_cwe.get(c)]
            maps = [verb_cwe[c] for c in cwes]
            eids = set().union(*[set(m) for m in maps]) if maps else set()
            return combine_scores(maps, eids, args.combine)
        raise ValueError(member)

    def cell_eval_eids(seed, cell, te_eids, scored):
        """(family/all positives ∪ all negatives) on the test side, restricted to
        `scored`. Mirrors compare_belief_audit's per-family eval-set construction
        and analyze_prompt_sweep's. overall = all positives ∪ all negatives."""
        neg = {e for e in te_eids if not cwe_of(e) and e in scored}
        if cell == "overall":
            pos = {e for e in te_eids if cwe_of(e) and e in scored}
        else:
            pos = {e for e in te_eids if family_of(e) == cell and e in scored}
        return pos, neg

    # --- accumulate per (side, member, cell) over seeds ---
    # acc[(side, member, cell)] = {"auc": [...per seed...], "n_pos": [...]}
    acc: dict[tuple, dict] = {}
    for side in ("probe", "verbalized"):
        for member in MEMBERS:
            for cell in CELLS:
                acc[(side, member, cell)] = {"auc": [], "n_pos": []}

    for seed in seeds:
        _tr, te_eids = cba.make_split_for_seed(eid_to_group, seed)
        te_eids = set(te_eids)

        for member in MEMBERS:
            for cell in CELLS:
                pmap = probe_member_map(member, seed, cell)
                vmap = verb_member_map(member, cell)
                # IDENTICAL eval example set for both sides of this (member, cell):
                # the eids BOTH sides scored. Keeps the probe-vs-verbalized contrast
                # apples-to-apples per cell (a missing prompt or unscored probe eid
                # can never shift the denominator for one side only).
                scored = _both_scored(pmap, vmap)
                pos, neg = cell_eval_eids(seed, cell, te_eids, scored)
                for side, score_map in (("probe", pmap), ("verbalized", vmap)):
                    rec = acc[(side, member, cell)]
                    rec["n_pos"].append(len(pos))
                    if not pos or not neg:
                        rec["auc"].append(None)
                        continue
                    eval_eids = sorted(pos | neg)
                    yv = np.array([1 if e in pos else 0 for e in eval_eids])
                    sv = np.array([score_map[e] for e in eval_eids], dtype=float)
                    rec["auc"].append(_auc(yv, sv))

    # --- assemble output matrix ---
    matrix: dict[str, dict] = {}
    for side in ("probe", "verbalized"):
        matrix[side] = {}
        for member in MEMBERS:
            matrix[side][member] = {}
            for cell in CELLS:
                rec = acc[(side, member, cell)]
                m, sd = _ms(rec["auc"])
                npv = [n for n in rec["n_pos"] if n is not None]
                n_pos_med = int(np.median(npv)) if npv else 0
                matrix[side][member][cell] = {
                    "auc_mean": m, "auc_std": sd,
                    "n_pos_median": n_pos_med,
                    "trust": n_pos_med >= min_trust_pos,
                }

    # --- per-CWE member low-n flags (ind-ensemble explainers), both sides ---
    ind_members = {}
    for c in VERB_CWE:
        fam = family_map.get(c)
        # probe n_test_pos median over seeds (member's own positives).
        probe_npos = [probe_n_test_pos[s].get(f"cwe:{c}") for s in seeds
                      if f"cwe:{c}" in probe_n_test_pos.get(s, {})]
        probe_npos = [n for n in probe_npos if n is not None]
        probe_med = int(np.median(probe_npos)) if probe_npos else 0
        ind_members[c] = {
            "family": fam,
            "probe_member_present": f"cwe:{c}" in probe_members,
            "verbalized_prompt_present": bool(verb_cwe.get(c)),
            "probe_n_test_pos_median": probe_med,
            "low_n": probe_med < min_trust_pos,
        }

    record = {
        "model": args.model,
        "combine": args.combine,
        "seeds": seeds,
        "min_trust_pos": min_trust_pos,
        "family_map": family_map,
        "probe_layer": pdoc.get("layer"),
        "probe_alpha": pdoc.get("alpha"),
        "member_to_prompt": {**VERB_SINGLE,
                             "ind-ensemble": f"MAX over {list(VERB_CWE.values())}",
                             "cat-ensemble": "MAX(V1_memory, V_injection)"},
        "probe_members_present": probe_members,
        "verbalized_cwes_present": verb_cwes_present,
        "missing_verbalized_single": missing_verb,
        "missing_verbalized_cwe": missing_verb_cwe,
        "ind_ensemble_members": ind_members,
        "matrix": matrix,
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(record, indent=2))
    print(f"[ens15] wrote {out}", file=sys.stderr)
    _print_matrix(args.model, args.combine, matrix, ind_members)


def _both_scored(map_a, map_b):
    """eids scored by BOTH maps (intersection of keys). Keeps probe and verbalized
    cells on the IDENTICAL eval example set so a cell-by-cell comparison is fair."""
    return set(map_a) & set(map_b)


def _fmt(cell):
    m, s = cell["auc_mean"], cell["auc_std"]
    if m is None:
        return "  n/a "
    star = "" if cell["trust"] else "*"
    return f"{m:.3f}±{s:.3f}{star}" if s is not None else f"{m:.3f}{star}"


def _print_matrix(model, combine, matrix, ind_members):
    """[ai-generated] Readable side × member × cell table. '*' = low-n (untrusted)."""
    print("", file=sys.stderr)
    print(f"=== ensemble-comparison matrix: {model or '(unnamed)'} "
          f"(combine={combine}) ===", file=sys.stderr)
    print("  example-AUC, mean±std over seeds; '*' = n_pos < MIN_TRUST_POS "
          "(low-n, untrusted)", file=sys.stderr)
    for side in ("probe", "verbalized"):
        print(f"\n[{side}]", file=sys.stderr)
        print(f"{'member':<14} {'memory':>16} {'injection':>16} {'overall':>16}",
              file=sys.stderr)
        for member in MEMBERS:
            cells = matrix[side][member]
            print(f"{member:<14} {_fmt(cells['memory']):>16} "
                  f"{_fmt(cells['injection']):>16} {_fmt(cells['overall']):>16}",
                  file=sys.stderr)
    low = [c for c, d in ind_members.items() if d["low_n"]]
    if low:
        print(f"\nlow-n per-CWE ind-ensemble members (probe n_test_pos < trust): "
              f"{low}", file=sys.stderr)
    print("\nReads: does prompt-specialization track probe-specialization (rows "
          "move the same way across sides)? does ind-ensemble beat its single "
          "category member? does cat-ensemble recover BOTH families in overall?",
          file=sys.stderr)


if __name__ == "__main__":
    main()
