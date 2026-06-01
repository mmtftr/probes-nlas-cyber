# [ai-generated]
"""Exp-10 aggregator: tabulate GENERAL vs per-CWE SPECIALIZED tokens_code AUC.

Reads one or more per_cwe_probe.py output JSONs and prints:
  1. a per-CWE table: CWE | family | n_train_pos | n_test_pos | general | spec |
     Δ(spec-gen) | trust — with an explicit "LOW-n" marker where n_test_pos is
     below the trust threshold (AUC unstable there, do not conclude).
  2. a per-FAMILY roll-up (injection-class vs memory-safety): mean general,
     mean specialized, mean Δ over the trustworthy CWEs only, plus a separate
     count of how many CWEs were dropped for low n.

Family means are weighted by n_test_pos (a 44-positive CWE should count more
than a 14-positive one) and are computed over TRUST==True CWEs only, so the
roll-up isn't dominated by noisy small-n cells. The raw per-CWE Δ for every CWE
(trustworthy or not) is still printed so nothing is hidden.

The headline question this answers:
  - per-FAMILY Δ ≈ 0 on memory-safety (spec no better than general, both low)
    ⇒ the memory-safety signal is largely ABSENT in activations, not a capacity
    under-allocation by the general probe.
  - per-FAMILY Δ > 0 on memory-safety (spec lifts a low general AUC) ⇒ the
    signal EXISTS but the general probe under-allocates capacity to it.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path


def _fmt(x) -> str:
    return f"{x:.3f}" if isinstance(x, float) and x == x else "  -  "


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+", help="per_cwe_probe.py output JSON(s)")
    ap.add_argument("--out", default=None, help="optional combined JSON dump")
    args = ap.parse_args()

    combined = {}
    for path in args.results:
        data = json.loads(Path(path).read_text())
        model = data.get("model") or Path(path).stem
        head = data.get("head", "?")
        neg_pool = data.get("neg_pool", "?")
        tag = f"{model} [layer {data.get('layer')}, head={head}, neg_pool={neg_pool}]"
        combined[tag] = data

        print(f"\n=== {tag} ===")
        print(f"{'CWE':9} {'family':10} {'tr_pos':>6} {'te_pos':>6} "
              f"{'general':>8} {'spec':>8} {'Δ':>7}  trust")
        fam_acc = {}  # family -> list of (n_test_pos, gen, spec) for trust==True
        for cwe, rec in sorted(data["by_cwe"].items()):
            fam = rec.get("family", "?")
            g = rec.get("general_tokens_code_auc")
            s = rec.get("specialized_tokens_code_auc")
            d = rec.get("delta_spec_minus_gen")
            trust = rec.get("trust", False)
            mark = "" if trust else "  LOW-n"
            err = rec.get("error")
            tail = f"  ({err})" if err else mark
            print(f"{cwe:9} {fam:10} {rec.get('n_train_pos', 0):>6} "
                  f"{rec.get('n_test_pos', 0):>6} {_fmt(g):>8} {_fmt(s):>8} "
                  f"{_fmt(d):>7} {tail}")
            if trust and isinstance(g, float) and g == g and isinstance(s, float) and s == s:
                fam_acc.setdefault(fam, []).append((rec.get("n_test_pos", 0), g, s))

        # Family roll-up (n-weighted, trust-only).
        print(f"\n  --- family roll-up (n_test_pos-weighted, trustworthy CWEs only) ---")
        for fam, rows in sorted(fam_acc.items()):
            wsum = sum(n for n, _, _ in rows) or 1
            gmean = sum(n * g for n, g, _ in rows) / wsum
            smean = sum(n * s for n, _, s in rows) / wsum
            n_cwe = len(rows)
            n_drop = sum(1 for r in data["by_cwe"].values()
                         if r.get("family") == fam and not r.get("trust", False))
            print(f"  {fam:10} n_cwe={n_cwe}  general={gmean:.3f}  "
                  f"specialized={smean:.3f}  Δ={smean - gmean:+.3f}  "
                  f"(dropped {n_drop} low-n CWEs)")
        if not fam_acc:
            print("  (no trustworthy CWEs — all below n threshold)")

    if args.out:
        Path(args.out).write_text(json.dumps(combined, indent=2))
        print(f"\n[compare] wrote combined dump -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
