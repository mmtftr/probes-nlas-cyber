"""Top-level CLI for the probe evaluation framework.

Single entry point that:
  1. Loads activations + pairs for a given layer.
  2. Runs every leakage-aware split.
  3. Adds regex / length / random baselines on each split.
  4. Writes Markdown + JSON reports to `data/eval/`.

Reproduces (and extends) `scripts/eval_splits.py`:
  - keeps the random / group_repo / heldout_cwe / heldout_lang / heldout_source splits
  - adds AUC bootstrap CIs, F1, recall@FPR, calibration (Brier, ECE)
  - adds baselines (random / regex / length)
  - emits machine-readable JSON alongside the Markdown
  - supports two probe modes:
      --refit (default) -> refit logreg per split; for OOD generalisation claims
      --no-refit + --probe data/probe.npz -> score with a fixed shipped probe

Usage (with uv, no venv needed):
  uv run --with scikit-learn --with numpy python scripts/run_eval_framework.py \\
    --activations data/activations_v2/activations_layer17.npz \\
    --pairs       data/pairs.jsonl \\
    --layer       17 \\
    --out-md      data/eval/report.md \\
    --out-json    data/eval/report.json

Or with the project venv active (requirements.txt provides sklearn/numpy):
  python scripts/run_eval_framework.py [...same args...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.eval import baselines as bl_mod  # noqa: E402
from src.eval import probe_io  # noqa: E402
from src.eval import protocol  # noqa: E402
from src.eval import report as report_mod  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", default="data/activations_v2/activations_layer17.npz")
    ap.add_argument("--pairs", default="data/pairs.jsonl")
    ap.add_argument("--layer", type=int, default=17)
    ap.add_argument("--out-md", default="data/eval/report.md")
    ap.add_argument("--out-json", default="data/eval/report.json")
    ap.add_argument("--bootstrap-n", type=int, default=1000)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument(
        "--no-baselines", action="store_true",
        help="Skip random/length/regex baselines (faster).",
    )
    ap.add_argument(
        "--include",
        nargs="*",
        default=["random", "group_repo", "heldout_cwe", "heldout_lang", "heldout_source"],
        help="Which split families to run.",
    )
    refit_grp = ap.add_mutually_exclusive_group()
    refit_grp.add_argument(
        "--refit", dest="refit", action="store_true", default=True,
        help="Refit a fresh logreg per split (default; honest OOD).",
    )
    refit_grp.add_argument(
        "--no-refit", dest="refit", action="store_false",
        help="Use a fixed pretrained probe (--probe) and only score test slices.",
    )
    ap.add_argument(
        "--probe", default=None,
        help="Path to a .npz probe (w/b/layer) to use with --no-refit.",
    )
    args = ap.parse_args()

    X, y = probe_io.load_activations(args.activations)
    rows = probe_io.load_pairs(args.pairs)
    if len(rows) != len(y):
        print(
            f"[run_eval] row count mismatch: pairs={len(rows)} activations={len(y)}",
            file=sys.stderr,
        )
        return 1
    print(
        f"[run_eval] X={X.shape}  y_pos={int(y.sum())}  layer={args.layer}  refit={args.refit}",
        file=sys.stderr,
    )

    probe = None
    if not args.refit:
        if not args.probe:
            print("[run_eval] --no-refit requires --probe PATH", file=sys.stderr)
            return 2
        probe = probe_io.load_probe(args.probe)
        print(f"[run_eval] loaded probe from {args.probe} (layer={probe.layer})", file=sys.stderr)

    baselines = None if args.no_baselines else bl_mod.all_baselines()
    full = protocol.full_report(
        X, y, rows,
        activations_path=args.activations,
        pairs_path=args.pairs,
        layer=args.layer,
        refit=args.refit,
        probe=probe,
        baselines=baselines,
        include=tuple(args.include),
        bootstrap_n=args.bootstrap_n,
        threshold=args.threshold,
    )

    report_mod.write_report(full, args.out_md, args.out_json)
    print(f"[run_eval] wrote {args.out_md}", file=sys.stderr)
    print(f"[run_eval] wrote {args.out_json}", file=sys.stderr)
    print(report_mod.render_markdown(full))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
