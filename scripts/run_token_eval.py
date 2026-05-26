"""Token-level eval CLI for the new probe path.

Reads:
  - `data/dataset.jsonl` (rows with char-range token_labels)
  - per-row per-token probability arrays produced by the token probe
    (one .npz with `probs_row_NNNN` keys; one array per row, variable
     length, dtype float32)
  - per-row per-token offsets (one .npz with `offsets_row_NNNN` keys;
    one (T, 2) int32 array per row)

Writes:
  - Markdown report at --out-md
  - JSON report at --out-json

Both per-row .npz files are typically produced by an updated
`src/extract_token_activations.py` + `src/stream_with_probe.py` pipeline.
The format is intentionally simple so the eval is decoupled from the
probe / extractor and can be tested with synthetic data
(see `scripts/test_token_eval.py`).

When the per-token files aren't available, run the sample-level path
instead (`scripts/run_eval_framework.py`) and treat the old probe as a
baseline via `--probe-baseline data/probe.npz`.

Status: the token-level extractor is currently broken (#17). This CLI
runs as soon as a fixed extractor lands. Smoke test runs against
synthetic data today.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.eval import baselines as bl_mod  # noqa: E402
from src.eval import token_data  # noqa: E402
from src.eval import token_protocol  # noqa: E402
from src.eval.code_mask import code_only_mask  # noqa: E402
from src.eval.token_data import (  # noqa: E402
    char_spans_to_token_spans,
    parse_spans,
)


def _load_per_row_npz(path: Path, prefix: str) -> list[np.ndarray]:
    npz = np.load(path)
    keys = sorted([k for k in npz.files if k.startswith(prefix)],
                  key=lambda k: int(k[len(prefix):]))
    return [npz[k] for k in keys]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/dataset.jsonl")
    ap.add_argument("--token-probs", required=True,
                    help=".npz with probs_row_NNNN arrays (one per dataset row)")
    ap.add_argument("--token-offsets", required=True,
                    help=".npz with offsets_row_NNNN (T,2) arrays")
    ap.add_argument("--out-md", default="data/eval/token_report.md")
    ap.add_argument("--out-json", default="data/eval/token_report.json")
    ap.add_argument("--bootstrap-n", type=int, default=1000)
    ap.add_argument("--probe-baseline", default=None,
                    help="Path to a sample-level probe .npz to include as baseline.")
    ap.add_argument("--sample-activations", default=None,
                    help="Sample-level activation .npz (X, y) aligned to dataset.jsonl rows. "
                         "Required when --probe-baseline is set.")
    ap.add_argument("--proximity-window", type=int, default=0,
                    help="Preprocessing axis: dilate the positive token "
                         "labels used by `tokens` and `tokens_code` by ±W "
                         "tokens around each annotated positive span. "
                         "Default 0 (exact-span labels). Does not affect "
                         "tokens_annotated_negative or example_max.")
    ap.add_argument("--no-code-only", action="store_true",
                    help="Skip the tokens_code aggregation level "
                         "(default: compute it). Falls back silently if "
                         "tree-sitter or a language grammar is missing.")
    args = ap.parse_args()

    rows = token_data.load_token_dataset(args.dataset)
    probs_per_row = _load_per_row_npz(Path(args.token_probs), "probs_row_")
    offsets_per_row = _load_per_row_npz(Path(args.token_offsets), "offsets_row_")
    if len(rows) != len(probs_per_row) or len(rows) != len(offsets_per_row):
        print(
            f"[run_token_eval] row count mismatch: dataset={len(rows)}, "
            f"probs={len(probs_per_row)}, offsets={len(offsets_per_row)}",
            file=sys.stderr,
        )
        return 1
    print(f"[run_token_eval] loaded {len(rows)} rows", file=sys.stderr)

    # Map char spans -> token spans per row.
    token_spans_per_row: list[list[tuple[int, int, int]]] = []
    code_only_masks: list[np.ndarray] | None = [] if not args.no_code_only else None
    for r, offs in zip(rows, offsets_per_row):
        tok_offsets = [(int(s), int(e)) for s, e in offs]
        char_spans = parse_spans(r)
        token_spans_per_row.append(char_spans_to_token_spans(char_spans, tok_offsets))
        if code_only_masks is not None:
            code_only_masks.append(
                code_only_mask(r.get("code", ""), r.get("lang", "") or "", tok_offsets)
            )
    if code_only_masks is not None:
        kept = sum(int(m.sum()) for m in code_only_masks)
        total = sum(int(m.shape[0]) for m in code_only_masks)
        dropped_pct = 100.0 * (1.0 - kept / max(1, total))
        print(
            f"[run_token_eval] tokens_code: kept {kept}/{total} tokens "
            f"({dropped_pct:.1f}% dropped)",
            file=sys.stderr,
        )

    baselines = bl_mod.all_baselines()
    sample_X = None
    if args.probe_baseline:
        if not args.sample_activations:
            print(
                "[run_token_eval] --probe-baseline requires --sample-activations",
                file=sys.stderr,
            )
            return 2
        baselines = bl_mod.with_probe_baseline(args.probe_baseline, broadcast=True)
        npz = np.load(args.sample_activations)
        sample_X = npz["X"]
        if sample_X.shape[0] != len(rows):
            print(
                f"[run_token_eval] sample activations shape {sample_X.shape} doesn't "
                f"match dataset rows ({len(rows)})",
                file=sys.stderr,
            )
            return 3

    full = token_protocol.full_token_report(
        rows, probs_per_row, token_spans_per_row,
        dataset_path=args.dataset,
        bootstrap_n=args.bootstrap_n,
        baselines=baselines,
        sample_X=sample_X,
        proximity_window=args.proximity_window,
        code_only_masks=code_only_masks,
    )

    # Render: simple markdown table; full JSON dump for downstream.
    md_lines: list[str] = []
    md_lines.append("# Token-level probe evaluation report")
    md_lines.append("")
    md_lines.append(f"Dataset: `{args.dataset}` (N={full.n_examples}, pos={full.n_pos_examples})")
    md_lines.append("")
    md_lines.append(f"## Per-split metrics (proximity_window W={args.proximity_window})")
    md_lines.append("")
    md_lines.append("| split | level | AUC (95% CI) | F1 | R@10%FPR | Brier | n_total | n_pos |")
    md_lines.append("|---|---|---|---|---|---|---:|---:|")
    for sp in full.splits:
        for level_name, m in (
            ("tokens", sp.tokens_metrics),
            ("tokens_code", sp.tokens_code_metrics),
            ("tokens_annotated_negative", sp.tokens_annotated_negative_metrics),
            ("example_max", sp.example_max_metrics),
        ):
            if m.get("n_total", 0) == 0:
                continue
            auc = m.get("auc", float("nan"))
            lo = m.get("auc_ci_lo", float("nan"))
            hi = m.get("auc_ci_hi", float("nan"))
            f1 = m.get("f1", float("nan"))
            r10 = m.get("recall_at_fpr_0.10", float("nan"))
            br = m.get("brier", float("nan"))
            md_lines.append(
                f"| `{sp.split_name}` | {level_name} | "
                f"{auc:.3f} ({lo:.3f}-{hi:.3f}) | {f1:.3f} | {r10:.3f} | {br:.3f} | "
                f"{m.get('n_total', 0)} | {m.get('n_pos', 0)} |"
            )
    md_lines.append("")
    md_lines.append("## Baselines (AUC at each level)")
    md_lines.append("")
    md_lines.append("| split | baseline | tokens | tokens_code | tokens_annotated_negative | example_max |")
    md_lines.append("|---|---|---:|---:|---:|---:|")
    for sp in full.splits:
        for bl_name, lvls in sp.baseline_aucs.items():
            md_lines.append(
                f"| `{sp.split_name}` | {bl_name} | "
                f"{lvls.get('tokens_auc', float('nan')):.3f} | "
                f"{lvls.get('tokens_code_auc', float('nan')):.3f} | "
                f"{lvls.get('tokens_annotated_negative_auc', float('nan')):.3f} | "
                f"{lvls.get('example_max_auc', float('nan')):.3f} |"
            )
    md = "\n".join(md_lines) + "\n"

    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(md)
    import json
    Path(args.out_json).write_text(json.dumps(full.to_dict(), indent=2, default=str))
    print(f"[run_token_eval] wrote {args.out_md}", file=sys.stderr)
    print(f"[run_token_eval] wrote {args.out_json}", file=sys.stderr)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
