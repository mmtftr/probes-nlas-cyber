# [ai-generated]
"""Tabulate the {K x agg} ensemble cells vs the K=1 baseline, overall + per-lang
+ per-CWE. Pure-local: reads the per-cell JSONs `train_ensemble.py` wrote, no GPU.

The K=1 baseline: any cell with K==1 is the single-linear probe (max == logsumexp
== softmax_gate collapse to a single direction at K=1). We use the K=1/max cell as
the anchor (falling back to any K=1 cell if max is absent) and report Delta =
cell - baseline for every (K, agg) at every level. The single-probe 0.788 (Qwen
L25) and the sweep-6 per-CWE numbers (C~0.59, UAF~0.52, NULL~0.55, OOB-read~0.56)
are the external reference points the hypothesis targets.

Usage:
    python aggregate_ensemble.py --cells-dir <dir of cell_*.json> \
        --out summary.json [--markdown summary.md]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def _tc(block) -> float | None:
    if not block:
        return None
    v = block.get("tokens_code_auc")
    return v if isinstance(v, (int, float)) else None


def _fmt(v) -> str:
    return f"{v:.3f}" if isinstance(v, (int, float)) and v == v else " . "


def _delta(v, base) -> str:
    if not (isinstance(v, (int, float)) and isinstance(base, (int, float))):
        return "  .  "
    d = v - base
    return f"{d:+.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells-dir", required=True,
                    help="dir of per-cell JSONs from train_ensemble.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--markdown", default=None)
    args = ap.parse_args()

    cells = []
    for p in sorted(Path(args.cells_dir).glob("*.json")):
        try:
            c = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if "overall" not in c:  # skip skipped/errored cells
            continue
        cells.append(c)
    if not cells:
        raise SystemExit(f"no usable cells in {args.cells_dir}")

    model = cells[0].get("model", "")
    # Baseline = K==1, prefer agg==max.
    k1 = [c for c in cells if c.get("K") == 1]
    baseline = next((c for c in k1 if c.get("agg") == "max"), k1[0] if k1 else None)
    base_overall = _tc(baseline.get("overall")) if baseline else None

    # Collect the union of lang / cwe keys across cells.
    langs, cwes = set(), set()
    for c in cells:
        langs |= set(c.get("by_lang", {}))
        cwes |= set(c.get("by_cwe", {}))
    langs = sorted(langs)
    cwes = sorted(cwes)

    def base_at(level, key=None):
        if not baseline:
            return None
        if level == "overall":
            return _tc(baseline.get("overall"))
        return _tc(baseline.get(level, {}).get(key))

    rows = []
    for c in sorted(cells, key=lambda x: (x.get("K", 0), x.get("agg", ""))):
        row = {
            "K": c.get("K"), "agg": c.get("agg"),
            "tau": c.get("tau"), "gate_mode": c.get("gate_mode"),
            "val_tokens_code_auc": c.get("val_tokens_code_auc"),
            "overall": _tc(c.get("overall")),
            "overall_delta": (None if base_overall is None or _tc(c.get("overall")) is None
                              else _tc(c.get("overall")) - base_overall),
            "by_lang": {lg: _tc(c.get("by_lang", {}).get(lg)) for lg in langs},
            "by_cwe": {cw: _tc(c.get("by_cwe", {}).get(cw)) for cw in cwes},
        }
        rows.append(row)

    summary = {
        "model": model,
        "baseline_cell": {"K": baseline.get("K"), "agg": baseline.get("agg")} if baseline else None,
        "baseline_overall_tokens_code_auc": base_overall,
        "langs": langs, "cwes": cwes,
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))

    if args.markdown:
        lines = [f"# Ensemble {{K x agg}} vs K=1 baseline — {model}", ""]
        lines.append(f"Baseline (K=1 {baseline.get('agg') if baseline else '?'}) "
                     f"overall tokens_code = {_fmt(base_overall)}")
        lines.append("")
        # Overall table.
        hdr = ["K", "agg", "val_tc", "overall", "Δ vs K=1"] + langs + cwes
        lines.append("| " + " | ".join(hdr) + " |")
        lines.append("|" + "|".join(["---"] * len(hdr)) + "|")
        for r in rows:
            cells_md = [
                str(r["K"]), r["agg"], _fmt(r["val_tokens_code_auc"]),
                _fmt(r["overall"]),
                _delta(r["overall"], base_overall),
            ]
            cells_md += [
                f"{_fmt(r['by_lang'][lg])} ({_delta(r['by_lang'][lg], base_at('by_lang', lg))})"
                for lg in langs
            ]
            cells_md += [
                f"{_fmt(r['by_cwe'][cw])} ({_delta(r['by_cwe'][cw], base_at('by_cwe', cw))})"
                for cw in cwes
            ]
            lines.append("| " + " | ".join(cells_md) + " |")
        Path(args.markdown).write_text("\n".join(lines) + "\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
