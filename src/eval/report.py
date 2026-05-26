"""Render a FullReport as Markdown or as a JSON-friendly dict.

The Markdown format mirrors `data/eval_splits.md` so this module can be a
drop-in replacement for `scripts/eval_splits.py`'s own Markdown writer,
just with more columns (CI, F1, recall@FPR, calibration, baselines).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .protocol import FullReport, SplitReport


def _fmt(v: float, n: int = 3, na: str = "n/a") -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return na
    return f"{v:.{n}f}"


def render_markdown(report: FullReport, headline_split_priority: tuple[str, ...] = ("heldout_lang", "heldout_cwe")) -> str:
    """Render the report as Markdown. Returns the string; caller writes file.

    `headline_split_priority` picks which split's AUC goes in the headline.
    Default: the worst credible OOD split is the honest number.
    """
    lines: list[str] = []
    lines.append("# Probe evaluation report")
    lines.append("")
    lines.append(
        f"Activations: `{report.activations_path}` (layer {report.layer}, "
        f"N={report.n_examples}, pos={report.n_pos})"
    )
    lines.append(f"Pairs metadata: `{report.pairs_path}`")
    lines.append(f"Probe fit mode: `{'refit per split (OOD)' if report.refit else 'fixed pretrained probe'}`")
    lines.append("")

    # Main table.
    lines.append("## Per-split metrics")
    lines.append("")
    cols = ["split", "AUC (95% CI)", "F1", "R@5%FPR", "R@10%FPR", "Brier", "ECE", "n_test", "pos"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join("---" for _ in cols) + "|")
    for sp in report.splits:
        ci = f"{_fmt(sp.auc)} ({_fmt(sp.auc_ci_lo)}-{_fmt(sp.auc_ci_hi)})"
        r5 = sp.recall_at_fpr.get("recall_at_fpr_0.05")
        r10 = sp.recall_at_fpr.get("recall_at_fpr_0.10")
        lines.append(
            f"| `{sp.split_name}` | {ci} | {_fmt(sp.f1)} | {_fmt(r5)} | {_fmt(r10)} | "
            f"{_fmt(sp.brier)} | {_fmt(sp.ece)} | {sp.n_test} | {sp.n_test_pos} |"
        )
    lines.append("")

    # Baselines table.
    have_bl = any(sp.baseline_aucs for sp in report.splits)
    if have_bl:
        lines.append("## Probe vs baselines (AUC)")
        lines.append("")
        bl_names: list[str] = []
        for sp in report.splits:
            for k in sp.baseline_aucs.keys():
                if k.endswith("__error"):
                    continue
                if k not in bl_names:
                    bl_names.append(k)
        header = ["split", "probe"] + bl_names
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join("---" for _ in header) + "|")
        for sp in report.splits:
            row = [f"`{sp.split_name}`", _fmt(sp.auc)]
            for bl in bl_names:
                row.append(_fmt(sp.baseline_aucs.get(bl)))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # Notes.
    if any(sp.note for sp in report.splits):
        lines.append("## Split notes")
        lines.append("")
        for sp in report.splits:
            if sp.note:
                lines.append(f"- `{sp.split_name}`: {sp.note}")
        lines.append("")

    # Headline.
    lines.append("## Headline")
    lines.append("")
    random_baseline = _find(report.splits, "random_stratified")
    group_repo = _find(report.splits, "group_repo")
    if random_baseline:
        lines.append(
            f"- Random stratified (leaky): **AUC={_fmt(random_baseline.auc)}** "
            f"(CI {_fmt(random_baseline.auc_ci_lo)}-{_fmt(random_baseline.auc_ci_hi)})"
        )
    if group_repo:
        lines.append(
            f"- Group-by-repo: **AUC={_fmt(group_repo.auc)}** "
            f"(CI {_fmt(group_repo.auc_ci_lo)}-{_fmt(group_repo.auc_ci_hi)})"
        )
    worst = _pick_worst_credible(report.splits, headline_split_priority)
    if worst:
        lines.append(
            f"- Worst credible OOD split (`{worst.split_name}`): **AUC={_fmt(worst.auc)}** "
            f"(CI {_fmt(worst.auc_ci_lo)}-{_fmt(worst.auc_ci_hi)}), "
            f"F1={_fmt(worst.f1)}, R@10%FPR={_fmt(worst.recall_at_fpr.get('recall_at_fpr_0.10'))}"
        )
        lines.append("")
        lines.append(
            f"Recommended writeup number: **AUC={_fmt(worst.auc)} "
            f"({_fmt(worst.auc_ci_lo)}-{_fmt(worst.auc_ci_hi)})** under `{worst.split_name}`."
        )
    lines.append("")
    return "\n".join(lines)


def _find(reports: list[SplitReport], name: str) -> SplitReport | None:
    for r in reports:
        if r.split_name == name:
            return r
    return None


def _pick_worst_credible(reports: list[SplitReport], priority: tuple[str, ...]) -> SplitReport | None:
    candidates: list[SplitReport] = []
    for r in reports:
        if any(r.split_name.startswith(p) for p in priority):
            if not math.isnan(r.auc):
                candidates.append(r)
    if not candidates:
        return None
    return min(candidates, key=lambda r: r.auc)


def render_json(report: FullReport, indent: int = 2) -> str:
    return json.dumps(report.to_dict(), indent=indent, default=_json_default)


def _json_default(v):
    import numpy as np
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return str(v)


def write_report(report: FullReport, md_path: str | Path, json_path: str | Path | None = None) -> None:
    Path(md_path).parent.mkdir(parents=True, exist_ok=True)
    Path(md_path).write_text(render_markdown(report))
    if json_path is not None:
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        Path(json_path).write_text(render_json(report))
