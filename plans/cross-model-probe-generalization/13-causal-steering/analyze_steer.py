# [ai-generated]
"""Exp-13 causal-steering ANALYSIS (LOCAL / CPU — no model load).

Reads the four `steer_13_<slug>.json` files produced by steer_judge.py (v2),
prints + plots P(yes) vs alpha curves per (direction x subset) per model, and
computes the SPECIFICITY VERDICT per model for the MEMORY direction.

Run AFTER the human pulls the steer JSONs from the cluster. Pure post-hoc.

=== INPUTS ==========================================================
The v2 steer JSON schema (per model):
  {model, layer, scale_def:"proj_std", alpha_grid:[...],
   degrade_thresh, directions:{<dir>:{scale,...}},
   by_direction:{<dir>:{by_subset:{memory_pos:{p_yes:[..], degraded:[..]},
                                    injection_pos:{...}, negative:{...}},
                        n_per_subset}},
   selfcheck:{ok, max_abs_diff, ...}}
Directions: "memory" (TEST), "injection" (real control), "random_0"/"random_1"
(random control). DEGRADED cells (yes+no mass collapsed) are EXCLUDED from the
verdict math — a degraded P(yes) is meaningless (the model isn't answering
yes/no), and including it would re-import the v1 blow-up artifact.

=== SPECIFICITY VERDICT (per model, for the MEMORY direction) =======
For the memory direction we report three, on NON-DEGRADED cells only:
  (i)   MONOTONICITY / Δ on memory_pos across alpha. We fit the sign of the
        trend (Spearman rho of P(yes) vs alpha over non-degraded alphas) and the
        endpoint Δ = P(yes)[max non-degraded alpha] − P(yes)[alpha=0]. A causal
        memory axis should show a monotone RISE on +alpha (rho>0, Δ>0).
  (ii)  Δ(memory_pos) vs Δ(negative) at +alpha. Δ_sub = P(yes)[+a*] −
        P(yes)[0] at the largest non-degraded +alpha a*. Memory-SPECIFIC if the
        memory_pos rise EXCEEDS the negative rise: Δ(memory_pos) − Δ(negative) > 0.
        (A direction that raised P(yes) equally on negatives would be a generic
        yes-bias axis, not memory-specific — the v1 worry.)
  (iii) Δ(memory_pos) for the MEMORY dir vs the RANDOM dirs at matched +alpha.
        Random dirs scaled by their OWN proj-std push the residual the same
        number of std along a meaningless axis. SPECIFIC if the memory dir's
        memory_pos rise >> the mean random-dir memory_pos rise at the same
        +alpha: Δ_mem(memory dir) − mean_k Δ_mem(random_k) > 0.

VERDICT label per model (heuristic, lead may recalibrate the cutoffs — they are
named constants below and printed):
  "memory-specific causal" : rise on +alpha (Δ>SPEC_MIN_RISE) AND
                             beats-negative (ii)>SPEC_MARGIN AND
                             beats-random (iii)>SPEC_MARGIN.
  "causal but generic"     : rise on +alpha but NOT beating negative/random
                             (generic yes-bias axis).
  "epiphenomenal / flat"   : |Δ| <= SPEC_MIN_RISE (steering moves nothing).
  "inconclusive"           : too many degraded +alpha cells to judge.

=== OUTPUTS =========================================================
- data/plots/cross-model/fig10_steering.png : grid of P(yes)-vs-alpha curves
  (rows = models, cols = subset {memory_pos, injection_pos, negative}); one line
  per direction; degraded points marked hollow.
- <results_dir>/fig10_steering_summary.json : the per-model verdict + the three
  metric values + the alphas used.

Usage:
  python analyze_steer.py --results-dir <dir-with-steer_13_*.json>
                          [--plot-out PATH] [--summary-out PATH]
The four expected slugs (one per roster model) are derived from MODELS below;
missing files are skipped with a warning (so a partial pull still analyzes).
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Roster: (display label, model id). Slug = id with '/' and non-[A-Za-z0-9._-]
# -> '_' (mirrors run_steer_node.sh's SLUG). Best layers documented for context.
MODELS = [
    ("Qwen2.5-Coder-32B", "Qwen/Qwen2.5-Coder-32B-Instruct"),  # L25
    ("gemma-3-27b",       "google/gemma-3-27b-it"),            # L19
    ("Qwen3-32B",         "Qwen/Qwen3-32B"),                   # L27
    ("Qwen3.6-27B",       "Qwen/Qwen3.6-27B"),                 # L30
]

DIRECTIONS = ["memory", "injection", "random_0", "random_1"]
SUBSETS = ["memory_pos", "injection_pos", "negative"]
DIR_COLORS = {"memory": "#C44E52", "injection": "#4C72B0",
              "random_0": "#999999", "random_1": "#CCCCCC"}

# Verdict cutoffs (heuristic; the lead owns them — printed in the summary).
# TODO(adhoc-decision): these thresholds gate the verdict LABEL only; the raw
# metric numbers are always emitted so the lead can relabel without re-running.
SPEC_MIN_RISE = 0.05   # |Δ(memory_pos)| below this on +alpha => flat/epiphenomenal
SPEC_MARGIN = 0.05     # memory must beat negative / random by at least this Δ


def slug_of(model_id: str) -> str:
    """run_steer_node.sh's SLUG: '/' -> '_', then any non-[A-Za-z0-9._-] -> '_'."""
    s = model_id.replace("/", "_")
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in s)


def _arr(cell, key):
    """A (p_yes|degraded) list from a by_subset cell as a numpy array, or an
    empty array if absent (robust to partial/older JSONs)."""
    return np.asarray(cell.get(key, []), dtype=float if key == "p_yes" else bool)


def _alpha0_index(alphas):
    idx = [i for i, a in enumerate(alphas) if a == 0.0]
    if not idx:
        raise SystemExit("[analyze] alpha grid has no 0.0 — cannot baseline")
    return idx[0]


def _max_pos_alpha_idx(alphas, degraded):
    """Index of the LARGEST positive alpha whose cell is NOT degraded, or None.
    `degraded` is the per-alpha bool array for the subset/direction in question."""
    cands = [(a, i) for i, a in enumerate(alphas)
             if a > 0.0 and i < len(degraded) and not bool(degraded[i])]
    if not cands:
        return None
    return max(cands, key=lambda t: t[0])[1]


def _spearman_sign(x, y):
    """Spearman rho between x and y (rank-Pearson), NaN if <2 points or constant.
    Stdlib/numpy only (no scipy dependency)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def memory_verdict(model_json: dict) -> dict:
    """Compute the three specificity metrics + a verdict label for the MEMORY
    direction of one model. Pure: reads, returns a fresh dict, mutates nothing.
    Degraded cells are EXCLUDED throughout."""
    alphas = list(model_json["alpha_grid"])
    a0 = _alpha0_index(alphas)
    # `by_direction` is the v2 schema. A stale v1-schema JSON (top-level
    # `by_subset`, single memory direction, no controls) lacks it — treat that
    # as inconclusive rather than crashing, so a leftover v1 file in the results
    # dir is skipped gracefully.
    bd = model_json.get("by_direction")

    out = {"alphas": alphas, "alpha0_index": a0,
           "metrics": {}, "notes": []}

    if not bd:
        out["verdict"] = "inconclusive"
        out["notes"].append("no by_direction (v1-schema or empty) — not analyzable")
        return out
    if "memory" not in bd:
        out["verdict"] = "inconclusive"
        out["notes"].append("memory direction missing from by_direction")
        return out

    mem_dir = bd["memory"]["by_subset"]

    # --- (i) monotonicity / endpoint Δ on memory_pos (memory direction) ---
    mp_py = _arr(mem_dir["memory_pos"], "p_yes")
    mp_deg = _arr(mem_dir["memory_pos"], "degraded")
    keep = ~mp_deg.astype(bool) if mp_deg.size == mp_py.size else np.ones_like(mp_py, bool)
    a_keep = np.asarray(alphas)[keep]
    py_keep = mp_py[keep]
    rho = _spearman_sign(a_keep, py_keep)
    base = float(mp_py[a0]) if not bool(mp_deg[a0] if mp_deg.size else False) else float("nan")
    imax = _max_pos_alpha_idx(alphas, mp_deg if mp_deg.size else np.zeros(len(alphas), bool))
    endpoint_delta = (float(mp_py[imax]) - base) if (imax is not None and not np.isnan(base)) else float("nan")
    out["metrics"]["i_monotonicity"] = {
        "spearman_rho_pyes_vs_alpha": rho,
        "endpoint_delta_memorypos": endpoint_delta,
        "max_pos_alpha_used": (alphas[imax] if imax is not None else None),
        "n_nondegraded_alphas": int(keep.sum()),
    }

    # --- (ii) Δ(memory_pos) vs Δ(negative) at the largest shared non-degraded +alpha ---
    neg = mem_dir["negative"]
    neg_py = _arr(neg, "p_yes")
    neg_deg = _arr(neg, "degraded")
    # shared +alpha index: largest +alpha non-degraded for BOTH memory_pos & negative.
    both_deg = np.zeros(len(alphas), bool)
    if mp_deg.size == len(alphas):
        both_deg |= mp_deg.astype(bool)
    if neg_deg.size == len(alphas):
        both_deg |= neg_deg.astype(bool)
    ishare = _max_pos_alpha_idx(alphas, both_deg)
    if ishare is not None:
        d_mem = float(mp_py[ishare]) - float(mp_py[a0])
        d_neg = float(neg_py[ishare]) - float(neg_py[a0])
        out["metrics"]["ii_beats_negative"] = {
            "alpha": alphas[ishare],
            "delta_memorypos": d_mem,
            "delta_negative": d_neg,
            "memory_minus_negative": d_mem - d_neg,
        }
    else:
        out["metrics"]["ii_beats_negative"] = None
        out["notes"].append("no shared non-degraded +alpha for memory_pos & negative")

    # --- (iii) Δ(memory_pos) memory dir vs random dirs at matched +alpha ---
    rand_names = [d for d in ("random_0", "random_1") if d in bd]
    iii = None
    if rand_names:
        # matched +alpha: largest +alpha non-degraded for memory dir AND every
        # random dir's memory_pos cell.
        comb_deg = np.zeros(len(alphas), bool)
        if mp_deg.size == len(alphas):
            comb_deg |= mp_deg.astype(bool)
        for rn in rand_names:
            rmp_deg = _arr(bd[rn]["by_subset"]["memory_pos"], "degraded")
            if rmp_deg.size == len(alphas):
                comb_deg |= rmp_deg.astype(bool)
        im = _max_pos_alpha_idx(alphas, comb_deg)
        if im is not None:
            d_mem = float(mp_py[im]) - float(mp_py[a0])
            rand_deltas = []
            for rn in rand_names:
                rpy = _arr(bd[rn]["by_subset"]["memory_pos"], "p_yes")
                rand_deltas.append(float(rpy[im]) - float(rpy[a0]))
            mean_rand = float(np.mean(rand_deltas))
            iii = {
                "alpha": alphas[im],
                "delta_memorypos_memory_dir": d_mem,
                "delta_memorypos_random_mean": mean_rand,
                "per_random": dict(zip(rand_names, rand_deltas)),
                "memory_minus_random": d_mem - mean_rand,
            }
        else:
            out["notes"].append("no shared non-degraded +alpha for memory & random dirs")
    out["metrics"]["iii_beats_random"] = iii

    # --- VERDICT label ---
    rise = out["metrics"]["i_monotonicity"]["endpoint_delta_memorypos"]
    ii = out["metrics"]["ii_beats_negative"]
    if imax is None:
        out["verdict"] = "inconclusive"
        out["notes"].append("all +alpha memory_pos cells degraded")
    elif np.isnan(rise) or abs(rise) <= SPEC_MIN_RISE:
        out["verdict"] = "epiphenomenal / flat"
    else:
        beats_neg = (ii is not None and ii["memory_minus_negative"] > SPEC_MARGIN)
        beats_rand = (iii is not None and iii["memory_minus_random"] > SPEC_MARGIN)
        if rise > SPEC_MIN_RISE and beats_neg and beats_rand:
            out["verdict"] = "memory-specific causal"
        else:
            out["verdict"] = "causal but generic"
    out["cutoffs"] = {"SPEC_MIN_RISE": SPEC_MIN_RISE, "SPEC_MARGIN": SPEC_MARGIN}
    return out


def make_plot(loaded, plot_out: Path):
    """Grid of P(yes)-vs-alpha curves: rows = models, cols = subsets; one line
    per direction. Degraded points are drawn hollow. Saves to plot_out.

    matplotlib is the ONLY heavy dep and is not a project requirement; if it is
    absent we WARN and skip the figure rather than crash — the verdict summary
    (the load-bearing output) is already written by the time we get here."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[analyze] matplotlib not installed — skipping fig10 (summary JSON "
              "still written). `uv pip install matplotlib` to enable the plot.",
              file=sys.stderr)
        return

    models = [(lbl, mj) for lbl, mj in loaded if mj is not None]
    if not models:
        print("[analyze] no models loaded — skipping plot", file=sys.stderr)
        return
    nrow, ncol = len(models), len(SUBSETS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.4 * nrow),
                             squeeze=False, sharex=True)
    for r, (lbl, mj) in enumerate(models):
        alphas = np.asarray(mj["alpha_grid"], float)
        bd = mj["by_direction"]
        for c, sub in enumerate(SUBSETS):
            ax = axes[r][c]
            for dname in DIRECTIONS:
                if dname not in bd or sub not in bd[dname]["by_subset"]:
                    continue
                cell = bd[dname]["by_subset"][sub]
                py = _arr(cell, "p_yes")
                deg = _arr(cell, "degraded")
                col = DIR_COLORS.get(dname, "#000000")
                ax.plot(alphas, py, "-", color=col, lw=1.6,
                        label=dname if (r == 0 and c == 0) else None, zorder=2)
                ok = ~deg.astype(bool) if deg.size == py.size else np.ones_like(py, bool)
                ax.scatter(alphas[ok], py[ok], s=26, color=col, zorder=3)
                if deg.size == py.size and deg.any():
                    ax.scatter(alphas[deg.astype(bool)], py[deg.astype(bool)],
                               s=34, facecolors="none", edgecolors=col, zorder=3)
            ax.axhline(0.5, ls="--", c="gray", lw=0.9)
            ax.axvline(0.0, ls=":", c="gray", lw=0.9)
            ax.set_ylim(-0.02, 1.02)
            ax.grid(alpha=0.2)
            if r == 0:
                ax.set_title(sub, fontsize=11, weight="bold")
            if c == 0:
                ax.set_ylabel(f"{lbl}\nL{mj['layer']}  mean P(yes)", fontsize=9)
            if r == nrow - 1:
                ax.set_xlabel("alpha (proj-std units)", fontsize=9)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(DIRECTIONS),
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("Exp-13 causal steering (v2, proj-std units): is the memory direction "
                 "memory-SPECIFIC?\nhollow = degraded cell (yes+no mass collapsed, "
                 "excluded from the verdict)", fontsize=12, weight="bold", y=1.04)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plot_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_out, dpi=130, bbox_inches="tight")
    print(f"[analyze] wrote plot {plot_out}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True,
                    help="dir containing steer_13_<slug>.json (one per model)")
    ap.add_argument("--plot-out", default=None,
                    help="default: <repo>/data/plots/cross-model/fig10_steering.png")
    ap.add_argument("--summary-out", default=None,
                    help="default: <results-dir>/fig10_steering_summary.json")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[3]
    results_dir = Path(args.results_dir)
    plot_out = Path(args.plot_out) if args.plot_out else (
        repo / "data" / "plots" / "cross-model" / "fig10_steering.png")
    summary_out = Path(args.summary_out) if args.summary_out else (
        results_dir / "fig10_steering_summary.json")

    loaded = []
    for lbl, mid in MODELS:
        f = results_dir / f"steer_13_{slug_of(mid)}.json"
        if not f.exists():
            print(f"[analyze] WARNING: {f} absent — skipping {lbl}", file=sys.stderr)
            loaded.append((lbl, None))
            continue
        loaded.append((lbl, json.loads(f.read_text())))

    summary = {"models": {}, "cutoffs": {"SPEC_MIN_RISE": SPEC_MIN_RISE,
                                         "SPEC_MARGIN": SPEC_MARGIN}}
    print(f"\n{'model':18} | verdict                 | Δmem(+α) | mem-neg | mem-rand | selfcheck")
    print("-" * 92)
    for lbl, mj in loaded:
        if mj is None:
            continue
        v = memory_verdict(mj)
        sc = mj.get("selfcheck", {})
        summary["models"][lbl] = {
            "model": mj.get("model"), "layer": mj.get("layer"),
            "scale_def": mj.get("scale_def"),
            "selfcheck_ok": sc.get("ok"), "selfcheck_max_abs_diff": sc.get("max_abs_diff"),
            "verdict": v["verdict"], "metrics": v["metrics"], "notes": v["notes"],
        }
        i = v["metrics"]["i_monotonicity"]
        ii = v["metrics"]["ii_beats_negative"]
        iii = v["metrics"]["iii_beats_random"]
        d_endpoint = i["endpoint_delta_memorypos"]
        mm_neg = ii["memory_minus_negative"] if ii else float("nan")
        mm_rand = iii["memory_minus_random"] if iii else float("nan")
        print(f"{lbl:18} | {v['verdict']:23} | {d_endpoint:+7.3f}  | "
              f"{mm_neg:+6.3f}  | {mm_rand:+7.3f}  | "
              f"ok={sc.get('ok')} d={sc.get('max_abs_diff')}")

    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, indent=2))
    print(f"\n[analyze] wrote summary {summary_out}", file=sys.stderr)

    make_plot(loaded, plot_out)


if __name__ == "__main__":
    main()
