# [ai-generated]
"""Collect metrics_grid.json from each model and build the cross-comparison
tables + plots for the subtractive-regime experiment.

Reads results/subtractive_<slug>/metrics_grid.json (rsynced from cluster).
Writes:
  RESULTS_TABLES.md   — per-model operating-layer 8-config table + cross-subset summary
  fig_cross.png       — base-trained vs subtractive-trained, on each eval set
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"

OP = {  # historical operating layer per model (exp-16 RESULTS.md)
    "Qwen_Qwen2.5-Coder-32B-Instruct": 25, "Qwen_Qwen2.5-Coder-7B-Instruct": 16,
    "google_gemma-3-1b-it": 25, "google_gemma-3-4b-it": 7,
    "google_gemma-3-12b-it": 15, "google_gemma-3-27b-it": 19, "google_gemma-3-12b-pt": 13,
}
SHORT = {k: k.replace("Qwen_Qwen2.5-Coder-", "Qwen-").replace("google_gemma-3-", "g3-") for k in OP}


def load():
    out = {}
    for d in sorted(RES.glob("subtractive_*")):
        slug = d.name.replace("subtractive_", "")
        p = d / "metrics_grid.json"
        if p.is_file():
            out[slug] = json.loads(p.read_text())["results"]
    return out


def at_layer(recs, layer):
    return {(r["train_subset"], r["granularity"], r["negatives"]): r
            for r in recs if r["layer"] == layer}


def best_layer(recs, key, metric="eval_subtractive_test_code_auc"):
    cand = [r for r in recs if (r["train_subset"], r["granularity"], r["negatives"]) == key
            and r[metric] == r[metric]]
    return max(cand, key=lambda r: r[metric]) if cand else None


def fmt(x):
    return f"{x:.3f}" if isinstance(x, float) and x == x else "  –  "


def main():
    data = load()
    if not data:
        print("no metrics_grid.json yet under results/"); return
    lines = ["[ai-generated]", "", "# 19 subtractive-regime — result tables", ""]
    for slug, recs in data.items():
        L = OP.get(slug)
        g = at_layer(recs, L)
        lines += [f"## {SHORT.get(slug, slug)}  (operating layer L{L})", "",
                  "| train_subset | gran | neg | sub-test tokAUC | base-test tokAUC | pairAcc sub | pairAcc add |",
                  "|---|---|---|---|---|---|---|"]
        for subset in ("base", "subtractive"):
            for gran in ("line", "token"):
                for neg in ("Y", "X"):
                    r = g.get((subset, gran, neg))
                    if not r:
                        continue
                    lines.append(f"| {subset} | {gran} | {neg} | "
                                 f"{fmt(r['eval_subtractive_test_code_auc'])} | "
                                 f"{fmt(r['eval_base_test_code_auc'])} | "
                                 f"{fmt(r.get('pair_acc_subtractive'))} | "
                                 f"{fmt(r.get('pair_acc_additive'))} |")
        lines.append("")
    # cross-subset headline: token/X, base-trained vs subtractive-trained
    lines += ["## Cross-comparison (granularity=token, negatives=X), operating layer", "",
              "| model | base→sub-test | sub→sub-test | base→base-test | sub→base-test | base pairAcc-add | sub pairAcc-add |",
              "|---|---|---|---|---|---|---|"]
    for slug, recs in data.items():
        L = OP.get(slug); g = at_layer(recs, L)
        b = g.get(("base", "token", "X")); s = g.get(("subtractive", "token", "X"))
        if not b or not s:
            continue
        lines.append(f"| {SHORT.get(slug, slug)} | {fmt(b['eval_subtractive_test_code_auc'])} | "
                     f"{fmt(s['eval_subtractive_test_code_auc'])} | {fmt(b['eval_base_test_code_auc'])} | "
                     f"{fmt(s['eval_base_test_code_auc'])} | {fmt(b.get('pair_acc_additive'))} | "
                     f"{fmt(s.get('pair_acc_additive'))} |")
    (HERE / "RESULTS_TABLES.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[analyze] models: {list(data)}")

    # plot: base vs subtractive trained (token,X) on sub-test, per model
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        models = [SHORT.get(s, s) for s in data]
        bt, st = [], []
        for slug, recs in data.items():
            g = at_layer(recs, OP.get(slug))
            b = g.get(("base", "token", "X")); s = g.get(("subtractive", "token", "X"))
            bt.append(b["eval_subtractive_test_code_auc"] if b else np.nan)
            st.append(s["eval_subtractive_test_code_auc"] if s else np.nan)
        x = np.arange(len(models)); w = 0.38
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.bar(x - w/2, bt, w, label="base-trained")
        ax.bar(x + w/2, st, w, label="subtractive-trained")
        ax.set_xticks(x); ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("subtractive-test token-code-AUC (honest)")
        ax.set_title("Cross-comparison: base vs subtractive training (token, X) on the clean subtractive test")
        ax.set_ylim(0.5, 0.9); ax.legend(); ax.grid(axis="y", alpha=.3)
        fig.tight_layout(); fig.savefig(HERE / "fig_cross.png", dpi=130)
        print(f"[analyze] wrote fig_cross.png")
    except Exception as e:
        print(f"[analyze] plot skipped: {e}")


if __name__ == "__main__":
    main()
