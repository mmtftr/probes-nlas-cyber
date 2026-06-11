# [ai-generated]
"""Aggregate the 5-fold × 3-seed CV (metrics_cv.json per model) into mean±std
tables + an error-bar plot. Reads results/cv_<slug>/metrics_cv.json (each carries
a pre-computed `aggregate`)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OP = {
    "Qwen_Qwen2.5-Coder-32B-Instruct": 25, "Qwen_Qwen2.5-Coder-7B-Instruct": 16,
    "google_gemma-3-1b-it": 25, "google_gemma-3-4b-it": 7,
    "google_gemma-3-12b-it": 15, "google_gemma-3-27b-it": 19, "google_gemma-3-12b-pt": 13,
}
SHORT = {k: k.replace("Qwen_Qwen2.5-Coder-", "Qwen-").replace("google_gemma-3-", "g3-") for k in OP}
BACKEND = {k: ("vLLM" if k.startswith("Qwen") else "HF") for k in OP}


def load():
    out = {}
    for d in sorted(RES.glob("cv_*")):
        p = d / "metrics_cv.json"
        if p.is_file():
            out[d.name.replace("cv_", "")] = json.loads(p.read_text())["aggregate"]
    return out


def cell(rec, key):
    m, s = rec.get(f"{key}_mean"), rec.get(f"{key}_std")
    return f"{m:.3f}±{s:.3f}" if m == m else "  –  "


def main():
    data = load()
    if not data:
        print("no metrics_cv.json yet"); return
    lines = ["[ai-generated]", "", "# 19 subtractive-regime — CV results (5-fold × 3-seed = 15 train/evals per config)", "",
             "mean±std over folds. Honest eval (tight-token ∩ is_code, code-only). "
             "Backend: Qwen=vLLM, gemma-3=HF (no vLLM EAGLE3 in 0.22.1).", ""]
    for slug, agg in data.items():
        L = OP.get(slug)
        g = {(r["train_subset"], r["granularity"], r["negatives"]): r for r in agg if r["layer"] == L}
        lines += [f"## {SHORT.get(slug, slug)}  (L{L}, {BACKEND.get(slug)})", "",
                  "| train | gran | neg | sub-test AUC | base-test AUC | pairAcc sub | pairAcc add |",
                  "|---|---|---|---|---|---|---|"]
        for subset in ("base", "subtractive"):
            for gran in ("line", "token"):
                for neg in ("Y", "X"):
                    r = g.get((subset, gran, neg))
                    if not r:
                        continue
                    lines.append(f"| {subset} | {gran} | {neg} | {cell(r,'sub_test_code_auc')} | "
                                 f"{cell(r,'base_test_code_auc')} | {cell(r,'pair_acc_sub')} | {cell(r,'pair_acc_add')} |")
        lines.append("")
    # cross-comparison token/X
    lines += ["## Cross-comparison (token, X) — mean±std", "",
              "| model | bk | base→sub | sub→sub | base→base | sub→base | pairAcc-add (sub) |",
              "|---|---|---|---|---|---|---|"]
    for slug, agg in data.items():
        L = OP.get(slug)
        g = {(r["train_subset"], r["granularity"], r["negatives"]): r for r in agg if r["layer"] == L}
        b, s = g.get(("base", "token", "X")), g.get(("subtractive", "token", "X"))
        if not b or not s:
            continue
        lines.append(f"| {SHORT.get(slug, slug)} | {BACKEND.get(slug)} | {cell(b,'sub_test_code_auc')} | "
                     f"{cell(s,'sub_test_code_auc')} | {cell(b,'base_test_code_auc')} | "
                     f"{cell(s,'base_test_code_auc')} | {cell(s,'pair_acc_add')} |")
    (HERE / "RESULTS_CV.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # error-bar plot: base vs subtractive trained on sub-test (token, X)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        models, bm, bs, sm, ss = [], [], [], [], []
        for slug, agg in data.items():
            L = OP.get(slug)
            g = {(r["train_subset"], r["granularity"], r["negatives"]): r for r in agg if r["layer"] == L}
            b, s = g.get(("base", "token", "X")), g.get(("subtractive", "token", "X"))
            if not b or not s:
                continue
            models.append(SHORT.get(slug, slug))
            bm.append(b["sub_test_code_auc_mean"]); bs.append(b["sub_test_code_auc_std"])
            sm.append(s["sub_test_code_auc_mean"]); ss.append(s["sub_test_code_auc_std"])
        x = np.arange(len(models)); w = 0.38
        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.bar(x - w/2, bm, w, yerr=bs, capsize=3, label="base-trained")
        ax.bar(x + w/2, sm, w, yerr=ss, capsize=3, label="subtractive-trained")
        ax.set_xticks(x); ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("subtractive-test token-code-AUC")
        ax.set_title("CV (5-fold×3-seed): base vs subtractive training on the clean test (token, X)  ·  error bars = ±1 std")
        ax.set_ylim(0.5, 0.9); ax.legend(); ax.grid(axis="y", alpha=.3)
        fig.tight_layout(); fig.savefig(HERE / "fig_cv.png", dpi=130)
        print("[analyze_cv] wrote fig_cv.png")
    except Exception as e:
        print(f"[analyze_cv] plot skipped: {e}")


if __name__ == "__main__":
    main()
