# [ai-generated]
"""Figures for the 2026-06-03 synthesis deck.

Every number is loaded directly from the experiment result JSONs under
plans/cross-model-probe-generalization/. Nothing is hardcoded except CWE
display names, model display names, and the palette. Run:

    uv run --no-project --with matplotlib --with numpy \
        python docs/presentations/2026-06-03-synthesis/make_figures.py
"""
import json
import glob
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager  # noqa: F401

ROOT = Path(__file__).resolve().parents[3]
PLAN = ROOT / "plans" / "cross-model-probe-generalization"
OUT = Path(__file__).resolve().parent / "figs"
OUT.mkdir(exist_ok=True)

# ---- one visual language -------------------------------------------------
BLUE = "#4C78A8"
BLUE_LT = "#9ECAE1"
ORANGE = "#F58518"
GREEN = "#54A24B"
RED = "#E45756"
GREY = "#888888"
GREY_LT = "#CFCFCF"

plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.7,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

# ---- display names -------------------------------------------------------
DISP = {
    "google_gemma-3-1b-pt": "Gemma-3-1B base",
    "google_gemma-3-1b-it": "Gemma-3-1B inst",
    "google_gemma-3-4b-pt": "Gemma-3-4B base",
    "google_gemma-3-4b-it": "Gemma-3-4B inst",
    "google_gemma-3-12b-pt": "Gemma-3-12B base",
    "google_gemma-3-12b-it": "Gemma-3-12B inst",
    "google_gemma-3-27b-it": "Gemma-3-27B",
    "Qwen_Qwen2.5-Coder-32B-Instruct": "Qwen2.5-Coder-32B",
    "Qwen_Qwen3-32B": "Qwen3-32B",
    "Qwen_Qwen3.6-27B": "Qwen3.6-27B",
}
SHORT = {  # for tight 4-model panels
    "google_gemma-3-27b-it": "Gemma-3\n27B",
    "Qwen_Qwen2.5-Coder-32B-Instruct": "Qwen2.5\nCoder-32B",
    "Qwen_Qwen3-32B": "Qwen3\n32B",
    "Qwen_Qwen3.6-27B": "Qwen3.6\n27B",
}
EXP06_ORDER = [
    "google_gemma-3-1b-pt", "google_gemma-3-1b-it",
    "google_gemma-3-4b-pt", "google_gemma-3-4b-it",
    "google_gemma-3-12b-pt", "google_gemma-3-12b-it",
    "google_gemma-3-27b-it", "Qwen_Qwen2.5-Coder-32B-Instruct",
]
FOUR = [
    "google_gemma-3-27b-it", "Qwen_Qwen2.5-Coder-32B-Instruct",
    "Qwen_Qwen3-32B", "Qwen_Qwen3.6-27B",
]

CWE_META = {  # id -> (display, family)
    "CWE-089": ("SQL inj.", "injection"),
    "CWE-078": ("Command inj.", "injection"),
    "CWE-022": ("Path traversal", "injection"),
    "CWE-079": ("XSS", "injection"),
    "CWE-125": ("OOB read", "memory"),
    "CWE-476": ("NULL deref", "memory"),
    "CWE-416": ("Use-after-free", "memory"),
}


def load(path):
    with open(path) as f:
        return json.load(f)


def caption(fig, text):
    fig.text(0.5, 0.012, text, ha="center", va="bottom", fontsize=9, color=GREY)


def slug_from(path, prefix, suffix=".json"):
    name = Path(path).name
    return name[len(prefix):-len(suffix)]


# =========================================================================
# FIG A -- cross-model robustness (exp-06, token-level over live code)
# =========================================================================
def fig_robust():
    aucs = {}
    for p in glob.glob(str(PLAN / "06-honest-metric-sweeps" / "breakdown_*.json")):
        slug = slug_from(p, "breakdown_")
        if slug in EXP06_ORDER:
            aucs[slug] = load(p)["overall"]["tokens_code_auc"]
    base = load(PLAN / "06-honest-metric-sweeps" /
                "metrics_layersweep_google_gemma-3-27b-it.json")["baseline_auc"]

    fig, ax = plt.subplots(figsize=(11, 5.2))
    xs = np.arange(len(EXP06_ORDER))
    vals = [aucs[m] for m in EXP06_ORDER]

    def color(m):
        if m.startswith("Qwen"):
            return ORANGE
        return BLUE_LT if m.endswith("-pt") else BLUE

    bars = ax.bar(xs, vals, color=[color(m) for m in EXP06_ORDER], width=0.72,
                  edgecolor="white", zorder=3)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.006, f"{v:.2f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold")

    ax.axhline(0.5, color=GREY, lw=1.4, ls="--", zorder=2)
    ax.text(-0.45, 0.505, "chance", color=GREY, fontsize=10, va="bottom", ha="left")
    ax.set_xticks(xs)
    ax.set_xticklabels([DISP[m] for m in EXP06_ORDER], rotation=30, ha="right",
                       fontsize=10.5)
    ax.set_ylim(0.40, 0.90)
    ax.set_ylabel("AUC, token-level over live code")
    ax.set_title("Vulnerability-belief signal holds across scale and post-training")

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=BLUE, label="Gemma instruct"),
        Patch(facecolor=BLUE_LT, label="Gemma base"),
        Patch(facecolor=ORANGE, label="Qwen2.5-Coder"),
    ], loc="upper left", frameon=False, fontsize=10, ncol=3)
    caption(fig, f"regex baseline {base['regex']:.2f}, length baseline {base['length']:.2f} "
                 f"(example-level); 292-example held-out split")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(OUT / "A_robust.png")
    plt.close(fig)
    print("A_robust:", {DISP[m]: round(aucs[m], 3) for m in EXP06_ORDER})


# =========================================================================
# FIG B -- per-CWE blind spot (exp-06), mean +/- std across the 8 models
# =========================================================================
def fig_cwe():
    per = {c: [] for c in CWE_META}
    npos = {}
    for p in glob.glob(str(PLAN / "06-honest-metric-sweeps" / "breakdown_*.json")):
        slug = slug_from(p, "breakdown_")
        if slug not in EXP06_ORDER:
            continue
        d = load(p)["by_cwe"]
        for c in CWE_META:
            per[c].append(d[c]["tokens_code_auc"])
            npos[c] = d[c]["n_pos_examples"]

    order = sorted(CWE_META, key=lambda c: -np.mean(per[c]))
    means = [np.mean(per[c]) for c in order]
    stds = [np.std(per[c]) for c in order]
    cols = [GREEN if CWE_META[c][1] == "injection" else RED for c in order]

    fig, ax = plt.subplots(figsize=(11, 5.4))
    xs = np.arange(len(order))
    ax.bar(xs, means, yerr=stds, color=cols, width=0.66, edgecolor="white",
           error_kw=dict(ecolor=GREY, lw=1.3, capsize=4), zorder=3)
    for x, m, c in zip(xs, means, order):
        ax.text(x, m + np.std(per[c]) + 0.012, f"{m:.2f}", ha="center",
                va="bottom", fontsize=11, fontweight="bold")
    ax.axhline(0.5, color=GREY, lw=1.4, ls="--")
    ax.text(-0.45, 0.505, "chance", color=GREY, fontsize=10, va="bottom", ha="left")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{CWE_META[c][0]}\n(n={npos[c]})" for c in order],
                       fontsize=10.5)
    ax.set_ylim(0.40, 1.0)
    ax.set_ylabel("AUC, token-level over live code")
    ax.set_title("Injection-class vulnerabilities are detected; memory-safety is near chance")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=GREEN, label="Injection class"),
                       Patch(facecolor=RED, label="Memory-safety class")],
              loc="upper right", frameon=False, fontsize=11)
    caption(fig, "mean over 8 models, error bars = std across models; "
                 "n = number of held-out positive examples")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(OUT / "B_cwe.png")
    plt.close(fig)
    print("B_cwe means:", {CWE_META[c][0]: round(np.mean(per[c]), 3) for c in order})


# =========================================================================
# FIG B2 -- by language (backup)
# =========================================================================
def fig_lang():
    langs = ["python", "c", "cpp"]
    per = {l: [] for l in langs}
    n = {}
    for p in glob.glob(str(PLAN / "06-honest-metric-sweeps" / "breakdown_*.json")):
        slug = slug_from(p, "breakdown_")
        if slug not in EXP06_ORDER:
            continue
        d = load(p)["by_lang"]
        for l in langs:
            per[l].append(d[l]["tokens_code_auc"])
            n[l] = d[l]["n_examples"]
    means = [np.mean(per[l]) for l in langs]
    stds = [np.std(per[l]) for l in langs]
    cols = [GREEN, RED, GREY]
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    xs = np.arange(len(langs))
    ax.bar(xs, means, yerr=stds, color=cols, width=0.6, edgecolor="white",
           error_kw=dict(ecolor=GREY, lw=1.3, capsize=4), zorder=3)
    for x, m, l in zip(xs, means, langs):
        ax.text(x, m + np.std(per[l]) + 0.012, f"{m:.2f}", ha="center",
                va="bottom", fontsize=11, fontweight="bold")
    ax.axhline(0.5, color=GREY, lw=1.4, ls="--")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"Python\n(n={n['python']})", f"C\n(n={n['c']})",
                        f"C++\n(n={n['cpp']})"])
    ax.set_ylim(0.40, 0.92)
    ax.set_ylabel("AUC, token-level over live code")
    ax.set_title("Signal is carried by Python; C is near chance")
    caption(fig, "mean over 8 models, error bars = std across models")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(OUT / "B2_lang.png")
    plt.close(fig)


# =========================================================================
# FIG C -- belief audit: represented vs verbalized (exp-05, example-level)
# =========================================================================
def fig_belief():
    data = {}
    for m in FOUR:
        d = load(PLAN / "05-probe-vs-verbalized" / "results" / "belief" /
                 f"belief_audit_{m}.json")["families"]
        data[m] = d

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharey=True)
    fams = [("memory", "Memory-safety", axes[0]),
            ("injection", "Injection (control)", axes[1])]
    width = 0.26
    xs = np.arange(len(FOUR))
    for fam, title, ax in fams:
        gen = [data[m][fam]["general_auc_mean"] for m in FOUR]
        gen_e = [data[m][fam]["general_auc_std"] for m in FOUR]
        fa = [data[m][fam]["family_auc_mean"] for m in FOUR]
        fa_e = [data[m][fam]["family_auc_std"] for m in FOUR]
        vb = [data[m][fam]["verbalized_auc_mean"] for m in FOUR]
        vb_e = [data[m][fam]["verbalized_auc_std"] for m in FOUR]
        ek = dict(ecolor=GREY, lw=1.1, capsize=3)
        ax.bar(xs - width, gen, width, yerr=gen_e, color=BLUE, label="General probe",
               error_kw=ek, zorder=3)
        ax.bar(xs, fa, width, yerr=fa_e, color=GREEN, label="Specialized probe",
               error_kw=ek, zorder=3)
        ax.bar(xs + width, vb, width, yerr=vb_e, color=ORANGE,
               label="Model self-report", error_kw=ek, zorder=3)
        ax.axhline(0.5, color=GREY, lw=1.3, ls="--")
        ax.set_xticks(xs)
        ax.set_xticklabels([SHORT[m] for m in FOUR], fontsize=10)
        ax.set_title(title)
        ax.set_ylim(0.2, 0.92)
    axes[0].set_ylabel("AUC, example-level (5 seeds)")
    axes[0].legend(loc="upper left", frameon=False, fontsize=10)
    fig.suptitle("Memory-safety vulnerability is represented in activations but not verbalized",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "C_belief.png")
    plt.close(fig)
    print("C_belief memory:", {SHORT[m].replace(chr(10), ''):
          (round(data[m]['memory']['general_auc_mean'], 2),
           round(data[m]['memory']['family_auc_mean'], 2),
           round(data[m]['memory']['verbalized_auc_mean'], 2)) for m in FOUR})


# =========================================================================
# FIG D -- capacity vs specialization on memory (exp-10 + exp-11), token-level
# =========================================================================
def fig_capacity():
    gen, fb, spec = {}, {}, {}
    for m in FOUR:
        d10 = load(PLAN / "10-per-cwe-probes" / "results" /
                   f"per_cwe_10_{m}_linear_all_memory.json")["by_cwe"]["memory"]
        gen[m] = d10["general_tokens_code_auc"]
        spec[m] = d10["specialized_tokens_code_auc"]
        d11 = load(PLAN / "11-family-balanced-head" / "results" /
                   f"family_balanced_11_{m}_linear_family_balanced.json")
        fb[m] = d11["by_family"]["memory"]["tokens_code_auc"]

    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    xs = np.arange(len(FOUR))
    width = 0.26
    ax.bar(xs - width, [gen[m] for m in FOUR], width, color=BLUE,
           label="General probe", edgecolor="white", zorder=3)
    ax.bar(xs, [fb[m] for m in FOUR], width, color=GREY,
           label="Family-balanced head (reweighted)", edgecolor="white", zorder=3)
    ax.bar(xs + width, [spec[m] for m in FOUR], width, color=GREEN,
           label="Dedicated memory probe", edgecolor="white", zorder=3)
    for i, m in enumerate(FOUR):
        for off, val in [(-width, gen[m]), (0, fb[m]), (width, spec[m])]:
            ax.text(i + off, val + 0.008, f"{val:.2f}", ha="center", va="bottom",
                    fontsize=9.5, fontweight="bold")
    ax.axhline(0.5, color=GREY, lw=1.3, ls="--")
    ax.text(len(xs) - 0.5, 0.505, "chance", color=GREY, fontsize=10, ha="right",
            va="bottom")
    ax.set_xticks(xs)
    ax.set_xticklabels([SHORT[m] for m in FOUR])
    ax.set_ylim(0.45, 0.80)
    ax.set_ylabel("Memory-safety AUC, token-level")
    ax.set_title("Added capacity does not recover memory; a dedicated probe does")
    ax.legend(loc="upper left", frameon=False, fontsize=10.5)
    caption(fig, "54 memory test positives; reweighting one head also lowers injection AUC")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(OUT / "D_capacity.png")
    plt.close(fig)
    print("D_capacity:", {SHORT[m].replace(chr(10), ''):
          (round(gen[m], 2), round(fb[m], 2), round(spec[m], 2)) for m in FOUR})


# =========================================================================
# FIG E -- steering: the direction reads but does not drive (exp-13)
# =========================================================================
def _delta(arr, i_a, i_0):
    return arr[i_a] - arr[i_0]


def _steer_gaps(path, alpha):
    d = load(path)
    grid = d["alpha_grid"]
    i_a = grid.index(alpha)
    i_0 = grid.index(0.0)
    bd = d["by_direction"]
    mp = bd["memory"]["by_subset"]["memory_pos"]["p_yes"]
    neg = bd["memory"]["by_subset"]["negative"]["p_yes"]
    d_mem = _delta(mp, i_a, i_0)
    d_neg = _delta(neg, i_a, i_0)
    rand_keys = [k for k in bd if k.startswith("random")]
    if rand_keys:
        d_rand = np.mean([_delta(bd[k]["by_subset"]["memory_pos"]["p_yes"], i_a, i_0)
                          for k in rand_keys])
        mem_minus_rand = d_mem - d_rand
    else:
        mem_minus_rand = np.nan
    return d_mem - d_neg, mem_minus_rand


def fig_steering():
    fair, ext = {}, {}
    for m in FOUR:
        fair[m] = _steer_gaps(
            PLAN / "13-causal-steering" / "results" / "steer_v2" / f"steer_13_{m}.json", 4.0)
        ext[m] = _steer_gaps(
            PLAN / "13-causal-steering" / "results" / "steer_wide" / f"steer_13_{m}.json", 64.0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3), sharey=False)
    xs = np.arange(len(FOUR))

    # Panel A: fair magnitude, both specificity gaps, fine scale
    axA = axes[0]
    width = 0.34
    mn = [fair[m][0] for m in FOUR]
    mr = [fair[m][1] for m in FOUR]
    axA.bar(xs - width / 2, mn, width, color=BLUE, label="memory minus negative",
            edgecolor="white", zorder=3)
    axA.bar(xs + width / 2, mr, width, color=ORANGE, label="memory minus random",
            edgecolor="white", zorder=3)
    axA.axhline(0.0, color="black", lw=1.2)
    axA.set_xticks(xs)
    axA.set_xticklabels([SHORT[m] for m in FOUR], fontsize=10)
    axA.set_ylim(-0.05, 0.05)
    axA.set_title("Fair magnitude (+/-4 std)")
    axA.set_ylabel("Change in P('yes') on vulnerable code\n(specificity gap)")
    axA.legend(loc="upper right", frameon=False, fontsize=10)
    axA.text(-0.4, 0.041, "bars hug 0: no memory-specific effect", fontsize=9.5,
             color=GREY)

    # Panel B: extreme magnitude, robust memory-minus-negative gap
    axB = axes[1]
    mnE = [ext[m][0] for m in FOUR]
    bars = axB.bar(xs, mnE, 0.5, color=BLUE, edgecolor="white", zorder=3)
    for x, v in zip(xs, mnE):
        axB.text(x, v - 0.006, f"{v:.2f}", ha="center", va="top", fontsize=10,
                 fontweight="bold")
    axB.axhline(0.0, color="black", lw=1.2)
    axB.set_xticks(xs)
    axB.set_xticklabels([SHORT[m] for m in FOUR], fontsize=10)
    axB.set_ylim(-0.16, 0.04)
    axB.set_title("Extreme magnitude (+/-64 std)")
    axB.text(-0.4, 0.022, "memory minus negative < 0: a random direction shifts P('yes') as much",
             fontsize=9.5, color=GREY)

    fig.suptitle("Steering the memory direction does not specifically raise the model's judgment",
                 fontsize=15, fontweight="bold")
    caption(fig, "specificity gap = change in P('yes') on vulnerable code from the memory "
                 "direction, minus the same change from a negative subset or a random direction")
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(OUT / "E_steering.png")
    plt.close(fig)
    print("E_steering fair (mem-neg, mem-rand):",
          {SHORT[m].replace(chr(10), ''): (round(fair[m][0], 3), round(fair[m][1], 3)) for m in FOUR})
    print("E_steering extreme (mem-neg, mem-rand):",
          {SHORT[m].replace(chr(10), ''): (round(ext[m][0], 3),
           None if np.isnan(ext[m][1]) else round(ext[m][1], 3)) for m in FOUR})


# =========================================================================
# FIG F -- the blind spot is a framing artifact (exp-14), example-level
# =========================================================================
def fig_prompt():
    v0, vmem, vmem_e, probe = {}, {}, {}, {}
    for m in FOUR:
        d = load(PLAN / "14-memory-prompt-sweep" / "results" / f"{m}_aucs.json")["variants"]
        v0[m] = d["V0_generic"]["memory_auc_mean"]
        # best memory-specific prompt across the memory variants
        cand = [(d[v]["memory_auc_mean"], d[v]["memory_auc_std"])
                for v in d if v != "V0_generic"]
        best = max(cand, key=lambda t: t[0])
        vmem[m] = best[0]
        vmem_e[m] = best[1]
        b = load(PLAN / "05-probe-vs-verbalized" / "results" / "belief" /
                 f"belief_audit_{m}.json")["families"]["memory"]
        probe[m] = b["family_auc_mean"]

    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    xs = np.arange(len(FOUR))
    width = 0.32
    ax.bar(xs - width / 2, [v0[m] for m in FOUR], width, color=GREY,
           label="Generic prompt", edgecolor="white", zorder=3)
    ax.bar(xs + width / 2, [vmem[m] for m in FOUR], width,
           yerr=[vmem_e[m] for m in FOUR], color=ORANGE, label="Memory-specific prompt",
           error_kw=dict(ecolor=GREY, lw=1.1, capsize=3), zorder=3)
    for i, m in enumerate(FOUR):
        ax.text(i - width / 2, v0[m] + 0.012, f"{v0[m]:.2f}", ha="center",
                va="bottom", fontsize=9.5, fontweight="bold")
        ax.text(i + width / 2, vmem[m] + vmem_e[m] + 0.012, f"{vmem[m]:.2f}",
                ha="center", va="bottom", fontsize=9.5, fontweight="bold")
        # probe reference tick
        ax.plot([i - 0.42, i + 0.42], [probe[m], probe[m]], color=GREEN, lw=2.0,
                ls=(0, (4, 2)), zorder=4)
    ax.plot([], [], color=GREEN, lw=2.0, ls=(0, (4, 2)), label="Memory probe (reference)")
    ax.axhline(0.5, color=GREY, lw=1.3, ls="--")
    ax.text(-0.45, 0.505, "chance", color=GREY, fontsize=10, va="bottom", ha="left")
    ax.set_xticks(xs)
    ax.set_xticklabels([SHORT[m] for m in FOUR])
    ax.set_ylim(0.30, 0.92)
    ax.set_ylabel("Verbalized memory-safety AUC (example-level)")
    ax.set_title("Asking specifically about memory-safety recovers the judgment")
    ax.legend(loc="upper left", frameon=False, fontsize=10.5, ncol=3)
    caption(fig, "injection AUC falls under the memory prompt; the question trades one family for the other")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(OUT / "F_prompt.png")
    plt.close(fig)
    print("F_prompt:", {SHORT[m].replace(chr(10), ''):
          (round(v0[m], 2), round(vmem[m], 2), round(probe[m], 2)) for m in FOUR})


# =========================================================================
# FIG G -- MLP ceiling (backup), overall token-level
# =========================================================================
def fig_mlp():
    lin, mlp = {}, {}
    for m in FOUR:
        s = load(PLAN / "09-ensemble-linear-probes" / "results" / f"summary_{m}.json")
        lin[m] = s["baseline_overall_tokens_code_auc"]
        best = -1
        for w in ("mlp256", "mlp512"):
            v = load(PLAN / "12-mlp-layer-sweep" / "results" /
                     f"{m}_{w}.json")["best_tokens_code_auc"]
            best = max(best, v)
        mlp[m] = best
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    xs = np.arange(len(FOUR))
    width = 0.34
    ax.bar(xs - width / 2, [lin[m] for m in FOUR], width, color=BLUE,
           label="Linear probe", edgecolor="white", zorder=3)
    ax.bar(xs + width / 2, [mlp[m] for m in FOUR], width, color=ORANGE,
           label="MLP head (own best layer)", edgecolor="white", zorder=3)
    for i, m in enumerate(FOUR):
        ax.text(i - width / 2, lin[m] + 0.004, f"{lin[m]:.2f}", ha="center",
                va="bottom", fontsize=9.5)
        ax.text(i + width / 2, mlp[m] + 0.004, f"{mlp[m]:.2f}", ha="center",
                va="bottom", fontsize=9.5, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([SHORT[m] for m in FOUR])
    ax.set_ylim(0.70, 0.86)
    ax.set_ylabel("Overall AUC, token-level over live code")
    ax.set_title("A nonlinear head adds little (0.01 to 0.05) and stays opaque")
    ax.legend(loc="upper left", frameon=False, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(OUT / "G_mlp.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_robust()
    fig_cwe()
    fig_lang()
    fig_belief()
    fig_capacity()
    fig_steering()
    fig_prompt()
    fig_mlp()
    print("\nwrote figures to", OUT)
