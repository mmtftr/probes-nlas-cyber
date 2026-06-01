import json, glob, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np

ROOT = "/tmp/r0910/cosine"
MCOL = {"Qwen_Qwen2.5-Coder-32B-Instruct": "#4C72B0", "Qwen_Qwen3-32B": "#55A868",
        "Qwen_Qwen3.6-27B": "#C44E52", "google_gemma-3-27b-it": "#8172B3"}
SHORT = {"Qwen_Qwen2.5-Coder-32B-Instruct": "Qwen2.5-Coder", "Qwen_Qwen3-32B": "Qwen3-32B",
         "Qwen_Qwen3.6-27B": "Qwen3.6-27B", "google_gemma-3-27b-it": "gemma-3-27b"}
# single-linear baselines at each model's best layer (handoff table) for reference
LINBASE = {"Qwen_Qwen2.5-Coder-32B-Instruct": 0.788, "Qwen_Qwen3-32B": 0.806,
           "Qwen_Qwen3.6-27B": 0.787, "google_gemma-3-27b-it": 0.770}
AGGSTYLE = {"logsumexp": "-", "max": "--"}

# data[slug][agg] = list of (lambda, cos, tokens_code) sorted by lambda
data = {}
for slug in MCOL:
    d = os.path.join(ROOT, slug)
    if not os.path.isdir(d):
        continue
    for f in glob.glob(os.path.join(d, "K8_*_lam*.json")):
        r = json.load(open(f))
        if "overall" not in r:
            continue
        data.setdefault(slug, {}).setdefault(r["agg"], []).append(
            (float(r["div_lambda"]), float(r["post_train_cos_abs_mean"]),
             float(r["overall"]["tokens_code_auc"])))
for slug in data:
    for agg in data[slug]:
        data[slug][agg].sort()

# x positions: map lambdas {0,1e-3,1e-2,1e-1,3e-1} to even spacing on a pseudo-log axis
LAMS = [0.0, 0.001, 0.01, 0.1, 0.3]
XPOS = {l: i for i, l in enumerate(LAMS)}
XLAB = ["0", "1e-3", "1e-2", "1e-1", "3e-1"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
for slug, byagg in sorted(data.items()):
    for agg, rows in sorted(byagg.items()):
        xs = [XPOS[l] for l, _, _ in rows]
        cos = [c for _, c, _ in rows]
        tc = [t for _, _, t in rows]
        lbl = f"{SHORT[slug]} ({agg})"
        axes[0].plot(xs, cos, AGGSTYLE[agg], color=MCOL[slug], marker="o", ms=5, label=lbl)
        axes[1].plot(xs, tc, AGGSTYLE[agg], color=MCOL[slug], marker="o", ms=5, label=lbl)
    axes[1].axhline(LINBASE[slug], color=MCOL[slug], ls=":", lw=1, alpha=0.5)

for ax in axes:
    ax.set_xticks(list(XPOS.values())); ax.set_xticklabels(XLAB)
    ax.set_xlabel("divergence penalty weight  λ")
axes[0].set_ylabel("mean |cosine| between K=8 directions")
axes[0].set_title("Penalty WORKS: λ↑ orthogonalises directions (|cos|↓)")
axes[1].set_ylabel("test tokens_code AUC (overall)")
axes[1].set_title("But AUC does NOT rise (dotted = single-linear baseline)")
axes[1].legend(fontsize=7, ncol=2, loc="lower left")
fig.suptitle("Exp-09 cosine-divergence sweep — forcing direction diversity does NOT improve the probe\n"
             "(orthogonalising K=8 directions lowers |cos| but leaves tokens_code flat ⇒ collapse is NOT the bottleneck)",
             fontsize=12, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.92])
out = "/tmp/plots/fig7_cosine.png"
fig.savefig(out, dpi=130)
dst = "/Users/mmtf/p/probes-nlas-cyber/data/plots/cross-model/fig7_cosine.png"
os.makedirs(os.path.dirname(dst), exist_ok=True)
import shutil; shutil.copy(out, dst)
n = sum(len(v) for byagg in data.values() for v in byagg.values())
print(f"plotted {n} cells across {len(data)} models -> {out}")
