# [ai-generated]
"""Grouped-bar figure of the SVEN<->PrimeVul cross-dataset transfer (token-AUC)."""
import json, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

R = Path(__file__).resolve().parent / "results"
files = {"Qwen-7B": "metrics_cross_Qwen_Qwen2.5-Coder-7B-Instruct.json",
         "gemma-12b-it": "metrics_cross_google_gemma-3-12b-it.json"}
conds = [("sven_cpp", "sven_cpp_test", "SVEN→SVEN"), ("sven_cpp", "pv_test", "SVEN→PV"),
         ("pv", "pv_test", "PV→PV"), ("pv", "sven_cpp_test", "PV→SVEN")]
fig, ax = plt.subplots(figsize=(8, 4.5)); x = np.arange(len(conds)); w = 0.38
for i, (name, fn) in enumerate(files.items()):
    g = {(r["train"], r["eval"]): r for r in json.loads((R / fn).read_text())["results"]}
    vals = [g[(t, e)]["token_code_auc"] for t, e, _ in conds]
    ax.bar(x + (i - 0.5) * w, vals, w, label=name)
    for xi, v in zip(x + (i - 0.5) * w, vals):
        ax.text(xi, v + 0.005, f"{v:.2f}", ha="center", fontsize=8)
ax.axhline(0.5, color="gray", ls=":", lw=1, label="chance")
ax.set_xticks(x); ax.set_xticklabels([c[2] for c in conds])
ax.set_ylabel("token-code-AUC (honest, C/C++ slice)"); ax.set_ylim(0.45, 0.72)
ax.set_title("exp-22: SVEN↔PrimeVul cross-dataset transfer\n"
             "SVEN→PV ≪ PV→PV, but PV→SVEN ≈ SVEN→SVEN (asymmetric)")
ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3); fig.tight_layout()
fig.savefig(R / "fig_cross_transfer.png", dpi=130)
print("wrote", R / "fig_cross_transfer.png")
