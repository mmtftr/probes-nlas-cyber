import json, glob, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np

MODELS=[("Qwen2.5-Coder","mat_Qwen_Qwen2.5-Coder-32B-Instruct.json"),
        ("Qwen3-32B","mat_qwen3-32b.json"),
        ("Qwen3.6-27B","mat_Qwen_Qwen3.6-27B.json"),
        ("gemma-3-27b","mat_google_gemma-3-27b-it.json")]
ROOT="/tmp/r0910"
MEMBERS=["general","memory","injection","ind-ensemble","cat-ensemble"]
CELLS=["memory","injection","overall"]
def auc(M,side,m,c):
    try:
        v=M[side][m][c]; return v["auc_mean"] if isinstance(v,dict) else v
    except Exception: return np.nan

# collect: data[side][cell][member] = list over models
loaded=[]
data={s:{c:{m:[] for m in MEMBERS} for c in CELLS} for s in ("probe","verbalized")}
for tag,f in MODELS:
    p=os.path.join(ROOT,f)
    if not os.path.exists(p): continue
    M=json.load(open(p))["matrix"]; loaded.append(tag)
    for s in ("probe","verbalized"):
        for c in CELLS:
            for m in MEMBERS:
                data[s][c][m].append(auc(M,s,m,c))
n=len(loaded)
fig,axes=plt.subplots(1,3,figsize=(16,5.6),sharey=True)
x=np.arange(len(MEMBERS)); w=0.38
for ax,c,title in zip(axes,CELLS,["MEMORY cell (mem-pos ∪ neg)","INJECTION cell","OVERALL (all-pos ∪ neg)"]):
    for i,(side,col,hatch) in enumerate([("probe","#4C72B0",None),("verbalized","#C44E52","///")]):
        means=[np.nanmean(data[side][c][m]) for m in MEMBERS]
        stds=[np.nanstd(data[side][c][m]) for m in MEMBERS]
        ax.bar(x+(i-0.5)*w, means, w, yerr=stds, capsize=3, label=side, color=col, edgecolor='k', lw=.5, hatch=hatch)
    ax.axhline(0.5,ls='--',c='gray',lw=1.1); ax.set_xticks(x); ax.set_xticklabels(MEMBERS,rotation=25,ha='right',fontsize=8.5)
    ax.set_title(title,fontsize=11,weight='bold'); ax.set_ylim(0.2,0.92); ax.grid(axis='y',alpha=.25)
axes[0].set_ylabel(f"example-level AUC (mean of {n} models, ±std)"); axes[0].legend(fontsize=10,loc='upper left')
fig.suptitle(f"Specialized PROBE ≈ specialized PROMPT: the verbalized analogue of a probe is a prompt (max-combine, {n} models)\n"
             "ind-ensemble (family-aware max over per-CWE members) recovers memory on BOTH sides where 'general' fails; "
             "cat-ensemble is contaminated on the memory cell",
             fontsize=11.5,weight='bold')
fig.text(0.5,0.005,"probe member = pooled/per-CWE span-max probe (example score = max-pooled tokens); verbalized member = the matching yes/no PROMPT, P(yes). "
         "ind-ensemble = MAX over per-CWE members; cat-ensemble = MAX(memory,injection).",ha='center',fontsize=8,style='italic')
fig.tight_layout(rect=[0,0.03,1,0.90])
out="/tmp/plots/fig9_ensemble_matrix.png"; fig.savefig(out,dpi=130)
dst="/Users/mmtf/p/probes-nlas-cyber/data/plots/cross-model/fig9_ensemble_matrix.png"
os.makedirs(os.path.dirname(dst),exist_ok=True)
import shutil; shutil.copy(out,dst)
print(f"plotted {n} models: {loaded} -> {out}")
PY_=0
