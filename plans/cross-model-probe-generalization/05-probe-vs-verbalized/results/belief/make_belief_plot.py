import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, os

# All three judges at EXAMPLE level (one score per function) — the apples-to-apples
# comparison, because the model's verbalized judgment is intrinsically one yes/no per
# function (no token-level form). Probes max-pool their token scores to example level.
FILES=[("Qwen2.5-Coder","/tmp/r0910/belief_qwen25.json"),
       ("Qwen3-32B","/tmp/r0910/belief_Qwen_Qwen3-32B.json"),
       ("Qwen3.6-27B","/tmp/r0910/belief_Qwen_Qwen3.6-27B.json"),
       ("gemma-3-27b","/tmp/r0910/belief_gemma.json")]
JUDGES=[("general probe","general","#4C72B0"),
        ("family probe","family","#55A868"),
        ("verbalized P(yes)","verbalized","#C44E52")]
fig,axes=plt.subplots(1,2,figsize=(15,5.8),sharey=True)
for ax,famkey,famlabel in [(axes[0],"memory","MEMORY-safety (CWE-416/476/125/787)"),
                           (axes[1],"injection","INJECTION (CWE-089/078/079/022/190)")]:
    x=np.arange(len(FILES)); w=0.26
    for i,(jl,jk,col) in enumerate(JUDGES):
        means=[]; stds=[]
        for _,f in FILES:
            d=json.load(open(f))["families"][famkey]
            means.append(d[f"{jk}_auc_mean"]); stds.append(d[f"{jk}_auc_std"])
        ax.bar(x+(i-1)*w,means,w,yerr=stds,capsize=3,label=jl,color=col,edgecolor='k',lw=.4)
    ax.axhline(0.5,ls='--',c='gray',lw=1.2); ax.text(len(FILES)-0.5,0.505,"chance",color='gray',fontsize=8,va='bottom',ha='right')
    ax.set_xticks(x); ax.set_xticklabels([m for m,_ in FILES],fontsize=9)
    ax.set_title(famlabel,fontsize=11,weight='bold'); ax.set_ylim(0.2,0.88); ax.grid(axis='y',alpha=.25)
axes[0].set_ylabel("example-level AUC (5 seeds, ±std)"); axes[0].legend(fontsize=9,loc='upper left')
fig.suptitle("Belief audit: memory-safety is REPRESENTED but not VERBALIZED\n"
             "family probe recovers memory-vuln (~0.73) while the general probe AND the model's own judgment are at/below chance; "
             "injection is represented AND verbalized",
             fontsize=12.5,weight='bold')
fig.text(0.5,0.005,"All three judges at EXAMPLE level (one score per function) — the apples-to-apples comparison; "
         "verbalized P(yes) is one yes/no per function (no token-level form), probes max-pool token scores to example level.",
         ha='center',fontsize=8,style='italic')
fig.tight_layout(rect=[0,0.03,1,0.90])
out="/tmp/plots/fig8_belief_audit.png"; fig.savefig(out,dpi=130)
dst="/Users/mmtf/p/probes-nlas-cyber/data/plots/cross-model/fig8_belief_audit.png"
os.makedirs(os.path.dirname(dst),exist_ok=True)
import shutil; shutil.copy(out,dst)
print(f"{'model':14} | mem gen/fam/verbal | inj gen/fam/verbal")
for m,f in FILES:
    d=json.load(open(f)); mm=d["families"]["memory"]; jj=d["families"]["injection"]
    print(f"{m:14} | {mm['general_auc_mean']:.3f}/{mm['family_auc_mean']:.3f}/{mm['verbalized_auc_mean']:.3f} | "
          f"{jj['general_auc_mean']:.3f}/{jj['family_auc_mean']:.3f}/{jj['verbalized_auc_mean']:.3f}")
print("wrote",out)
