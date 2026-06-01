import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, os

MODELS=[("Qwen2.5-Coder","Qwen_Qwen2.5-Coder-32B-Instruct"),
        ("Qwen3-32B","Qwen_Qwen3-32B"),
        ("Qwen3.6-27B","Qwen_Qwen3.6-27B"),
        ("gemma-3-27b","google_gemma-3-27b-it")]

def get(model_slug, fam):
    pc=json.load(open(f"/tmp/r0910/pc10_{model_slug}_{fam}.json"))["by_cwe"][fam]
    bel=json.load(open(f"/tmp/r0910/belief_{ 'qwen25' if 'Qwen2.5' in model_slug else ('gemma' if 'gemma' in model_slug else model_slug)}.json"))["families"][fam]
    return (pc["general_tokens_code_auc"], pc["specialized_tokens_code_auc"],
            bel["verbalized_auc_mean"], bel["verbalized_auc_std"])

JUDGES=[("general probe (tokens_code)","#4C72B0",False),
        ("family probe (tokens_code)","#55A868",False),
        ("verbalized P(yes) — EXAMPLE-level","#C44E52",True)]
fig,axes=plt.subplots(1,2,figsize=(15,5.8),sharey=True)
for ax,fam,famlabel in [(axes[0],"memory","MEMORY-safety (CWE-416/476/125/787)"),
                        (axes[1],"injection","INJECTION (CWE-089/078/079/022/190)")]:
    x=np.arange(len(MODELS)); w=0.26
    cols=[[],[],[]]; errs=[[],[],[]]
    for _,slug in MODELS:
        g,fp,vb,vbs=get(slug,fam)
        cols[0].append(g); cols[1].append(fp); cols[2].append(vb)
        errs[0].append(0); errs[1].append(0); errs[2].append(vbs)
    for i,(lab,col,hatch) in enumerate(JUDGES):
        ax.bar(x+(i-1)*w,cols[i],w,yerr=errs[i],capsize=3,label=lab,color=col,
               edgecolor='k',lw=.5,hatch=('///' if hatch else None))
    ax.axhline(0.5,ls='--',c='gray',lw=1.2); ax.text(len(MODELS)-0.5,0.505,"chance",color='gray',fontsize=8,va='bottom',ha='right')
    ax.set_xticks(x); ax.set_xticklabels([m for m,_ in MODELS],fontsize=9)
    ax.set_title(famlabel,fontsize=11,weight='bold'); ax.set_ylim(0.3,0.92); ax.grid(axis='y',alpha=.25)
axes[0].set_ylabel("AUC"); axes[0].legend(fontsize=8.5,loc='upper left')
fig.suptitle("Belief audit: memory-safety is REPRESENTED (token-level tokens_code) but not VERBALIZED\n"
             "probes read memory-vuln from activations (family 0.66–0.73 > general 0.52–0.59, tokens_code) while the model's own judgment is "
             "at/below chance (verbalized 0.39–0.55, example-level); injection is represented AND verbalized",
             fontsize=11.5,weight='bold')
fig.text(0.5,0.005,"NOTE: probes are token-level tokens_code AUC (our standard metric); verbalized P(yes) is necessarily EXAMPLE-level "
         "(one yes/no per function — no token-level form). Hatched bar = example-level.",ha='center',fontsize=8,style='italic')
fig.tight_layout(rect=[0,0.03,1,0.90])
out="/tmp/plots/fig8_belief_audit.png"; fig.savefig(out,dpi=130)
dst="/Users/mmtf/p/probes-nlas-cyber/data/plots/cross-model/fig8_belief_audit.png"
os.makedirs(os.path.dirname(dst),exist_ok=True)
import shutil; shutil.copy(out,dst)
# print the table too
print(f"{'model':14} | mem: gen/fam(tc) verbal(ex) | inj: gen/fam(tc) verbal(ex)")
for m,slug in MODELS:
    gm,fm,vm,_=get(slug,"memory"); gi,fi,vi,_=get(slug,"injection")
    print(f"{m:14} | {gm:.3f}/{fm:.3f}  {vm:.3f}      | {gi:.3f}/{fi:.3f}  {vi:.3f}")
print("wrote",out)
