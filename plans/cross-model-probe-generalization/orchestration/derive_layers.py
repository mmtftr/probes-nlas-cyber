#!/usr/bin/env python3
# [ai-generated] Inspect a metrics_layersweep.json: baselines, top layers, and
# derive LAYERS/FEATURESETS/LAYER params for exp 03/04/05 from the NEW sweep.
import json, sys

d = json.load(open(sys.argv[1]))
L = [x for x in d["layers"] if x.get("test_ex_auc") == x.get("test_ex_auc")]  # drop NaN
print(f"model={d['model']}  n_layers={d['n_layers']}  layers_done={d['layers_done']}")
print(f"best_layer(by ex_auc)={d['best_layer']} frac={d['best_layer_frac']:.3f} "
      f"ex_auc={d['best_test_ex_auc']:.4f} tok_auc={d['best_test_tok_auc']:.4f}")
print("baselines:", d["baseline_auc"])
n_test = L[0].get("n_test_ex") if L else None
print("n_test_ex:", n_test)

byex = sorted(L, key=lambda x: -x["test_ex_auc"])[:8]
bytok = sorted(L, key=lambda x: -x["test_tok_auc"])[:8]
print("\nTop8 by EX_AUC:")
for x in byex:
    print(f"  L{x['layer']:2d} frac{x['layer_frac']:.2f} ex={x['test_ex_auc']:.3f} tok={x['test_tok_auc']:.3f}")
print("Top8 by TOK_AUC:")
for x in bytok:
    print(f"  L{x['layer']:2d} frac{x['layer_frac']:.2f} ex={x['test_ex_auc']:.3f} tok={x['test_tok_auc']:.3f}")

# --- derive params for downstream experiments (rank by ex_auc, the headline metric) ---
ranked = [x["layer"] for x in sorted(L, key=lambda x: -x["test_ex_auc"])]
best = ranked[0]
top4 = sorted(ranked[:4])
neigh = sorted({max(0, best - 2), best, min(d["n_layers"] - 1, best + 2)})
print("\n--- derived downstream params ---")
print("LAYER (exp05)      =", best)
print("LAYERS (exp03)     =", ",".join(map(str, top4)))
concat = ",".join(map(str, top4))
single = str(best)
nb = ",".join(map(str, neigh))
print("FEATURESETS (exp04)=", f"{concat};{single};{nb}")
