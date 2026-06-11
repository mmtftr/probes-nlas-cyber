# TODO

Running to-do list for the probes/NLAs project. Append; check items off in place.

## Probe error analysis

- [x] **Per-CWE-trained probes (token AUC) — ALREADY DONE in exp-10.**
  `plans/cross-model-probe-generalization/10-per-cwe-probes/` trains per-CWE
  specialized probes vs the general probe and reports `tokens_code_auc`.
  **Memory signal EXISTS:** specialized lifts memory substantially (Qwen-32B L25:
  CWE-125 0.571→0.732, CWE-416 0.435→0.766, CWE-787 0.459→0.670 low-n, CWE-476
  0.494→0.640). See [[per-cwe-memory-signal-exists]].

- [~] **exp-21 — RETRACTED (wrong metric).** Reported pair-ranking (vuln vs its
  own patch) not `tokens_code_auc`, and missed exp-10 → falsely concluded "memory
  unlearnable". The only genuinely-new piece vs exp-10 is the cross-CWE
  **transfer** matrix (train CWE-X → test CWE-Y). TODO: recompute that transfer
  matrix on `tokens_code_auc` from the saved per-CWE logits
  (`runs/percwe_*/logits_percwe.npz`) — no re-extraction needed.

- [ ] **vLLM as the default extractor on the cluster.** vLLM 0.22.1 is installed at
  `$WORK/.python_deps_vllm` but is *off* the default `PYTHONPATH` (env.sh uses
  `.python_deps5`), so the extractor silently falls back / errors on
  `--backend vllm`. Wire `.python_deps_vllm` into `env.sh` (or the job) and
  verify the uncommitted `extract_vllm` path runs end-to-end on the cluster —
  it has never been validated on a full the cluster extraction.

- [x] **Pooled-probe token-level FN/FP analysis (subtractive subset).** What does
  the pooled vulnerability probe consistently detect vs miss across Qwen and
  Gemma, and what spuriously fires. → `20-fn-fp-token-analysis/`.
</content>
</invoke>
