[ai-generated]

# exp-20 — Token-level FN/FP error analysis of the pooled vulnerability probe

## Aim
Characterize, at the **token level** on the **SVEN-subtractive** subset, what the
pooled span-max vulnerability probe consistently detects vs misses **across Qwen
and Gemma**, and what it spuriously fires on. (Example-level performance is
already poor; the question is which *tokens/spans* drive FN and FP, and whether
the pattern is shared across model families.)

## Inputs
- **Logits:** exp-16 per-token logit dumps (`16-token-logit-dump/results/
  logitdump_<MODEL>/logits_layer<NN>.npz`) at each model's operating layer
  (`OP_LAYER`, from exp-16 `relabel_recompute.py`): Qwen-32B L25, Qwen-7B L16,
  gemma-1b L25, gemma-4b L7, gemma-12b-it L15, gemma-27b-it L19, gemma-12b-pt L13.
- **Subset:** exp-19 `subtractive_membership.json` (478 vuln+safe pairs; 97 vuln
  in the held-out group-clean test split).
- **Labels:** honest ADR-0004 label recomputed in-harness — token is positive iff
  its char span overlaps a difflib delete/replace span in `before` **and** is
  tree-sitter live-code (`is_code`). Code-only regime.
- **Operating point:** per-model F1-max threshold on subtractive **train** code
  tokens. `TODO(adhoc-decision)`: F1-max-on-train chosen as each model's best
  operating point; example-detection results are threshold-robust (bimodal).

## Outputs (this dir)
- `extract_fn_fp.py` → `analysis.json` (thresholds, per-token confusion,
  detection matrix), `fn_corpus.json` (97 test vuln, marked span + detect bits),
  `fp_corpus.json` (6119 FP spans), `fp_sample.json` (129 curated for agents).
- `fp_buckets.py` → `fp_buckets.json` (population lexical FP breakdown).
- `slices/` per-agent inputs; `categorization.json` merged taxonomy + provenance.
- `RESULTS.md`.

## Result format
- Cross-model detection histogram + consistently-detected / -missed / -differential
  (CWE breakdown).
- FN taxonomy (6 categories, 35 hard-missed) and FP taxonomy (5 categories,
  sample + population counts), compiled from 6 sub-agents.

## Interpretation hints
- If detect/miss splits by **CWE class** (injection caught, memory-safety missed)
  uniformly across families → the probe is a lexical **sink-presence** detector,
  not a vulnerability detector. If families differ on *which* vulns → genuine
  capability divergence. (Observed: the former; families differ only in
  trigger-happiness/calibration, not capability.)
