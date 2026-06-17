# Per-CWE matched-patch (each vulnerable function vs its OWN fix)

Eval = token-level ROC-AUC on live code. Positives = the CWE's annotated
vulnerable tokens; negatives = the same before-functions' OTHER code tokens
plus their patched counterparts' tokens (same file/function patched pool;
usually the paired fix, ordinal pairing within the 7 duplicate-key groups).
So language/project/style/template are held constant; no unrelated function
contributes a negative. Language null is
exactly 0.5 for single-language CWEs (all memory = C/C++); CWE-022 (mixed
Python/C) keeps a residual, marked `†`. `*` = n<10 held-out pairs
(untrusted). Probe trained CWE-X-vuln vs all-clean, then evaluated
matched-patch; lexical baselines on the identical token axis.

## Qwen2.5-Coder-32B (layer 25)

| CWE | type | family | n | lang-null | probe | char-ngram | token-unigram |
|---|---|---|---:|---:|---|---|---|
| CWE-022 | path traversal | inj | 15 | 0.369† | 0.771 [0.69, 0.83] | 0.753 [0.60, 0.88] | 0.653 [0.59, 0.72] |
| CWE-078 | command injection | inj | 19 | 0.500 | 0.833 [0.72, 0.94] | 0.872 [0.80, 0.96] | 0.692 [0.63, 0.80] |
| CWE-079 | XSS | inj | 10 | 0.515 | 0.532 [0.42, 0.74] | 0.578 [0.45, 0.83] | 0.522 [0.47, 0.62] |
| CWE-089 | SQL injection | inj | 44 | 0.500 | 0.933 [0.91, 0.94] | 0.975 [0.96, 0.99] | 0.760 [0.73, 0.78] |
| CWE-125 | out-of-bounds read | mem | 19 | 0.500 | 0.633 [0.57, 0.70] | 0.545 [0.43, 0.65] | 0.591 [0.56, 0.62] |
| CWE-190 | integer overflow | mem | 4 * | 0.500 | 0.724 [0.68, 0.76] | 0.805 [0.74, 0.90] | 0.684 [0.62, 0.70] |
| CWE-416 | use-after-free | mem | 14 | 0.500 | 0.610 [0.51, 0.71] | 0.542 [0.44, 0.62] | 0.538 [0.50, 0.58] |
| CWE-476 | NULL dereference | mem | 16 | 0.500 | 0.544 [0.48, 0.63] | 0.506 [0.42, 0.60] | 0.557 [0.53, 0.58] |
| CWE-787 | out-of-bounds write | mem | 5 * | 0.500 | 0.509 [0.47, 0.60] | 0.672 [0.38, 0.82] | 0.503 [0.46, 0.59] |

`*` n<10 (untrusted). `†` residual token-level language structure (CWE-022 is the only mixed Python/C class).

## Gemma-3-1B (layer 25)

| CWE | type | family | n | lang-null | probe | char-ngram | token-unigram |
|---|---|---|---:|---:|---|---|---|
| CWE-022 | path traversal | inj | 15 | 0.358† | 0.718 [0.63, 0.79] | 0.765 [0.60, 0.89] | 0.647 [0.59, 0.70] |
| CWE-078 | command injection | inj | 19 | 0.500 | 0.771 [0.66, 0.87] | 0.876 [0.79, 0.96] | 0.659 [0.61, 0.74] |
| CWE-079 | XSS | inj | 10 | 0.510 | 0.611 [0.54, 0.75] | 0.578 [0.43, 0.83] | 0.552 [0.52, 0.61] |
| CWE-089 | SQL injection | inj | 44 | 0.500 | 0.886 [0.86, 0.90] | 0.975 [0.96, 0.99] | 0.737 [0.71, 0.76] |
| CWE-125 | out-of-bounds read | mem | 19 | 0.500 | 0.657 [0.59, 0.73] | 0.596 [0.48, 0.71] | 0.584 [0.54, 0.62] |
| CWE-190 | integer overflow | mem | 4 * | 0.500 | 0.713 [0.67, 0.81] | 0.829 [0.76, 0.92] | 0.693 [0.64, 0.71] |
| CWE-416 | use-after-free | mem | 14 | 0.500 | 0.603 [0.52, 0.68] | 0.506 [0.43, 0.60] | 0.523 [0.48, 0.57] |
| CWE-476 | NULL dereference | mem | 16 | 0.500 | 0.507 [0.45, 0.55] | 0.434 [0.35, 0.53] | 0.518 [0.49, 0.54] |
| CWE-787 | out-of-bounds write | mem | 5 * | 0.500 | 0.537 [0.44, 0.60] | 0.791 [0.50, 0.91] | 0.532 [0.49, 0.55] |

`*` n<10 (untrusted). `†` residual token-level language structure (CWE-022 is the only mixed Python/C class).
