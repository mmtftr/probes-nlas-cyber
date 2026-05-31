[ai-generated]

# Activation extraction & dataset handling

Lessons from extracting per-token hidden states across many models on SVEN.

## Store float32, not float16
- Gemma-3 has "massive activations" (>65504) in mid layers that **saturate to
  inf in float16 → NaN** in `roc_auc_score` and training. Store activations as
  **float32**. (551 GB for all 62 Gemma layers — fine on scratch.)
- Even so, guard every per-layer train with `np.isfinite(X).all()` and a
  per-layer try/except so one bad layer can't abort a sweep.

## Streaming all layers
- All layers × all tokens won't fit in RAM (~275–551 GB). Stream each layer to
  its own `np.lib.format.open_memmap(..., mode="w+", dtype=float32)` and write
  one row-slice per example. `hidden_states` has `n_layers+1` entries
  (embeddings + one per block) — index `hs[li+1]` for block `li`.
- Write a `DONE_EXTRACT` marker + `meta.json` (model, n_layers, hidden, n_tokens,
  pos_tokens) so extraction is idempotent and downstream readers self-describe.
- Activations are **split-independent**: cache them once, then re-run probe
  training under many splits / losses / α without re-extracting. This is what
  makes the variance ([[span-max-loss-tuning]]) and loss sweeps cheap (~10 min).

## Tokenizer offsets are the hard requirement
- Labelling needs char→token mapping, i.e. `return_offsets_mapping=True`. Models
  whose tokenizer can't provide it are unusable without risky reconstruction:
  - **OpenCoder** (INFLMTokenizer, SentencePiece, no fast variant) → no
    `offset_mapping`. **Dropped.**
  - **Devstral / Mistral-Small-3.2** (Tekken): the HF-side tokenizer expands to
    151000-vocab vs the model's 131072 embeddings → token-id OOB → CUDA
    device-side assert. `MistralCommonTokenizer` absent in transformers 5.9.0;
    mistral-common gives no offsets. **Dropped.** See `decisions/0001-*`.
- Robust loaders (`src/data/extract_token_activations.py`): `AutoTokenizer(...,
  trust_remote_code=True)`, fall back to `AutoProcessor.tokenizer` for VLMs but
  **re-raise the original error** if that fails (don't mask the real cause);
  `AutoModelForCausalLM` → fall back to `AutoModelForImageTextToText`.

## Split design (group-clean)
- SVEN ships vuln/fix **pairs**; a pair must never straddle train/test or the
  probe sees the answer. Shuffle at the **pair-group** level (`pair_group_key`:
  repo → file::func → code-hash), hold out 20% of *groups*, seed-controlled.
- Vary only the OUTER held-out split for variance; keep the internal 90/10
  epoch-selection split fixed (it's part of the recipe, not a dataset split).
