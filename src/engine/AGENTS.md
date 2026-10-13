# Engine Agent Guidance

These instructions apply to the embedding engine and backend implementation.

Also follow `/AGENTS.md` and `/src/AGENTS.md`.

## Engine Contract

The current inference flow is conceptually:

```text
EmbeddingInput
    ↓
query/document prefix
    ↓
tokenization
    ↓
backend inference (llama_encode, one sequence at a time)
    ↓
mean-pooled, projected raw embedding (768)
    ↓
finite-value check
    ↓
Matryoshka truncation
    ↓
optional L2 normalization
```

Preserve separation between shared retrieval semantics and backend-specific inference.

## Backend Boundary

`EmbeddingBackend` is an internal abstraction.

`LlamaBackend` is currently the production implementation.

The shared engine owns:

- retrieval-prefix behavior;
- tokenization;
- maximum-length enforcement;
- post-processing;
- output truncation;
- optional normalization.

A backend should return the raw pooled representation expected by the shared engine.

Do not move shared semantics into `LlamaBackend` unless the architecture intentionally changes.

## Retrieval Semantics

Callers provide typed inputs:

```rust
EmbeddingInput::query(...)
EmbeddingInput::document(...)
```

Do not require callers to manually add retrieval prefixes. Prefixes come from validated bundle metadata and are applied internally.

## Current Model Contract

The current model/runtime contract is described in `docs/architecture.md` § "Model" and defined by the public constants in `src/engine/mod.rs`:

- `RAW_EMBEDDING_DIMENSION` (768);
- `EMBEDDING_DIMENSION` (512, the default returned dimension);
- `MAX_LENGTH` (8192);
- `QUERY_PREFIX` / `DOCUMENT_PREFIX` (EmbeddingGemma 2's `query` and `document` prompts);
- pooling: mean over every token, then the model's 512→768 projection. The backend returns the projected `n_embd_out` vector, never the 512-wide hidden state.

At load time the engine reads the raw dimension, maximum length, and prefixes from the bundle's `build-info.json`; the constants do not override them, except that `max_length` must lie in `1..=MAX_LENGTH`. `EMBEDDING_DIMENSION` is the `EngineConfig::default()` output size. The same values are also repeated elsewhere, such as `BENCHMARK_MAX_LENGTH` in `src/benchmarking.rs`, so changing a constant alone is an incomplete change.

If intentionally changing one, inspect and update all related runtime validation, constants, bundle metadata, tests, fixtures, benchmark tools, CI/release bundle assembly, and documentation.

Never update only one copy.

## GGUF Bundle

A runtime bundle contains at least:

```text
gguf_bundle/
├── model.gguf
├── tokenizer.json
└── build-info.json
```

Use `bundle.rs` and `scripts/assemble_gguf_bundle.py`, which every workflow uses to build its bundle, as the authoritative metadata contract. `docs/bundle-format.md` documents it and must stay in sync.

Loading fails explicitly only for what it checks: missing files, unreadable or unsupported `build-info.json` metadata (including a `max_length` outside `1..=8192`), a tokenizer or GGUF that fails to load, a GGUF pooled width (`n_embd_out`) that differs from `raw_embedding_dimension`, a GGUF that does not declare mean pooling, a tokenizer vocabulary size (added tokens included) that differs from the GGUF's, and a tokenizer that tokenizes `TOKENIZER_CHECK_TEXT` differently from the GGUF's vocabulary. `docs/architecture.md` § "Load path" lists the steps and `docs/bundle-format.md` the per-field rules. Keep these checks explicit. A new check needs a test and a matching update to both documents.

The two tokenizer checks are sanity checks, not an identity check. They reject a tokenizer from a model with another vocabulary size, or one that lost its special tokens, but a tokenizer that agrees with the GGUF on the check text and differs elsewhere loads without error and produces wrong embeddings; only the bundle-gated tests catch that, and only for the bundle and inputs they run (`docs/testing.md`). So:

- the tokenizer must be the one pinned in `scripts/assemble_gguf_bundle.py`: `unsloth/embeddinggemma-2` at its pinned revision, Unsloth's safetensors copy of the model (the GGUF repository has no `tokenizer.json`). Its agreement with the GGUF's vocabulary is shown by the load-time check and by the llama.cpp reference test's token ids, not by provenance;
- do not use `assets/tokenizer.json` as a runtime replacement. It is a stale 30k-vocab placeholder, and loading rejects it;
- do not describe either check as validating tokenizer/model identity.

## llama.cpp FFI

Treat `src/engine/llama/` as a safety-sensitive boundary.

When changing it:

- use the bindings associated with the repository-pinned static artifact;
- verify pointer ownership and lifetime assumptions;
- validate llama.cpp return values;
- preserve input/output ordering;
- verify sequence and batching semantics;
- verify pooling behavior;
- verify expected embedding width (the projected `n_embd_out`, not `n_embd`);
- keep `llama_encode` for this non-causal encoder, and `n_ubatch` equal to the context length: the encoder cannot split a sequence across micro-batches;
- check memory and latency at 8192 tokens when changing context, batch or attention parameters (`docs/architecture.md` § "Long inputs and memory");
- do not assume another upstream llama.cpp version behaves identically.

Do not retain pointers beyond the lifetime guaranteed by the underlying library.

Any change affecting inference execution should be covered by model-backed testing when the environment permits.

## Performance Work

Do not optimize by changing semantics.

For performance-sensitive changes:

1. establish a baseline;
2. measure the affected scenario;
3. preserve correctness/parity;
4. report benchmark environment and configuration.
