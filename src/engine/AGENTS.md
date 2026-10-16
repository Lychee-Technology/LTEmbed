# Engine Agent Guidance

These instructions apply to the embedding engine and backend implementation.

Also follow `/AGENTS.md` and `/src/AGENTS.md`.

## Engine Contract

The current inference flow is conceptually:

```text
EmbeddingInput
    ↓
size check (an input over the byte bound is rejected untokenized)
    ↓
query/document prefix
    ↓
tokenization (the GGUF's own vocabulary, through the backend)
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
- maximum-length enforcement, without truncation: first on the input's size in bytes against the bound the backend derives from its vocabulary (`EmbeddingBackend::max_input_bytes`), then on the backend's token ids;
- post-processing;
- output truncation;
- optional normalization.

A backend tokenizes with its model file's own vocabulary (`EmbeddingBackend::tokenize`), and sizes the byte bound from it (`EmbeddingBackend::max_input_bytes`), because the vocabulary lives in the GGUF, and returns the raw pooled representation expected by the shared engine.

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

At load time the engine reads the raw dimension, maximum length, and prefixes from the bundle's `build-info.json`; the constants do not override them, except that `max_length` must lie in `1..=MAX_LENGTH`. `EMBEDDING_DIMENSION` is the `EngineConfig::default()` output size. The same values are also repeated elsewhere, such as in the `build-info.json` that `scripts/assemble_gguf_bundle.py` writes, so changing a constant alone is an incomplete change.

If intentionally changing one, inspect and update all related runtime validation, constants, bundle metadata, tests, fixtures, benchmark tools, CI/release bundle assembly, and documentation.

Never update only one copy.

## GGUF Bundle

A runtime bundle contains at least:

```text
gguf_bundle/
├── model.gguf
└── build-info.json
```

Use `bundle.rs` and `scripts/assemble_gguf_bundle.py`, which every workflow uses to build its bundle, as the authoritative metadata contract. `docs/bundle-format.md` documents it and must stay in sync.

Loading fails explicitly only for what it checks: missing files, unreadable or unsupported `build-info.json` metadata (including a `max_length` outside `1..=8192`), a GGUF that fails to load, a GGUF pooled width (`n_embd_out`) that differs from `raw_embedding_dimension`, a GGUF that does not declare mean pooling, and a GGUF whose tokenization of `SPECIAL_TOKENS_CHECK_TEXT` does not start with its BOS and end with its EOS. A `tokenizer.json` in the directory is not read. `docs/architecture.md` § "Load path" lists the steps and `docs/bundle-format.md` the per-field rules. Keep these checks explicit. A new check needs a test and a matching update to both documents.

The BOS/EOS check is a sanity check, not an identity check. It rejects a GGUF whose tokenization lost the special tokens the model wraps every input in, but a GGUF whose vocabulary splits text differently from the model's own tokenizer loads without error and produces wrong embeddings; only the bundle-gated tests catch that, and only for the GGUF and inputs they run (`docs/testing.md`). So:

- the GGUF must be one pinned in `scripts/assemble_gguf_bundle.py`. Its vocabulary's agreement with the model's own tokenizer is shown by the tokenizer parity tests (`src/engine/tokenizer_parity.rs`, `docs/testing.md` § "Tokenizer parity"): against the pinned `tokenizer.json` of `unsloth/embeddinggemma-2` on a fixed corpus of about 25,000 texts, and against the token-id and llama.cpp reference fixtures. It is not shown by provenance;
- do not add a `tokenizer.json` or a second tokenizer to the runtime path;
- do not describe the check as validating tokenizer/model identity.

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
