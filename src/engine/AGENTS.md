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

## Embedding Space

`EmbeddingEngine::embedding_space()` names the vector space of an engine's output:
the sha256 of the `model.gguf` it loaded, the prefixes it adds, and its `EngineConfig`
(`src/engine/space.rs`, `docs/architecture.md` § "Embedding space"). Callers store its string
form beside their indexes and refuse to mix vectors whose spaces differ. So:

- build it only from what the engine uses. Do not add bundle claims such as `target_id`,
  `quant`, `provenance.json` or GGUF `general.*` metadata: a bundle edited by hand must not
  be able to report another model's space;
- add a setting to it if, and only if, the setting changes vectors by more than rounding;
- bump `STRING_FORM_VERSION` when the same model file and settings would give different
  vectors (pooling, projection, truncation, normalization, prefix handling, or a llama.cpp
  release that changes how the model tokenizes or pools); otherwise stored tags match
  vectors they no longer describe. `docs/development.md` § "Static llama.cpp artifacts"
  makes that decision part of every release bump. A change to the string form itself
  needs a bump too, and makes every stored tag stop matching, so that callers rebuild;
- keep the hash lazy. Hashing at load would add to every load (measured in
  `docs/architecture.md` § "Embedding space").

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

Loading fails explicitly only for what it checks: missing files, unreadable or unsupported `build-info.json` metadata (including a `max_length` outside `1..=8192`), a GGUF that fails to load, a GGUF pooled width (`n_embd_out`) that differs from `raw_embedding_dimension`, a GGUF that does not declare mean pooling, a GGUF whose tokenization of `SPECIAL_TOKENS_CHECK_TEXT` does not start with its BOS and end with its EOS, and a `model.gguf` path that no longer names the file the engine opened before llama.cpp loaded it. A `tokenizer.json` in the directory is not read. `docs/architecture.md` § "Load path" lists the steps and `docs/bundle-format.md` the per-field rules. Keep these checks explicit. A new check needs a test and a matching update to both documents.

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
- call `llama_encode` and anything else that touches the context only under the context's mutex. Tokenizing takes no lock because `llama_tokenize` only reads shared state; the argument and its llama.cpp references are at `tokenize_with_vocab` in `src/engine/llama/mod.rs` (`docs/architecture.md` § "Concurrency"). Recheck it when the llama.cpp commit changes, and do not call anything else on the model outside the mutex without making the same argument;
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
