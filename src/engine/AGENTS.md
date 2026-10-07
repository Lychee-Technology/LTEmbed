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
backend inference
    ↓
last-token pooled raw embedding
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
- `QUERY_PREFIX` / `DOCUMENT_PREFIX`;
- pooling: last token.

At load time the engine reads the raw dimension, maximum length, and prefixes from the bundle's `build-info.json`; the constants do not override them. `EMBEDDING_DIMENSION` is the `EngineConfig::default()` output size. The same values are also repeated elsewhere, such as `BENCHMARK_MAX_LENGTH` in `src/benchmarking.rs`, so changing a constant alone is an incomplete change.

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

Use `bundle.rs` and current CI/release bundle generation as the authoritative metadata contract. `docs/bundle-format.md` documents it and must stay in sync.

Loading fails explicitly only for what it checks: missing files, unreadable or unsupported `build-info.json` metadata, a tokenizer or GGUF that fails to load, and a GGUF embedding width that differs from `raw_embedding_dimension`. `docs/architecture.md` § "Load path" lists the steps and `docs/bundle-format.md` the per-field rules. Keep these checks explicit. A new check needs a test and a matching update to both documents.

Loading does **not** check that `tokenizer.json` belongs to the model. A valid but mismatched tokenizer loads without error and produces wrong embeddings; only the bundle-gated tests catch it, and only for the bundle they run against (`docs/testing.md`). So:

- the tokenizer must come from the same Hugging Face repository and revision as the GGUF;
- do not use `assets/tokenizer.json` as a runtime replacement because it parses. It is a stale 30k-vocab placeholder;
- do not describe tokenizer/model matching as validated unless a load-time check is added.

## llama.cpp FFI

Treat `src/engine/llama/` as a safety-sensitive boundary.

When changing it:

- use the bindings associated with the repository-pinned static artifact;
- verify pointer ownership and lifetime assumptions;
- validate llama.cpp return values;
- preserve input/output ordering;
- verify sequence and batching semantics;
- verify pooling behavior;
- verify expected embedding width;
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
