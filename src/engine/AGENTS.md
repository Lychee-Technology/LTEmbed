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

The current model/runtime contract includes:

- raw embedding dimension: 768;
- default returned embedding dimension: 512;
- maximum tokenizer length: 8192;
- pooling: last token.

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

Use `bundle.rs` and current CI/release bundle generation as the authoritative metadata contract.

The tokenizer must match the model.

Do not use `assets/tokenizer.json` as a runtime replacement merely because it parses successfully. It is currently treated as a stale placeholder.

Bundle incompatibilities should fail explicitly during model loading.

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
