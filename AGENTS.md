# AGENTS.md

This file defines repository-wide rules for coding agents working on LTEmbed.

More specific guidance exists in nested `AGENTS.md` files. When working inside one of those directories, read and follow the nearest applicable instructions in addition to this file.

## Project

LTEmbed is a Rust text-embedding library centered on `EmbeddingEngine`.

The current `main` branch uses:

- llama.cpp / GGUF for inference;
- statically linked llama.cpp/ggml artifacts;
- Hugging Face `tokenizer.json`;
- typed query/document retrieval inputs;
- last-token pooling;
- Matryoshka output truncation;
- optional L2 normalization.

ONNX Runtime and matrixmultiply implementations belong to earlier architecture phases or other branches and are not the current `main` runtime.

## Source of Truth

When repository sources disagree, use this priority:

1. current implementation under `src/`;
2. `Cargo.toml`, `build.rs`, and `rust-toolchain.toml`;
3. `.github/` CI and release workflows;
4. tests and executable examples;
5. current README/documentation;
6. historical plans, experiments, and migration notes.

Never change working current code merely to make it agree with obsolete documentation.

## Global Invariants

Do not casually change these semantics:

- callers use `EmbeddingInput::query(...)` or `EmbeddingInput::document(...)`;
- retrieval prefixes are applied internally;
- the current backend is llama.cpp/GGUF;
- runtime bundles use the model's matching tokenizer;
- malformed or incompatible bundles fail explicitly;
- correctness fixtures are independent reference data, not output captured from the implementation being tested.

If one of these contracts intentionally changes, update implementation, tests, examples, workflows, and documentation together.

## Build Environment

The crate requires the static llama.cpp artifacts consumed by `build.rs`.

`STATIC_LLAMA_DIR` must point to a verified extracted artifact bundle.

The primary native target is ARM64 Linux. On unsupported development hosts, use the repository's documented container workflow rather than inventing a second build path.

Use the Rust toolchain pinned in `rust-toolchain.toml`.

## Scope Discipline

Keep changes focused on the requested task.

Do not opportunistically:

- reintroduce old inference backends;
- expand the public API;
- update unrelated dependencies;
- regenerate fixtures;
- rewrite large modules;
- weaken validation;
- modify historical benchmark data.

Raise unrelated problems separately.

## Baseline Validation

For Rust changes, run as much of the following as the environment supports:

```bash
cargo fmt --all --check
cargo clippy --all-targets -- -D warnings
cargo test --lib
cargo test --test tokenizer_reload_tests
cargo test --test benchmarking_support_tests --bin benchmark_ltembed
cargo check --example api_usage
```

Some commands require ARM64 Linux and valid static llama.cpp artifacts.

Never report a command as passing unless it actually ran successfully. If validation cannot run because of architecture, artifacts, model weights, Docker, or another environmental limitation, state that explicitly.

## Public Examples

Keep `examples/api_usage.rs` synchronized with the actual public API.

Documentation examples should preferably follow that executable example instead of creating parallel pseudo-APIs.

## Nested Guidance

Before working in these areas, read their local instructions:

- `src/AGENTS.md` — Rust implementation and public API rules
- `src/engine/AGENTS.md` — inference pipeline, GGUF bundle, and llama.cpp backend
- `tests/AGENTS.md` — test tiers and fixture contracts
- `scripts/AGENTS.md` — benchmark/reference tooling
- `docs/AGENTS.md` — current documentation rules
- `docs/history/AGENTS.md` — historical-document preservation
- `.github/AGENTS.md` — CI, artifact pinning, and releases

## Completion Report

For non-trivial work, report:

- what changed;
- why;
- important design decisions;
- checks actually executed;
- checks that could not be executed and why;
- remaining risks or follow-up work.

Keep the report factual.
