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
- a runtime bundle's `tokenizer.json` is the model's own tokenizer. Nothing checks this: a valid but mismatched tokenizer loads and produces wrong embeddings;
- the bundle checks made at load time (required files, `build-info.json` metadata, GGUF embedding width; see `docs/bundle-format.md`) fail explicitly. Do not weaken them, and do not describe a check the loader does not make;
- correctness fixtures are independent reference data, not output captured from the implementation being tested.

If one of these contracts intentionally changes, update implementation, tests, examples, workflows, and documentation together.

## Build Environment

Cargo commands that compile the crate (`build`, `check`, `clippy`, `test`, `run`, `doc`) run `build.rs`, which needs `STATIC_LLAMA_DIR` pointing at a verified, extracted static llama.cpp release. The archives are ARM64 Linux objects, so run those commands on ARM64 Linux or in the `linux/arm64` container. Commands that do not compile the crate, such as `cargo fmt`, `cargo metadata`, and `cargo tree`, need neither and work on any host.

`docs/development.md` is the authority for this setup: § "Static llama.cpp artifacts" for download and verification, § "Non-aarch64 hosts" for the container. Use that container workflow rather than inventing a second build path.

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

The baseline is the command list in `docs/development.md` § "Checks before pushing". It mirrors the CI `Test`, `Lint`, and `Python script tests` jobs: formatting, Clippy, the Rust test targets, the example build, the model-backed integration tests, and the Python script tests. For code changes, run as much of it as the environment supports, and do not present a subset as the full baseline.

Every cargo command in the baseline except `cargo fmt --check` compiles the crate, so it needs the setup under "Build Environment". The model-backed integration tests also need a GGUF bundle.

Never report a command as passing unless it actually ran successfully. A model-backed test that skipped because no bundle was available did not pass. If validation cannot run because of architecture, artifacts, model weights, Docker, or another environmental limitation, state that explicitly.

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

## Keeping This Guidance Current

The `AGENTS.md` files link to authoritative documents such as `docs/README.md`, `docs/development.md`, `docs/testing.md`, and `scripts/README.md` instead of copying their file or command lists. Keep it that way: add a link, not another inventory.

When a change renames, moves, or removes a path, command, or constant that an `AGENTS.md` names, update that `AGENTS.md` in the same change. `git grep -n '<old name>'` finds the references.

## Completion Report

For non-trivial work, report:

- what changed;
- why;
- important design decisions;
- checks actually executed;
- checks that could not be executed and why;
- remaining risks or follow-up work.

Keep the report factual.
