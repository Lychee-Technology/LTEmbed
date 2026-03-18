# Remove Lambda Packaging And Add Rust API Example Design

**Date:** 2026-03-17
**Issue Theme:** remove AWS Lambda packaging from the repository and replace it with a first-class Rust API example for LTEmbed

## Goal

Make LTEmbed unambiguously a Rust library crate by removing the in-repo AWS Lambda binary and its deployment tooling, then add a runnable example that demonstrates the library's public Rust API directly.

## Context

LTEmbed's public identity is already that of a transport-agnostic embedding library. However, the current crate manifest and repository layout still include a Lambda deployment binary in [`src/main.rs`](../../../src/main.rs), Lambda-specific dependencies in [`Cargo.toml`](../../../Cargo.toml), and packaging helpers in [`build.sh`](../../../build.sh) and [`Dockerfile`](../../../Dockerfile).

This creates two problems:

- the crate appears to depend on Lambda even though the library API does not
- new users looking for basic Rust usage examples are shown deployment concerns instead of the core library API

The requested change is therefore not just a cleanup. It is a correction of the repository boundary so the code, manifest, and documentation all match the intended product: a reusable Rust library.

## Scope

This work includes:

- removing the Lambda binary target and Lambda-only dependencies from the crate manifest
- deleting repository files whose only purpose is Lambda packaging or Lambda HTTP handling
- adding a runnable example under `examples/` that demonstrates LTEmbed's Rust API
- updating documentation so examples and project structure no longer mention Lambda as an in-repo concern
- adding or updating tests/checks needed to keep the new example and crate layout maintainable

This work does not include:

- introducing a replacement HTTP server, CLI tool, or deployment target
- reorganizing the repository into a Cargo workspace
- changing LTEmbed's public API surface beyond what is required for the example to consume it
- changing model assets or benchmark logic unrelated to the removed Lambda path

## Options Considered

### Option 1: Delete Lambda artifacts and add a Rust API example

Pros:

- matches the requested end state exactly
- removes misleading dependencies from the main crate
- makes the first example of LTEmbed usage directly about the library API

Cons:

- deletes a deployment example that may have been useful historically

### Option 2: Keep Lambda code behind a Cargo feature

Pros:

- preserves the old deployment path
- smaller code diff

Cons:

- the repository still advertises Lambda as a first-class concern
- feature-gated deployment code still complicates the manifest and maintenance story
- does not satisfy the user's direction to remove Lambda

### Option 3: Split Lambda into a separate crate inside a workspace

Pros:

- clean separation of concerns
- preserves both library and deployment example

Cons:

- over-scoped for the requested outcome
- adds workspace complexity with no current need

### Recommendation

Choose option 1.

This is the smallest change that makes the repository consistent with LTEmbed's actual purpose and gives users the example they now need: direct Rust API usage.

## Design

### Crate Boundary

The crate should expose only the library target. The `[[bin]] bootstrap` entry will be removed from [`Cargo.toml`](../../../Cargo.toml), along with Lambda-only dependencies such as `lambda_http`, `tokio`, `tracing`, and `tracing-subscriber`.

After this change:

- `cargo build`, `cargo test`, and `cargo doc` operate on the library and its normal test/example targets
- `Cargo.lock` no longer includes the Lambda dependency chain after it is refreshed
- repository readers no longer infer that Lambda is part of the supported public shape of the crate

### Example Program

Add a new example program under `examples/` with a name that reflects intent, for example `api_usage.rs`.

The example should demonstrate the minimum successful LTEmbed flow:

1. read `assets/config.json`
2. construct `ZeroVecEngine` with `assets/model.safetensors`, the config contents, `assets/tokenizer.json`, and `MeanPooling`
3. call `embed_batch` on one or two representative strings
4. print a compact summary of the result, such as embedding count, vector length, and the first few coordinates

The example should optimize for clarity, not completeness. It is documentation users can run, not an application framework.

### Example Behavior With Missing Assets

The example will depend on local assets. That is acceptable because this repository already keeps example assets under `assets/`.

When required files are missing, the example should fail with a clear error message that tells the caller which asset is required. It should not silently skip work or panic with an opaque backtrace if a straightforward error message can be returned.

### Documentation Updates

Documentation should be aligned to the new repository boundary:

- [`docs/design.md`](../../../docs/design.md) should replace the Lambda deployment section with a Rust API example section or a short note pointing readers to `cargo run --example ...`
- [`docs/integ-test.md`](../../../docs/integ-test.md) should stop naming `src/main.rs` as a deployment-specific out-of-scope concern
- any project structure listings should remove `src/main.rs`, `build.sh`, and Lambda packaging references when they no longer exist

The key message should be consistent everywhere: LTEmbed is a library crate, and the repository's first example of usage is a direct Rust example.

### File Removals

The expected removals are:

- [`src/main.rs`](../../../src/main.rs)
- [`build.sh`](../../../build.sh)
- Lambda-specific content in [`Dockerfile`](../../../Dockerfile), if the file has no remaining purpose after the Lambda path is removed

If `Dockerfile` remains useful for non-Lambda development, it should be rewritten to reflect that narrower purpose. Otherwise it should be deleted.

## Testing Strategy

The implementation should follow TDD.

Required verification targets:

- a regression test or compile-level check proving the new example is valid
- `cargo test`
- `cargo fmt --all --check`
- `cargo clippy --all-targets -- -D warnings`

For the example itself, the preferred lightweight verification is `cargo check --example <name>` or `cargo run --example <name>` when assets are available locally. If the example is exercised in tests, those tests should avoid brittle stdout matching and focus on successful construction and embedding behavior.

## Deliverables

- Lambda binary target and Lambda-only dependencies removed from the crate
- Lambda-specific repository files removed or rewritten
- one runnable Rust API example added under `examples/`
- documentation updated to reflect the library-only positioning
- GitHub issues created to track implementation

## Risks And Mitigations

**Risk:** documentation still references Lambda after the code is removed.  
**Mitigation:** update every file that currently names `src/main.rs`, Lambda packaging, or Lambda deployment as part of the repository structure.

**Risk:** the new example is too dependent on local assets to be useful.  
**Mitigation:** keep the example minimal, document required files clearly, and print concise success output so users can confirm it worked quickly.

**Risk:** removing the binary target leaves stale lockfile entries or CI assumptions behind.  
**Mitigation:** refresh `Cargo.lock` and run the full repository verification commands after the implementation lands.

## Proposed GitHub Issues

### Issue 1: Remove Lambda binary target and dependencies from LTEmbed

Scope:

- remove `[[bin]] bootstrap`
- remove Lambda-only dependencies from `Cargo.toml`
- delete `src/main.rs`
- refresh `Cargo.lock`

Acceptance criteria:

- the crate builds and tests without any Lambda dependency chain
- no repository entry point depends on `lambda_http`, `tokio`, or `tracing` for deployment

### Issue 2: Remove Lambda-specific packaging artifacts and stale documentation

Scope:

- delete or rewrite `build.sh` and `Dockerfile`
- update design and testing docs to remove Lambda-specific language
- update project structure listings accordingly

Acceptance criteria:

- repository docs consistently describe LTEmbed as a library crate
- no remaining top-level packaging file implies Lambda deployment support unless it serves another explicit purpose

### Issue 3: Add a runnable Rust API example for LTEmbed

Scope:

- add `examples/api_usage.rs` or equivalent
- demonstrate `ZeroVecEngine::new` and batch embedding
- document how to run the example and what assets it expects

Acceptance criteria:

- `cargo run --example <name>` succeeds when required assets are present
- the example gives a clear, minimal demonstration of LTEmbed's Rust API with no HTTP or Lambda concepts
