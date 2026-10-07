# Source Agent Guidance

These instructions apply to Rust source code under `src/`.

Repository-wide rules in `/AGENTS.md` also apply.

## Rust Changes

Follow existing module boundaries and the conventions in `docs/rust-coding-std.md`.

Prefer:

- small public APIs;
- typed configuration;
- explicit error propagation;
- clear ownership and lifetimes;
- backend-independent behavior in shared layers;
- measurement-backed performance changes.

Avoid unnecessary allocation in inference-sensitive paths.

Run `rustfmt` and keep Clippy clean with `-D warnings`.

## Public API

Treat changes to the following as potentially breaking:

- `EmbeddingEngine`;
- `EmbeddingInput`;
- `EmbeddingInputKind`;
- `EngineConfig`;
- exported constants;
- externally observable error behavior.

Before changing public API:

1. inspect all repository call sites;
2. inspect `examples/api_usage.rs`;
3. inspect integration tests;
4. prefer an internal solution when possible;
5. update examples and current documentation together.

Do not expose backend internals solely to make an implementation easier.

## Error Handling

Use the typed hierarchy in `src/error.rs`.

Runtime validation should normally return an appropriate `LTEmbedError`.

Do not silently accept malformed configuration, silently alter requested semantics, convert useful typed failures into opaque strings, or panic on normal invalid runtime input.

Build-time prerequisite checks may fail hard where `build.rs` already establishes that convention.

## Dependencies

Before adding a dependency:

- check whether the standard library or an existing dependency is sufficient;
- consider ARM64 Linux compatibility;
- consider native build requirements;
- consider binary/package size;
- disable unnecessary default features where appropriate.

Do not update unrelated dependencies as part of another task.

Changes involving `tokenizers`, llama.cpp artifacts, or model-related dependencies require extra parity testing.

## Unsafe Code

Avoid introducing `unsafe` outside narrowly contained FFI boundaries.

Any new unsafe block must have a clear, reviewable invariant. When possible, expose a safe internal abstraction around unsafe operations.
