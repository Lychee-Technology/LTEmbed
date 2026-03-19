# Development Workflow

## Rust Toolchain

This repository pins Rust via `rust-toolchain.toml`.

Use the pinned toolchain for local development so that `rustfmt`, `clippy`, and CI all agree.

## Install Git Hooks

Run:

```bash
./scripts/install-git-hooks.sh
```

This configures the current clone or worktree to use versioned hooks from `.githooks/`.

## Hook Behavior

- `pre-commit`: runs `cargo fmt --all --check` when staged Rust-related files changed
- `pre-push`: runs `cargo clippy --all-targets -- -D warnings`

## Manual Checks

If you want to run the same checks manually:

```bash
cargo fmt --all --check
cargo clippy --all-targets -- -D warnings
cargo test
```

## API Sanity Check

For a quick end-to-end check against local assets, run:

```bash
cargo check --all-targets
cargo run --example api_usage
cargo run --bin benchmark_ltembed -- --mode correctness --model-dir assets
```

The root [`README.md`](../README.md) is the primary guide for the Rust API surface and expected asset layout.
