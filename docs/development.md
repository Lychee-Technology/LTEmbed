# Development Workflow

## Rust Toolchain

This repository pins Rust via `rust-toolchain.toml`.

Use the pinned toolchain for local development so that `rustfmt`, `clippy`, and CI all agree.

## Branch Intent

- The inference backend is **llama.cpp / GGUF** (`EmbeddingEngine` + `LlamaBackend`), consuming
  the prebuilt static archives from `Lychee-Technology/static-llama-cpp-rs-builder`. ONNX
  Runtime has been removed.
- The `ort` branch is the frozen ONNX Runtime backup (`OnnxEngine` + `ort_bundle`).
- Use the `matrixmultiply` branch for the legacy matrixmultiply backend, kernel tuning, and related benchmark experiments.

## Building (aarch64 Linux only)

The crate links the prebuilt static llama.cpp archives, so it builds and tests **only on
`aarch64-unknown-linux-gnu`** with the release artifacts present. On a macOS/Apple-Silicon
host, work inside the dev container:

```bash
# One-time: download + SHA-verify the builder release into .llama-artifacts/extracted/
#           and a GGUF into .llama-artifacts/gguf/ (see docs/llama-cpp-spike-results.md),
#           then build the image:
docker build -f .llama-artifacts/llama-dev.Dockerfile -t ltembed-llama-dev .llama-artifacts

# Run any cargo command in the container (STATIC_LLAMA_DIR must point at the verified release):
.llama-artifacts/dev.sh cargo build
.llama-artifacts/dev.sh cargo test         # golden parity needs LTEMBED_TEST_BUNDLE_DIR too
```

`build.rs` reads `STATIC_LLAMA_DIR` and emits the link line from the release's `consume.build.rs`.
A GGUF bundle is a directory with `model.gguf` + `tokenizer.json` + `build-info.json` (use the
model's real tokenizer — **not** `assets/tokenizer.json`, which is a stale placeholder).

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

For a quick end-to-end check against a local GGUF bundle (inside the dev container), run:

```bash
cargo check --all-targets
cargo run --example api_usage            # looks for ./gguf_bundle
# End-to-end inference through the real engine; --mode is one of warm|cold|retrieval.
cargo run --bin benchmark_ltembed -- --mode retrieval --bundle-dir <gguf_bundle> \
  --retrieval-eval-path scripts/retrieval_eval_cases.json --output-dimension 512 --l2-normalize true
```

The root [`README.md`](../README.md) is the primary guide for the Rust API surface and expected asset layout.
