# LTEmbed Documentation

Everything outside [history/](#history) describes the current `main` branch: the
llama.cpp/GGUF backend. Start with the [project README](../README.md) for a summary.

## By task

| I want to… | Read |
|---|---|
| Build LTEmbed and get my first embedding | [getting-started.md](./getting-started.md) |
| Understand how text becomes a vector | [architecture.md](./architecture.md) |
| Make or check a model bundle | [bundle-format.md](./bundle-format.md) |
| Set up a dev environment, including on macOS or x86_64 | [development.md](./development.md) |
| Know what the tests cover and what CI runs | [testing.md](./testing.md) |
| Measure latency, parity or retrieval quality | [benchmarking.md](./benchmarking.md) |
| Cut a release or see why Q5_K_M was chosen | [releases.md](./releases.md) |
| Use the Python benchmark and fixture scripts | [scripts/README.md](../scripts/README.md) |
| Follow the Rust coding conventions | [rust-coding-std.md](./rust-coding-std.md) |

## History

These documents are kept for their decisions and data. **They do not describe current
`main`.** Each one starts with a note saying what it applied to and what has changed. Code,
commands and file names in them may no longer exist.

**ONNX Runtime backend** (`main` until #149, 2026-07-09; frozen on the `ort` branch)

- [history/ort/ort-rust-lambda-guidelines.md](./history/ort/ort-rust-lambda-guidelines.md):
  building ONNX Runtime and the `ort` crate for arm64 Lambda
- [history/ort/lambda-s3-files-deployment.md](./history/ort/lambda-s3-files-deployment.md):
  a Lambda ZIP design with `libonnxruntime.so` in the package and the model on an S3 Files
  mount

**Pure-Rust `matrixmultiply` backend** (on the `matrixmultiply` branch)

- [history/matrixmultiply/matrixmultiply-neon-tuning.md](./history/matrixmultiply/matrixmultiply-neon-tuning.md):
  NEON GEMM kernel tuning investigation
- [history/matrixmultiply/matrixmultiply-neon-8x12-plan.md](./history/matrixmultiply/matrixmultiply-neon-8x12-plan.md):
  plan for an opt-in 8x12 NEON kernel, not carried to `main`
- [history/matrixmultiply/issue-30-profiling.md](./history/matrixmultiply/issue-30-profiling.md):
  profiling notes for issue #30, measured on Apple Silicon

**Migration to llama.cpp/GGUF**

- [history/migrations/llama-cpp-rs-migration-evaluation.md](./history/migrations/llama-cpp-rs-migration-evaluation.md):
  evaluation of replacing `ort` with `llama-cpp-rs`. Superseded: `main` uses raw FFI to
  static archives instead.
- [history/migrations/llama-cpp-spike-results.md](./history/migrations/llama-cpp-spike-results.md):
  the de-risking spike that picked Q5_K_M. Its still-valid results are in
  [releases.md](./releases.md#why-q5_k_m) and
  [benchmarking.md](./benchmarking.md#recorded-results).

**Plans**

- [history/plans/ltembed-backend-branch-split-plan.md](./history/plans/ltembed-backend-branch-split-plan.md):
  splitting the `matrixmultiply` line off `main` (May 2026). Carried out; its premise that
  `main` stays ONNX Runtime-only was superseded by #149.
