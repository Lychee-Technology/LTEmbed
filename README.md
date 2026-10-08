# LTEmbed

LTEmbed is a Rust library that turns text into embedding vectors for retrieval. It runs
[`jinaai/jina-embeddings-v5-text-nano-retrieval`](https://huggingface.co/jinaai/jina-embeddings-v5-text-nano-retrieval)
on the CPU through **llama.cpp**, using a quantized **GGUF** model. Its entry point is
[`EmbeddingEngine`](./src/engine/mod.rs).

- **Backend:** llama.cpp, statically linked from the prebuilt archives of
  [`Lychee-Technology/static-llama-cpp-rs-builder`](https://github.com/Lychee-Technology/static-llama-cpp-rs-builder)
  through raw FFI. There is no ONNX Runtime or other shared library to ship; only system
  libraries (`libstdc++`, `libpthread`, `libm`, `libdl`) are linked dynamically.
- **Platform:** Linux on two CPU baselines. `aarch64-unknown-linux-gnu` uses archives built
  for Graviton2 / Neoverse N1. `x86_64-unknown-linux-gnu` uses archives built for
  x86-64-v3 (AVX2, FMA, BMI2). x86-64 CPUs below v3 are not supported: the code faults
  with `SIGILL`. On macOS or such CPUs, build inside a `linux/arm64` container.
- **Output:** 768 raw dimensions, truncated to 512 and L2-normalized by default.
- **Max input:** 8192 tokens. Longer inputs return an error; nothing is truncated.

## Model bundle

The engine loads a directory with three files:

| File | Contents |
|---|---|
| `model.gguf` | The GGUF model. CI and releases use the `Q5_K_M` quant. |
| `tokenizer.json` | The model's own Hugging Face tokenizer |
| `build-info.json` | Metadata: format, pooling, query/document prefixes, dimensions, max length |

The release workflow packages a ready-made `gguf_bundle/`, or you can assemble one with
`curl`. See [docs/bundle-format.md](./docs/bundle-format.md).

## Usage

```rust
use ltembed::engine::{EmbeddingEngine, EmbeddingInput, EngineConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine = EmbeddingEngine::from_gguf_bundle_dir(
        "gguf_bundle",
        EngineConfig {
            output_dimension: 512,
            l2_normalize: true,
        },
    )?;

    let query = engine.embed(EmbeddingInput::query("Hello, world!"))?;
    let docs = engine.embed_batch(&[
        EmbeddingInput::document("LTEmbed Rust API example"),
        EmbeddingInput::document("Another passage"),
    ])?;

    assert_eq!(query.len(), 512);
    assert_eq!(docs.len(), 2);
    Ok(())
}
```

A runnable version is in [`examples/api_usage.rs`](./examples/api_usage.rs).

**Queries vs. documents.** Wrap search text in `EmbeddingInput::query` and the passages
you search over in `EmbeddingInput::document`. The engine adds the `Query: ` or
`Document: ` prefix from `build-info.json`; pass raw text.

`EngineConfig` picks the output dimension (1–768) and whether to L2-normalize.
`from_gguf_bundle_dir_with_threads` sets the llama.cpp thread count (default 1). One engine
handles one call at a time.

## Building

Every `cargo build`, `test`, `clippy` or `run` needs `STATIC_LLAMA_DIR` set to the
extracted, checksum-verified static llama.cpp release that `.github/workflows/ci.yml`
pins (`v0.1.159-1`, llama.cpp `v0.6.0`), in the variant for the host: `aarch64-graviton2`
or `x86_64-v3`.

```bash
export STATIC_LLAMA_DIR=/abs/path/to/.llama-artifacts/extracted
cargo build --release
cargo run --example api_usage        # needs ./gguf_bundle
```

Step-by-step instructions are in [docs/getting-started.md](./docs/getting-started.md).
Download steps and the container setup for other hosts are in
[docs/development.md](./docs/development.md).

## Documentation

| | |
|---|---|
| [Getting started](./docs/getting-started.md) | First build and first embedding |
| [Architecture](./docs/architecture.md) | Load and inference pipeline, errors, concurrency |
| [Bundle format](./docs/bundle-format.md) | Bundle files and `build-info.json` fields |
| [Development](./docs/development.md) | Toolchain, static llama.cpp release, containers, hooks |
| [Testing](./docs/testing.md) | Test suites, fixtures, what CI runs |
| [Benchmarking](./docs/benchmarking.md) | Latency, parity and retrieval benchmarks, recorded results |
| [Releases](./docs/releases.md) | Release tarballs, why Q5_K_M |
| [Scripts](./scripts/README.md) | Python benchmark and fixture tools |

The full index, including archived design documents, is [docs/README.md](./docs/README.md).

## Out of scope

- **Deployment.** This repository ships a library and a release tarball with the source
  and a model bundle. It has no AWS Lambda handler, packaging or deploy workflow. The
  bundle is sized to fit Lambda's 250 MB package limit, but Lambda is not a supported
  target.
- **Other platforms and accelerators.** No macOS or Windows builds, no x86-64 builds
  below x86-64-v3, and no GPU.
- **Other models.** Only this model is tested, and there is no model conversion tooling.
- **Multi-sequence batching.** `embed_batch` encodes inputs one after another.

## Other branches

- `ort`: the earlier ONNX Runtime backend (`OnnxEngine`, `ort_bundle`), frozen since the
  move to llama.cpp in #149.
- `matrixmultiply`: the earlier pure-Rust BERT backend and its NEON tuning experiments.

Their design notes are in [docs/history/](./docs/README.md#history). They do not describe
`main`.
