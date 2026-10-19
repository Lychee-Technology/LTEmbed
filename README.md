# LTEmbed

LTEmbed is a Rust library that turns text into embedding vectors for retrieval. It runs
[EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) on the CPU through
**llama.cpp**, using the 5-bit **GGUF** `embeddinggemma-2-UD-Q5_K_XL.gguf` from
[`unsloth/embeddinggemma-2-GGUF`](https://huggingface.co/unsloth/embeddinggemma-2-GGUF). Its
entry point is [`EmbeddingEngine`](./src/engine/mod.rs).

> [!IMPORTANT]
> EmbeddingGemma 2 replaced `jinaai/jina-embeddings-v5-text-nano-retrieval`. Vectors from
> the two models are not comparable: re-embed stored vectors before switching. See
> [docs/model-migration.md](./docs/model-migration.md).

- **Backend:** llama.cpp, statically linked from the prebuilt archives of
  [`Lychee-Technology/static-llama-cpp-rs-builder`](https://github.com/Lychee-Technology/static-llama-cpp-rs-builder)
  through raw FFI. There is no ONNX Runtime or other shared library to ship; only system
  libraries (`libstdc++`, `libpthread`, `libm`, `libdl`) are linked dynamically.
- **Platform:** Linux on two CPU baselines. `aarch64-unknown-linux-gnu` uses archives built
  for Graviton2 / Neoverse N1. `x86_64-unknown-linux-gnu` uses archives built for
  x86-64-v3 (AVX2, FMA, BMI2). x86-64 CPUs below v3 are not supported: the code faults
  with `SIGILL`. On macOS on Apple silicon, build inside a `linux/arm64` container. Intel
  Macs are not supported: there that container would run under emulation.
- **Output:** mean pooling, projected from the 512-wide hidden state to 768 raw dimensions;
  truncated to 512 and L2-normalized by default. The model is trained for 768, 512, 256
  and 128.
- **Max input:** 8192 tokens (`engine.max_length()`). Longer inputs return an error;
  nothing is truncated, and `engine.token_count(input)` counts an input before it is
  embedded. Memory grows with input length, from about 0.45 GB for short inputs to about
  1.6 GB for an 8192-token input; a bundle can set a lower limit (see
  [docs/architecture.md](./docs/architecture.md#long-inputs-and-memory)).

## Model bundle

The engine loads a directory with two files:

| File | Contents |
|---|---|
| `model.gguf` | The GGUF model. CI and releases use the `UD-Q5_K_XL` quant. |
| `build-info.json` | Metadata: format, pooling, query/document prefixes, dimensions, max length |

Inputs are tokenized with the GGUF's own vocabulary; the bundle has no `tokenizer.json`.

The release workflow packages a ready-made `gguf_bundle/`, or you can assemble one from the
pinned Hugging Face revisions with `python3 scripts/assemble_gguf_bundle.py gguf_bundle`.
See [docs/bundle-format.md](./docs/bundle-format.md).

## Licensing

LTEmbed's original source code is dual-licensed under either the
[MIT license](./LICENSE-MIT) or the [Apache License 2.0](./LICENSE-APACHE), at your
option (`MIT OR Apache-2.0`). Third-party models, tokenizers, assets and dependencies
are not relicensed by this grant; their applicable upstream terms remain in effect.

The statically linked llama.cpp/ggml components are **MIT-licensed**. Their upstream
copyright and permission notice must accompany redistributed binaries, even when
statically linked; see [native dependency licenses and attribution](./licenses/llama.cpp/NOTICE.md).

The model, EmbeddingGemma 2, is licensed under the **Apache License 2.0**. Release
tarballs carry its license text and attribution in `licenses/embeddinggemma-2/`
([NOTICE.md](./licenses/embeddinggemma-2/NOTICE.md)); if you pass the bundle on without
the rest of the tarball, pass that directory on with it.

The previous model, Jina v5 Nano, is **CC BY-NC 4.0 (NonCommercial only)**; commercial use
requires separate permission from [sales@jina.ai](mailto:sales@jina.ai). `main` no longer
ships it, and the `v0.1.0` release asset that held it has been deleted with that release;
the `v0.1.0` tag remains and holds source code only. A copy of the asset downloaded before
then is still under the model's terms. The model's terms, attribution and redistribution
checklist are in
[licenses/jina-embeddings-v5-text-nano-retrieval/NOTICE.md](./licenses/jina-embeddings-v5-text-nano-retrieval/NOTICE.md),
which does not clear redistribution of that asset, resolve its tokenizer's upstream
rights, or grant additional rights.

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
you search over in `EmbeddingInput::document`. The engine adds the model's prompt from
`build-info.json` (`task: search result | query: ` or `title: none | text: `); pass raw
text.

`EngineConfig` picks the output dimension (1–768) and whether to L2-normalize.
`from_gguf_bundle_dir_with_threads` sets the llama.cpp thread count (default 1). One engine
encodes one input at a time, under a mutex; calls tokenize without it, so `token_count`
never waits ([docs/architecture.md](./docs/architecture.md#concurrency)).

**Token limit.** `engine.max_length()` is the most tokens an input may take, and
`engine.token_count(input)` counts an input as the engine does, prompt, `<bos>` and `<eos>`
included, over the limit too. A longer input returns `InputTooLong` and is never truncated,
so count and split long text first. An input too large to fit at all, over 761,856 bytes
with its prompt for the release bundle, has no count: `token_count` returns `InputTooLong`
for it as well, without tokenizing it
([docs/getting-started.md](./docs/getting-started.md#token-limit)).

**Tagging an index.** `engine.embedding_space()` names the model file, prompts and settings
the vectors depend on. Store its string form with an index and check it before every write
and search, so that vectors of different models or settings are never mixed
([docs/model-migration.md](./docs/model-migration.md#moving-an-index)).

## Building

Every `cargo build`, `test`, `clippy` or `run` needs `STATIC_LLAMA_DIR` set to the
extracted, checksum-verified static llama.cpp release that `.github/workflows/ci.yml`
pins (`v0.1.160-1`, llama.cpp `v0.7.0`), in the variant for the host: `aarch64-graviton2`
or `x86_64-v3`.

```bash
# aarch64 Linux:
export STATIC_LLAMA_DIR=/abs/path/to/.llama-artifacts/extracted
# x86-64-v3 Linux instead:
# export STATIC_LLAMA_DIR=/abs/path/to/.llama-artifacts/extracted-x86_64-v3
cargo build --release
cargo run --example api_usage        # needs ./gguf_bundle
```

Step-by-step instructions are in [docs/getting-started.md](./docs/getting-started.md).
Download steps and the container setup for other ARM64 hosts are in
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
| [Releases](./docs/releases.md) | Release tarballs, why UD-Q5_K_XL |
| [Model migration](./docs/model-migration.md) | Moving an index from the previous model; the Rust API changes (`HFTokenizer` removed) |
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
