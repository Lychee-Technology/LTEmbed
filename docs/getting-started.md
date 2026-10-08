# Getting Started

From a fresh clone to your first embedding.

## 1. Check the platform

LTEmbed links prebuilt static llama.cpp archives, so it builds only on Linux, on one of two
CPU baselines:

- **aarch64** (`aarch64-unknown-linux-gnu`): `armv8.2-a+fp16+dotprod+rcpc`, tuned for AWS
  Graviton2 (Neoverse N1). The CPU needs those extensions.
- **x86_64** (`x86_64-unknown-linux-gnu`): x86-64-v3 (AVX2, BMI1/2, F16C, FMA, LZCNT,
  MOVBE). Older x86-64 CPUs are not supported. On such a CPU the archives crash with
  `SIGILL` rather than fall back. `/lib64/ld-linux-x86-64.so.2 --help` must list
  `x86-64-v3 (supported, searched)`.

On macOS, or on an x86_64 CPU below v3, run the commands below inside a `linux/arm64`
container; see [development.md](./development.md#container-builds).

You also need:

- Rust `1.94.0` (pinned in `rust-toolchain.toml`; `rustup` installs it automatically)
- `gh` (the GitHub CLI), `curl`, `sha256sum` and `tar`

## 2. Fetch the static llama.cpp release

Follow [development.md → Static llama.cpp artifacts](./development.md#static-llamacpp-artifacts).
It downloads and verifies the pinned release's variant for your CPU
(`aarch64-graviton2` or `x86_64-v3`) into `.llama-artifacts/extracted/`. Then:

```bash
export STATIC_LLAMA_DIR="$PWD/.llama-artifacts/extracted"
```

Without this variable, every `cargo build`, `cargo test` and `cargo run` fails in
`build.rs`.

## 3. Get a model bundle

A bundle is a directory with `model.gguf`, `tokenizer.json` and `build-info.json`. To
assemble the same Q5_K_M bundle that CI uses into `./gguf_bundle`, run the script in
[bundle-format.md → Assembling a bundle by hand](./bundle-format.md#assembling-a-bundle-by-hand).

Use the model's own `tokenizer.json` from Hugging Face. `assets/tokenizer.json` in this
repository is a different tokenizer, used only by unit tests.

## 4. Build and run the example

```bash
cargo run --example api_usage
```

`examples/api_usage.rs` looks for `gguf_bundle/` in the current directory and its parents,
embeds two queries, and prints:

```
inputs: 2
embedding_dim: 512
first_embedding_head: [...]
```

## 5. Use the API

```rust
use ltembed::engine::{EmbeddingEngine, EmbeddingInput, EngineConfig};

let engine = EmbeddingEngine::from_gguf_bundle_dir(
    "gguf_bundle",
    EngineConfig { output_dimension: 512, l2_normalize: true },
)?;

let query = engine.embed(EmbeddingInput::query("how do I treat a cold?"))?;
let docs = engine.embed_batch(&[
    EmbeddingInput::document("Rest and fluids help most colds."),
    EmbeddingInput::document("Rust is a systems programming language."),
])?;
```

Each call returns `Vec<f32>` (or one per input). With `l2_normalize: true`, the dot product
of two vectors is their cosine similarity.

### Queries and documents

The bundle defines separate prefixes for queries and documents. Wrap search text in
`EmbeddingInput::query` and the passages being searched in `EmbeddingInput::document`. The engine adds the
`Query: ` or `Document: ` prefix from `build-info.json`. Pass raw text; do not add the
prefix yourself.

### `EngineConfig`

| Field | Meaning | `Default` |
|---|---|---|
| `output_dimension` | Keep the first N of the 768 raw dimensions (Matryoshka truncation). Must be 1–768. | `512` |
| `l2_normalize` | Scale each vector to unit length after truncation | `true` |

`EngineConfig::default()` gives `512` and `true`. Pick one setting and use it for both
indexing and querying.

### Threads and concurrency

- `from_gguf_bundle_dir` uses one llama.cpp thread.
- `from_gguf_bundle_dir_with_threads(dir, config, n)` sets `n` threads; `0` is rejected.
- `EmbeddingEngine` is `Send + Sync`. Calls on one engine are serialized by an internal
  mutex. To run calls in parallel, load several engines.
- `embed_batch` encodes its inputs one after another; it is not faster per input than
  repeated `embed` calls apart from saved overhead.

### Errors

All methods return `ltembed::error::LTEmbedError`. The ones you are most likely to hit:

| Error | Cause |
|---|---|
| `ModelLoad(MissingFile { .. })` | The bundle directory lacks one of the three files |
| `ModelLoad(UnsupportedModelFormat { .. })` | `build-info.json` is from an ONNX-era bundle |
| `ModelLoad(Config(_))` | `output_dimension` is 0 or above 768, or `n_threads` is 0 |
| `ModelLoad(Runtime(_))` | The GGUF or tokenizer failed to load, or the GGUF does not match `build-info.json` |
| `InputTooLong { tokens, max }` | An input is longer than 8192 tokens. Nothing is truncated, so split long text first. |

The full list is in [architecture.md](./architecture.md#errors).

## Next steps

- How it works: [architecture.md](./architecture.md)
- Running the tests: [testing.md](./testing.md)
- Measuring latency and quality: [benchmarking.md](./benchmarking.md)
- Getting a bundle from a release: [releases.md](./releases.md)
