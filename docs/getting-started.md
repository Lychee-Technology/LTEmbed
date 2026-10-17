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

On macOS on Apple silicon, run the commands below inside a `linux/arm64` container; see
[development.md](./development.md#container-builds). An Intel Mac, like an x86_64 CPU
below v3, has no supported build: there the container would run under emulation.

You also need:

- Rust `1.94.0` (pinned in `rust-toolchain.toml`; `rustup` installs it automatically)
- `gh` (the GitHub CLI), `curl`, `sha256sum`, `tar` and Python 3
- about 0.45 GB of memory for short inputs, and up to about 1.6 GB for an 8192-token input
  (see [architecture.md](./architecture.md#long-inputs-and-memory))

## 2. Fetch the static llama.cpp release

Follow [development.md → Static llama.cpp artifacts](./development.md#static-llamacpp-artifacts).
It downloads and verifies the pinned release's variant for your CPU into its own
directory: `.llama-artifacts/extracted/` for `aarch64-graviton2`,
`.llama-artifacts/extracted-x86_64-v3/` for `x86_64-v3`. Then, from the repository root:

```bash
# aarch64 Linux:
export STATIC_LLAMA_DIR="$PWD/.llama-artifacts/extracted"
# x86-64-v3 Linux instead:
# export STATIC_LLAMA_DIR="$PWD/.llama-artifacts/extracted-x86_64-v3"
```

Without this variable, every `cargo build`, `cargo test` and `cargo run` fails in
`build.rs`.

## 3. Get a model bundle

A bundle is a directory with `model.gguf` and `build-info.json`; the engine tokenizes with
the GGUF's own vocabulary. To assemble the same `UD-Q5_K_XL` bundle that CI uses into
`./gguf_bundle` (about 210 MB of downloads, checked against pinned hashes):

```bash
python3 scripts/assemble_gguf_bundle.py gguf_bundle
```

See [bundle-format.md → Assembling a bundle](./bundle-format.md#assembling-a-bundle) for
other quants and for files already on disk.

## 4. Build and run the example

```bash
cargo run --example api_usage
```

`examples/api_usage.rs` looks for `gguf_bundle/` in the current directory and its parents,
embeds two queries, and prints the engine's [embedding space](#tagging-an-index) and the
first embedding's head:

```
embedding_space: ltembed-space/1;model-sha256=a9d7a3b7…;query-prefix=task: search result | query: ;document-prefix=title: none | text: ;output-dimension=512;l2-normalize=true
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
`EmbeddingInput::query` and the passages being searched in `EmbeddingInput::document`. The
engine adds EmbeddingGemma 2's `task: search result | query: ` or `title: none | text: `
prompt from `build-info.json`. Pass raw text; do not add the prompt yourself, or it is
applied twice.

### `EngineConfig`

| Field | Meaning | `Default` |
|---|---|---|
| `output_dimension` | Keep the first N of the 768 raw dimensions (Matryoshka truncation). Must be 1–768; the model is trained for 768, 512, 256 and 128. | `512` |
| `l2_normalize` | Scale each vector to unit length after truncation | `true` |

`EngineConfig::default()` gives `512` and `true`. Pick one setting and use it for both
indexing and querying.

### Tagging an index

`engine.embedding_space()?` names the vector space the engine embeds into: the sha256 of
its `model.gguf`, the prompts it adds, and its `EngineConfig`. Vectors are comparable only
between engines whose spaces are equal. Store its string form (`to_string()`) with an index
and compare it with the engine's before adding to or searching the index; see
[model-migration.md](./model-migration.md#moving-an-index). The first call hashes
`model.gguf`, about 0.1 s; later calls return the stored value.

### Threads and concurrency

- `from_gguf_bundle_dir` uses one llama.cpp thread.
- `from_gguf_bundle_dir_with_threads(dir, config, n)` sets `n` threads; `0` is rejected.
- `EmbeddingEngine` is `Send + Sync`. Calls on one engine tokenize their inputs in
  parallel, but encode them one at a time, under an internal mutex
  ([architecture.md](./architecture.md#concurrency)). To run inference in parallel, load
  several engines.
- `embed_batch` encodes its inputs one after another; it is not faster per input than
  repeated `embed` calls apart from saved overhead.

### Errors

All methods return `ltembed::error::LTEmbedError`. The ones you are most likely to hit:

| Error | Cause |
|---|---|
| `ModelLoad(MissingFile { .. })` | The bundle directory lacks `model.gguf` or `build-info.json` |
| `ModelLoad(UnsupportedModelFormat { .. })` | `build-info.json` is from an ONNX-era bundle |
| `ModelLoad(Config(_))` | `output_dimension` is 0 or above 768, or `n_threads` is 0 |
| `ModelLoad(UnsupportedPooling { .. })` | `build-info.json` is from a bundle for the previous model (`last_token` pooling); see [model-migration.md](./model-migration.md) |
| `ModelLoad(Runtime(_))` | The GGUF failed to load, does not match `build-info.json`, or its tokenization does not add BOS and EOS |
| `InputTooLong { tokens, bytes, max }` | An input is longer than the bundle's `max_length` (8192 tokens). `tokens` is its token count, or `None` for an input too large to fit at all (over 761,856 bytes with its prefix), which is rejected without being tokenized. Nothing is truncated, so split long text first. |

The full list is in [architecture.md](./architecture.md#errors).

## Next steps

- How it works: [architecture.md](./architecture.md)
- Running the tests: [testing.md](./testing.md)
- Measuring latency and quality: [benchmarking.md](./benchmarking.md)
- Getting a bundle from a release: [releases.md](./releases.md)
