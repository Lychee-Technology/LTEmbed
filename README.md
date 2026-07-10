# LTEmbed

LTEmbed is a Rust embedding library centered on [`EmbeddingEngine`](./src/engine/mod.rs).
The inference backend is **llama.cpp / GGUF** for `jinaai/jina-embeddings-v5-text-nano-retrieval`,
linked from the prebuilt static archives published by
[`Lychee-Technology/static-llama-cpp-rs-builder`](https://github.com/Lychee-Technology/static-llama-cpp-rs-builder).

The backend lives behind an internal `EmbeddingBackend` trait (`LlamaBackend` is the only
implementation today) so additional backends can be added without touching the shared
prefix → tokenize → pool → truncate/normalize pipeline.

## Backend Branches

- The inference backend is llama.cpp/GGUF (`EmbeddingEngine`).
- The `ort` branch is the frozen ONNX Runtime backup (`OnnxEngine` / `ort_bundle`).
- The legacy matrixmultiply backend and related ARM64 tuning experiments live on the `matrixmultiply` branch.

## Building

The crate links the prebuilt static llama.cpp archives, so it builds **only on
`aarch64-unknown-linux-gnu`** with `STATIC_LLAMA_DIR` pointing at a verified, extracted
release. On a macOS/Apple-Silicon host, work in the dev container — see
[`docs/development.md`](./docs/development.md).

## Bundle Layout

A GGUF bundle directory contains:

- `model.gguf` — the GGUF model (e.g. `Q5_K_M`)
- `tokenizer.json` — the **model's own** tokenizer (a 128k BPE tokenizer; note
  `assets/tokenizer.json` is a stale placeholder and must not be used for real bundles)
- `build-info.json` — metadata (`model_format: "gguf"`, pooling, prefixes, dims, max length)

There is no ONNX Runtime dynamic library or `ORT_DYLIB_PATH` — llama.cpp is statically linked.

Runtime notes:

- `EmbeddingEngine::from_gguf_bundle_dir(dir, config)` loads a bundle directory.
- `EngineConfig` controls the returned embedding dimension and whether outputs are L2-normalized.
- The engine validates bundle metadata at startup and returns `ModelLoad` on missing files or
  incompatible metadata (including a non-`gguf` `model_format`).
- The engine runs with 1 llama.cpp thread by default;
  `EmbeddingEngine::from_gguf_bundle_dir_with_threads(...)` sets it explicitly (`0` is rejected with `ModelLoad`).

## API

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

    let single = engine.embed(EmbeddingInput::query("Hello, world!"))?;
    let batch = engine.embed_batch(&[
        EmbeddingInput::query("alpha"),
        EmbeddingInput::document("beta"),
    ])?;

    assert_eq!(single.len(), 512);
    assert_eq!(batch[0].len(), 512);
    Ok(())
}
```

## Retrieval Semantics

- Callers pass typed inputs: `EmbeddingInput::query(...)` or `EmbeddingInput::document(...)`.
- LTEmbed reads `Query: ` and `Document: ` prefixes from bundle metadata and applies them internally.
- The model runs as a non-causal encoder with last-token pooling; the raw pooled output is `768` dimensions.
- `EngineConfig` can truncate to a smaller output dimension (Matryoshka) and optionally L2-normalize before returning results.
- Maximum tokenizer length is `8192`; overlong inputs return `InputTooLong`.

## Benchmarks, Fixtures, And Releases

- `scripts/bench_pytorch.py` is the Python reference runner for the Jina model.
- `scripts/generate_fixtures.py` regenerates the **immutable PyTorch/F32 golden** fixtures
  (`kind + text + embedding` schema) — never regenerate the golden from GGUF output.
- `scripts/run_embedding_benchmarks.py` orchestrates warm, cold, and correctness runs against a GGUF `--bundle-dir`.
- Quant selection and parity/latency results are documented in [`docs/llama-cpp-spike-results.md`](./docs/llama-cpp-spike-results.md).

## Out Of Scope

- Lambda packaging is not part of the current default path.
