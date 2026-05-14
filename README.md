# LTEmbed

LTEmbed is a Rust embedding library centered on [`OnnxEngine`](./src/engine.rs). The default path uses ONNX Runtime (`ort`) with a builder-produced `ort_bundle/` for `jinaai/jina-embeddings-v5-text-nano-retrieval`.

## Backend Branches

- `main` is the ONNX Runtime / `ort_bundle` line.
- The legacy matrixmultiply backend and related ARM64 tuning experiments live on the `matrixmultiply` branch.

## Bundle Layout

Expected local bundle contents:

- `model.ort` — ONNX model weights (path configured separately from bundle)
- `ort_bundle/tokenizer.json`
- `ort_bundle/build-info.json`

The ONNX Runtime dynamic library (`libonnxruntime.so`) is resolved automatically:
first from `ORT_DYLIB_PATH` env var, then from `ort_bundle/libonnxruntime.so` if present.

Runtime notes:

- `OnnxEngine::from_bundle_dir(...)` takes a bundle directory and an explicit
  path to `model.ort`.
- `OnnxEngineConfig` controls the returned embedding dimension and whether outputs are L2-normalized.
- The engine validates bundle metadata at startup and returns `ModelLoad` on missing files or incompatible metadata.

## API

```rust
use ltembed::engine::{EmbeddingInput, OnnxEngine, OnnxEngineConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine = OnnxEngine::from_bundle_dir(
        "ort_bundle",
        "ort_bundle/model.ort",
        OnnxEngineConfig {
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
- The model's raw pooled output is `768` dimensions.
- `OnnxEngineConfig` can truncate to a smaller output dimension and optionally L2-normalize before returning results.
- Maximum tokenizer length is `8192`; overlong inputs return `InputTooLong`.

## Benchmarks, Fixtures, And Releases

- `scripts/bench_pytorch.py` is the Python reference runner for the Jina model.
- `scripts/generate_fixtures.py` regenerates correctness fixtures in the new `kind + text + embedding` schema.
- `scripts/run_embedding_benchmarks.py` orchestrates warm, cold, and correctness runs against `ort_bundle_dir`.
- LTEmbed release tarballs are expected to contain a source snapshot plus `ort_bundle/` at the tarball root.

## Out Of Scope

- Lambda packaging is not part of the current default path.
