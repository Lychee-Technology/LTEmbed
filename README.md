# LTEmbed

LTEmbed is a Rust embedding library centered on [`OnnxEngine`](./src/engine.rs). The default path uses ONNX Runtime (`ort`) with `jinaai/jina-embeddings-v5-text-nano-retrieval`.

## Model Layout

Expected local assets:

- `assets/tokenizer.json`
- `assets/onnx/model_q4f16.onnx`

Runtime notes:

- `ORT_DYLIB_PATH` can point to the ONNX Runtime shared library when the host environment does not already provide it.
- The engine validates the ONNX graph contract at startup and returns `ModelLoad` on missing files or incompatible I/O.

## API

```rust
use ltembed::engine::{EmbeddingInput, OnnxEngine};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine = OnnxEngine::new(
        "assets/onnx/model_q4f16.onnx",
        "assets/tokenizer.json",
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
- LTEmbed applies the required `Query: ` or `Document: ` prefix internally.
- The model's raw pooled output is `768` dimensions.
- LTEmbed truncates to `512` and then L2-normalizes before returning results.
- Maximum tokenizer length is `8192`; overlong inputs return `InputTooLong`.

## Benchmarks And Fixtures

- `scripts/bench_pytorch.py` is the Python reference runner for the Jina model.
- `scripts/generate_fixtures.py` regenerates correctness fixtures in the new `kind + text + embedding` schema.
- `scripts/run_embedding_benchmarks.py` orchestrates warm, cold, and correctness runs against the ONNX path.

## Out Of Scope

- Lambda packaging is not part of the current default path.
- The handwritten BERT / `model.safetensors` route is no longer the primary integration story.
