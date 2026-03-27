# Design

## Overview

LTEmbed's default embedding path is `OnnxEngine`. It owns:

- an `ort::session::Session`
- an `HFTokenizer`
- ONNX input/output contract metadata

The target model is `jinaai/jina-embeddings-v5-text-nano-retrieval`, loaded from `assets/onnx/model_q4f16.onnx`.

## Public Contract

`OnnxEngine::new(model_path, tokenizer_path)` loads the tokenizer and ONNX graph, then validates:

- required inputs: `input_ids`, `attention_mask`
- required output: `last_hidden_state`
- raw hidden size: `768`

`embed` and `embed_batch` accept typed retrieval inputs:

```rust
use ltembed::engine::{EmbeddingInput, OnnxEngine};

let engine = OnnxEngine::new(
    "assets/onnx/model_q4f16.onnx",
    "assets/tokenizer.json",
)?;
let query = engine.embed(EmbeddingInput::query("hello"))?;
let doc = engine.embed(EmbeddingInput::document("world"))?;
```

The engine, not the caller, applies the retrieval prefixes.

## Inference Pipeline

1. Convert `EmbeddingInputKind` into `Query: ` or `Document: ` prefixed text.
2. Tokenize with explicit `InputTooLong` failure at `8192`.
3. Build padded `input_ids` and `attention_mask` tensors.
4. Run ORT inference and read `last_hidden_state`.
5. Apply last-token pooling using the final active token from `attention_mask`.
6. Truncate the pooled `768`-d vector to `512`.
7. L2-normalize the truncated vector.

## Benchmark And Fixture Semantics

- Benchmark scenarios store `text + kind`, not caller-prefixed strings.
- Python reference scripts follow the same last-token pooling and `768 -> 512 -> normalize` post-processing contract.
- Golden fixtures use the new schema:

```json
{
  "model": "jinaai/jina-embeddings-v5-text-nano-retrieval",
  "raw_dim": 768,
  "dim": 512,
  "max_length": 8192,
  "fixtures": [
    {
      "kind": "query",
      "text": "Hello, world!",
      "embedding": []
    }
  ]
}
```

## Runtime Notes

- ORT dynamic loading follows [`ort-rust-lambda-guidelines.md`](./ort-rust-lambda-guidelines.md).
- `ORT_DYLIB_PATH` may be used to point at the ONNX Runtime shared library.
- Lambda packaging and minimal ORT builds are deferred work; the current design only aligns the runtime contract and asset layout.
