# Design

## Overview

LTEmbed's default embedding path is `OnnxEngine`. It owns:

- an `ort::session::Session`
- an `HFTokenizer`
- bundle metadata plus explicit runtime postprocess config

The target model is `jinaai/jina-embeddings-v5-text-nano-retrieval`, loaded from `ort_bundle/model.ort`.

## Public Contract

`OnnxEngine::from_bundle_dir(bundle_dir, config)` loads `model.ort`, `tokenizer.json`, `libonnxruntime.so`, and `build-info.json`, then validates:

- required inputs: `input_ids`, `attention_mask`
- required output: `last_hidden_state`
- retrieval metadata: prefixes, pooling, raw hidden size
- runtime postprocess config: output dimension and L2 normalization

`embed` and `embed_batch` accept typed retrieval inputs:

```rust
use ltembed::engine::{EmbeddingInput, OnnxEngine, OnnxEngineConfig};

let engine = OnnxEngine::from_bundle_dir(
    "ort_bundle",
    OnnxEngineConfig {
        output_dimension: 512,
        l2_normalize: true,
    },
)?;
let query = engine.embed(EmbeddingInput::query("hello"))?;
let doc = engine.embed(EmbeddingInput::document("world"))?;
```

The engine, not the caller, applies the retrieval prefixes.

## Inference Pipeline

1. Convert `EmbeddingInputKind` into `Query: ` or `Document: ` prefixed text.
2. Tokenize with explicit `InputTooLong` failure at the bundle metadata max length.
3. Build padded `input_ids` and `attention_mask` tensors.
4. Run ORT inference and read `last_hidden_state`.
5. Apply last-token pooling using the final active token from `attention_mask`.
6. Truncate the pooled `768`-d vector to `config.output_dimension`.
7. L2-normalize the truncated vector when `config.l2_normalize = true`.

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
- The default path loads `libonnxruntime.so` from `ort_bundle/`.
- LTEmbed release tarballs will carry a source snapshot plus `ort_bundle/` at the root.
- Lambda packaging and minimal ORT build production are deferred work; the current design only aligns the runtime and release asset contract.
