# Design

## Overview

LTEmbed's embedding path is `EmbeddingEngine`. It owns:

- an `EmbeddingBackend` (currently `LlamaBackend`, backed by statically-linked llama.cpp/GGUF)
- an `HFTokenizer`
- bundle metadata plus explicit runtime postprocess config

The target model is `jinaai/jina-embeddings-v5-text-nano-retrieval`, loaded from a GGUF bundle
(`model.gguf`). The backend is isolated behind the `EmbeddingBackend` trait: an implementation
returns the raw, un-normalized, last-token-pooled `768`-d vector per input, and the shared engine
owns prefixing, tokenization, truncation, and normalization.

## Public Contract

`EmbeddingEngine::from_gguf_bundle_dir(bundle_dir, config)` loads `model.gguf`,
`tokenizer.json`, and `build-info.json` from the directory, then validates:

- `model_format` is `gguf` (stale ORT metadata is rejected)
- retrieval metadata: prefixes, pooling, raw hidden size
- runtime postprocess config: output dimension and L2 normalization

`embed` and `embed_batch` accept typed retrieval inputs:

```rust
use ltembed::engine::{EmbeddingEngine, EmbeddingInput, EngineConfig};

let engine = EmbeddingEngine::from_gguf_bundle_dir(
    "gguf_bundle",
    EngineConfig {
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
3. Feed the token ids to the backend, which runs the model as a non-causal encoder with
   last-token pooling and returns the raw, un-normalized `768`-d pooled vector per input.
4. Truncate the pooled `768`-d vector to `config.output_dimension` (Matryoshka).
5. L2-normalize the truncated vector when `config.l2_normalize = true`.

## Benchmark And Fixture Semantics

- Benchmark scenarios store `text + kind`, not caller-prefixed strings.
- Python reference scripts follow the same last-token pooling and `768 -> 512 -> normalize` post-processing contract.
- Golden fixtures are the **immutable PyTorch/F32 reference** (do not regenerate from GGUF output):

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

- llama.cpp/ggml is statically linked from a prebuilt release (see `build.rs` and
  [`development.md`](./development.md)); there is no runtime dynamic library or `ORT_DYLIB_PATH`.
- The crate builds only on `aarch64-unknown-linux-gnu` with `STATIC_LLAMA_DIR` set.
- Lambda packaging and the GGUF release pipeline are deferred work; the current design only
  aligns the runtime and bundle contract.
