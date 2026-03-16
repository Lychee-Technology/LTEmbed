# LTEmbed — ARM64 Embedding Engine

## 1. Overview

**LTEmbed** is a Rust library for generating L2-normalized vector embeddings from BERT-family models. It exposes a simple, idiomatic Rust API and is optimized for **ARM64 serverless compute** (AWS Graviton), where its minimal binary size and zero-copy model loading make it particularly effective.

The v1 release targets **e5-small-v2**. Support for additional models (e.g., bge-small-zh) is planned for future releases.

```rust
// Typical usage
let engine = ZeroVecEngine::new(
    "assets/model.safetensors",
    &config_json,
    "assets/tokenizer.json",
    Box::new(MeanPooling),
)?;

let embedding: Vec<f32> = engine.embed("query: Hello, world!")?;
// → 384 f32 values, L2-normalized
```

LTEmbed is transport-agnostic. It does not depend on any HTTP framework or runtime. Example integrations (AWS Lambda, CLI, gRPC server) are deployment-layer concerns outside the library's scope.

---

## 2. Public API

### `ZeroVecEngine`

```rust
pub struct ZeroVecEngine { /* opaque */ }

impl ZeroVecEngine {
    /// Load the engine from local file paths.
    /// Weights are memory-mapped (no heap copy of the model file).
    /// Intended to be called once at startup and reused across requests.
    pub fn new(
        safetensors_path: &str,
        config_json: &str,
        tokenizer_path: &str,
        pooling: Box<dyn Pooling>,
    ) -> Result<Self, LTEmbedError>;

    /// Embed a single text string.
    /// Returns a 384-dimensional L2-normalized vector.
    /// Returns Err(LTEmbedError::InputTooLong) if the tokenized length exceeds 512.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, LTEmbedError>;

    /// Embed a batch of texts. Processes each input sequentially (v1).
    /// Returns one Vec<f32> per input, in the same order.
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, LTEmbedError>;
}
```

### `LTEmbedError`

```rust
#[derive(Debug, thiserror::Error)]
pub enum LTEmbedError {
    #[error("Model load failed: {0}")]
    ModelLoad(String),

    #[error("Tokenization failed: {0}")]
    Tokenization(String),

    #[error("Inference failed: {0}")]
    Inference(String),

    #[error("Input too long: {tokens} tokens exceeds the {max} token limit")]
    InputTooLong { tokens: usize, max: usize },

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}
```

### `Pooling` Trait

```rust
pub trait Pooling: Send + Sync {
    /// Collapse [seq_len][hidden_size] last hidden state → [hidden_size] vector.
    /// `attention_mask`: 1 = real token, 0 = padding.
    fn pool(
        &self,
        last_hidden_state: &[Vec<f32>],
        attention_mask: &[u32],
    ) -> Result<Vec<f32>, LTEmbedError>;
}

pub struct MeanPooling; // average of non-padding tokens
pub struct CLSPooling;  // [CLS] token at index 0
```

---

## 3. Internal Architecture

The library is divided into four layers to separate concerns:

### Layer 1: Public Interface (`engine.rs`)

`ZeroVecEngine` is the single entry point callers interact with. It holds the model, tokenizer, and pooling strategy, and orchestrates the inference pipeline.

### Layer 2: Pluggable Traits (`traits/`)

- **`Tokenizer`** — converts text to `(input_ids, attention_mask, token_type_ids)`. Returns `Err(InputTooLong)` when the encoded length exceeds `max_length`. Does not silently truncate.
- **`Pooling`** — collapses the last hidden state tensor to a single vector.

### Layer 3: Model Backend (`models/`)

A thin wrapper around `candle_transformers::models::bert::BertModel`. Loads weights via `VarBuilder::from_mmaped_safetensors` (zero-copy mmap). Executes the BERT forward pass and returns the last hidden state as `Vec<Vec<f32>>`.

### Layer 4: Math Utilities (`utils.rs`)

`l2_normalize(v: &[f32]) -> Vec<f32>` — normalizes the pooled vector to unit length so that downstream cosine similarity searches work correctly:

$$v_{norm} = \frac{v}{\sqrt{\sum_{i=1}^{n} v_i^2}}$$

---

## 4. Memory & Performance

### Initialization

1. `mmap` syscall maps `model.safetensors` into virtual address space — no heap copy (~1 ms).
2. `BertModel::load` traverses tensor headers to build the model graph.
3. **First `embed()` call triggers OS page faults** as weight pages are loaded from disk on demand. For a 130 MB model, this takes ~1–3 s.

The mmap approach achieves the minimum possible initialization overhead: no heap allocation, minimal startup latency. For serverless deployments where cold starts matter, consider calling `engine.embed("warmup")` immediately after `new()` to front-load the page faults before the first real request.

### Warm Invocations

Subsequent calls to `embed()` run in tens of milliseconds. ARM64 NEON SIMD acceleration is enabled when the binary is compiled with:

```bash
RUSTFLAGS="-C target-cpu=neoverse-n1"   # AWS Graviton2
```

Without this flag the compiler targets the conservative ARMv8 baseline. See `build.sh`.

### Batching (v1)

`embed_batch()` processes inputs sequentially. True batched inference (stacking N inputs into a single forward pass) is planned as a future optimization. For the typical serverless payload of 1–10 inputs, sequential processing is appropriate.

### Deployment Package Size (AWS Lambda)

| Component | Estimated Size |
|---|---|
| `model.safetensors` (e5-small-v2) | ~130 MB |
| `bootstrap` binary (Rust release build) | ~30–60 MB |
| `tokenizer.json` + `config.json` | ~2 MB |
| **Total** | **~165–195 MB** |

Lambda's 250 MB unzipped limit accommodates e5-small-v2. Larger models (e.g., bge-base-zh ~400 MB) require a Lambda Layer or Amazon EFS strategy.

---

## 5. Example: AWS Lambda Deployment

LTEmbed is a library. The repository includes an example Lambda binary (`src/main.rs`) that demonstrates one way to deploy it. The binary:

- Wraps the library with `lambda_http`
- Initializes the engine in a `static OnceLock` (once per container instance, no re-initialization on warm invocations)
- Accepts `POST` requests with JSON body `{"inputs": ["text1", "text2"]}`
- Returns `{"embeddings": [[...], ...]}` on success
- Maps library errors to appropriate HTTP responses (`LTEmbedError::InputTooLong` → 400, others → 500)

The library itself has no dependency on `lambda_http`, HTTP, or any async runtime. These are binary-layer concerns only.

---

## 6. Project Structure

```
LTEmbed/
├── Cargo.toml
├── src/
│   ├── lib.rs               # Library root — public API
│   ├── engine.rs            # ZeroVecEngine
│   ├── error.rs             # LTEmbedError
│   ├── utils.rs             # l2_normalize
│   ├── traits/
│   │   ├── mod.rs
│   │   ├── tokenizer.rs     # Tokenizer trait + HFTokenizer
│   │   └── pooling.rs       # Pooling trait + MeanPooling / CLSPooling
│   ├── models/
│   │   ├── mod.rs
│   │   └── bert.rs          # candle-transformers BertModel wrapper
│   └── main.rs              # Example: AWS Lambda deployment binary
├── tests/
│   ├── integration_tests.rs
│   └── fixtures/
│       └── test_fixtures.json
├── benches/
│   └── inference.rs         # Criterion warm-invocation benchmarks
├── scripts/
│   └── generate_fixtures.py
├── build.sh                 # Cross-compile for aarch64-unknown-linux-gnu
├── Dockerfile               # Amazon Linux 2023 build environment
└── assets/
    ├── config.json          # BERT architecture config (committed)
    ├── tokenizer.json       # WordPiece vocabulary (committed)
    └── model.safetensors    # e5-small-v2 weights (~130 MB, NOT committed)
```

---

## 7. Future Roadmap

- **Multi-model support** — runtime model selection via config, enabling bge-small-zh and other BERT-family models without code changes.
- **True batched inference** — single forward pass for N inputs, improving throughput for bulk embedding workloads.
- **INT8 quantization** — reduce model size and improve inference speed.
- **Additional deployment examples** — CLI tool, gRPC server, HTTP server (Axum).
