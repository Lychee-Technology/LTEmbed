# LTEmbed Design

## 1. Overview

**LTEmbed** is a Rust library for generating L2-normalized vector embeddings from BERT-family models. The main integration surface is [`ZeroVecEngine`](../src/engine.rs), which loads local assets, tokenizes input text, runs inference, applies pooling, and returns normalized `Vec<f32>` outputs.

The current repository is centered on direct Rust library usage. It also includes benchmark-oriented entry points, but the core product is the library API itself.

The v1 asset set in this repository targets **e5-small-v2**. With the bundled assets, embeddings are 384-dimensional.

Typical usage:

```rust
use ltembed::engine::ZeroVecEngine;
use ltembed::traits::pooling::MeanPooling;

let config_json = std::fs::read_to_string("assets/config.json")?;
let engine = ZeroVecEngine::new(
    "assets/model.safetensors",
    &config_json,
    "assets/tokenizer.json",
    Box::new(MeanPooling),
)?;

let embedding = engine.embed("query: Hello, world!")?;
assert_eq!(embedding.len(), 384);
```

## 2. Public API

### `ZeroVecEngine`

```rust
pub struct ZeroVecEngine { /* opaque */ }

impl ZeroVecEngine {
    pub fn new(
        safetensors_path: &str,
        config_json: &str,
        tokenizer_path: &str,
        pooling: Box<dyn Pooling>,
    ) -> Result<Self, LTEmbedError>;

    pub fn embed(&self, text: &str) -> Result<Vec<f32>, LTEmbedError>;

    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, LTEmbedError>;
}
```

Key details:

- `config_json` is the JSON contents, not a file path.
- `embed()` returns one L2-normalized embedding for one input string.
- `embed_batch()` returns one embedding per input, in the same order.
- Inputs longer than 512 tokens are rejected with `LTEmbedError::InputTooLong`.

### `LTEmbedError`

```rust
#[derive(Debug, thiserror::Error)]
pub enum LTEmbedError {
    ModelLoad(String),
    Tokenization(String),
    Inference(String),
    InputTooLong { tokens: usize, max: usize },
    Io(std::io::Error),
    Json(serde_json::Error),
}
```

The library returns typed errors instead of silently truncating input or hiding failures behind unstructured strings.

### `Pooling`

```rust
pub trait Pooling: Send + Sync {
    fn pool(
        &self,
        last_hidden_state: &[f32],
        seq_len: usize,
        hidden_size: usize,
        attention_mask: &[u32],
    ) -> Result<Vec<f32>, LTEmbedError>;
}

pub struct MeanPooling;
pub struct CLSPooling;
```

- `MeanPooling` averages non-padding token vectors.
- `CLSPooling` returns the first token representation.
- Callers can provide custom pooling implementations through the trait.

## 3. Architecture

### Layer 1: Public Engine

[`src/engine.rs`](../src/engine.rs) defines `ZeroVecEngine`, which owns:

- a `Bert` model backend
- an `HFTokenizer`
- a `Box<dyn Pooling>`

`ZeroVecEngine` orchestrates the full inference pipeline:

1. tokenize input text
2. run model inference
3. pool token-level hidden states
4. L2-normalize the output vector

### Layer 2: Tokenization

[`src/traits/tokenizer.rs`](../src/traits/tokenizer.rs) defines:

- `Tokenizer`
- `TokenizerOutput`
- `HFTokenizer`

`HFTokenizer` uses the Hugging Face `tokenizers` crate and explicitly rejects overlong inputs instead of truncating them.

### Layer 3: Model Backend

[`src/models/bert.rs`](../src/models/bert.rs) contains LTEmbed's from-scratch BERT inference backend built on:

- `safetensors`
- `memmap2`
- `matrixmultiply`

Model weights are memory-mapped from `model.safetensors`, so initialization avoids copying the full weight file into heap memory. The backend supports both single-input forward passes and padded batched forward passes.

### Layer 4: Pooling and Normalization

[`src/traits/pooling.rs`](../src/traits/pooling.rs) converts token-level hidden states into a single embedding vector. [`src/utils.rs`](../src/utils.rs) provides in-place L2 normalization so outputs are ready for cosine-similarity workloads.

## 4. Performance Characteristics

### Initialization

`ZeroVecEngine::new()`:

1. parses `config.json`
2. memory-maps `model.safetensors`
3. loads `tokenizer.json`
4. prepares the model for inference

The first inference call may still incur OS page faults as model pages are touched for the first time. If first-call latency matters, call `engine.embed("warmup")` immediately after initialization.

### Batch Inference

`embed_batch()` is not a simple loop over `embed()`. It tokenizes the full input slice, pads to the maximum sequence length in the batch, runs a batched forward pass, then pools and normalizes each result.

### ARM64 Focus

The repository is tuned for ARM64-class deployment targets and developer machines, but the Rust API itself is transport-agnostic. Library consumers can embed LTEmbed in local tools, services, or benchmark harnesses without changing the core engine interface.

## 5. Repository Structure

```text
LTEmbed/
├── README.md
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── engine.rs
│   ├── error.rs
│   ├── utils.rs
│   ├── benchmarking.rs
│   ├── gemm.rs
│   ├── traits/
│   │   ├── tokenizer.rs
│   │   └── pooling.rs
│   ├── models/
│   │   └── bert.rs
│   └── bin/
│       └── benchmark_ltembed.rs
├── examples/
│   ├── api_usage.rs
│   └── benchmark_candle.rs
├── tests/
│   ├── integration_tests.rs
│   └── fixtures/
│       └── test_fixtures.json
├── benches/
│   └── inference.rs
├── scripts/
│   └── generate_fixtures.py
└── assets/
    ├── config.json
    ├── tokenizer.json
    └── model.safetensors
```

Notes:

- `examples/api_usage.rs` is the clearest runnable Rust API example.
- `src/bin/benchmark_ltembed.rs` is the repository's benchmark-oriented executable.
- The root [`README.md`](../README.md) is the primary quick-start document for Rust API consumers.

## 6. Verification References

Common verification entry points:

```bash
cargo check --all-targets
cargo test
cargo run --example api_usage
cargo run --bin benchmark_ltembed -- --mode correctness --model-dir assets
```

For API consumers, start with the root [`README.md`](../README.md). For test intent and tiers, see [`docs/integ-test.md`](./integ-test.md).
