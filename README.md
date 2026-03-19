# LTEmbed

LTEmbed is a Rust embedding engine for BERT-family models. The primary entry point is [`ZeroVecEngine`](./src/engine.rs), which loads a local `safetensors` model, tokenizes text with Hugging Face `tokenizers`, runs inference, applies pooling, and returns L2-normalized embedding vectors.

The repository also contains runnable examples and benchmark binaries, but the core integration surface is the Rust library API.

## Requirements

LTEmbed expects local model assets:

- `assets/config.json`
- `assets/tokenizer.json`
- `assets/model.safetensors`

The bundled `assets/` directory in this repository contains a working `e5-small-v2` layout. With those assets, embeddings are 384-dimensional.

## Quick Start

```rust
use ltembed::engine::ZeroVecEngine;
use ltembed::traits::pooling::MeanPooling;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config_json = std::fs::read_to_string("assets/config.json")?;

    let engine = ZeroVecEngine::new(
        "assets/model.safetensors",
        &config_json,
        "assets/tokenizer.json",
        Box::new(MeanPooling),
    )?;

    let single = engine.embed("query: Hello, world!")?;
    let batch = engine.embed_batch(&["query: alpha", "query: beta"])?;

    println!("single embedding dim: {}", single.len());
    println!("batch size: {}", batch.len());
    println!("first value: {:.6}", single[0]);

    Ok(())
}
```

## Rust API

### `ZeroVecEngine::new`

```rust
pub fn new(
    safetensors_path: &str,
    config_json: &str,
    tokenizer_path: &str,
    pooling: Box<dyn Pooling>,
) -> Result<Self, LTEmbedError>
```

Notes:

- `config_json` is the JSON contents, not a path. Read the file first and pass the string.
- `safetensors_path` and `tokenizer_path` are file paths.
- `pooling` can be `MeanPooling`, `CLSPooling`, or your own `Pooling` implementation.

### `ZeroVecEngine::embed`

```rust
pub fn embed(&self, text: &str) -> Result<Vec<f32>, LTEmbedError>
```

Behavior:

- Returns one embedding vector for one input string.
- Output vectors are L2-normalized before being returned.
- For the bundled `e5-small-v2` assets, the returned vector length is `384`.

### `ZeroVecEngine::embed_batch`

```rust
pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, LTEmbedError>
```

Behavior:

- Returns one embedding per input, in the same order as `texts`.
- `embed_batch(&[])` returns an empty `Vec`.
- Batch outputs are also L2-normalized.

## Pooling

LTEmbed exposes pooling through [`traits::pooling`](./src/traits/pooling.rs):

- `MeanPooling`: averages non-padding token vectors
- `CLSPooling`: returns the first token representation

If you need a different output policy, implement the `Pooling` trait and pass it to `ZeroVecEngine::new`.

## Error Semantics

The library uses [`LTEmbedError`](./src/error.rs) for initialization and inference failures:

- `ModelLoad`
- `Tokenization`
- `Inference`
- `InputTooLong { tokens, max }`
- `Io`
- `Json`

Important behavior:

- Inputs are not silently truncated.
- Tokenized inputs longer than 512 tokens return `LTEmbedError::InputTooLong`.

## Running This Repository

Build or test the crate:

```bash
cargo test
```

Run the direct Rust API example:

```bash
cargo run --example api_usage
```

Run the LTEmbed benchmark binary against local assets:

```bash
cargo run --bin benchmark_ltembed -- --mode correctness --model-dir assets
```

The repository also includes:

- [`examples/api_usage.rs`](./examples/api_usage.rs): minimal direct Rust API example
- [`examples/benchmark_candle.rs`](./examples/benchmark_candle.rs): Candle baseline benchmark/example
- [`tests/integration_tests.rs`](./tests/integration_tests.rs): integration coverage for model loading, error handling, normalization, and batch consistency
