// tests/integration_tests.rs
//
// Run with: cargo test --test integration_tests
//
// Requires model assets for most tests (see below).
// test_missing_model_file_returns_model_load_error always runs — no assets needed.

use approx::assert_relative_eq;
use ltembed::engine::ZeroVecEngine;
use ltembed::error::LTEmbedError;
use ltembed::traits::engine::EmbeddingEngine;
use ltembed::traits::pooling::MeanPooling;
use serde::Deserialize;
use std::path::Path;

const SAFETENSORS: &str = "assets/model.safetensors";
const CONFIG: &str = "assets/config.json";
const TOKENIZER: &str = "assets/tokenizer.json";
const FIXTURES: &str = "tests/fixtures/test_fixtures.json";

const DUMMY_CONFIG: &str = r#"{
    "hidden_size": 384, "num_hidden_layers": 12, "num_attention_heads": 12,
    "intermediate_size": 1536, "max_position_embeddings": 512,
    "vocab_size": 30522, "type_vocab_size": 2, "hidden_act": "gelu",
    "layer_norm_eps": 1e-12,
    "hidden_dropout_prob": 0.1,
    "initializer_range": 0.02,
    "pad_token_id": 0,
    "classifier_dropout": null
}"#;

fn assets_available() -> bool {
    Path::new(SAFETENSORS).exists() && Path::new(CONFIG).exists() && Path::new(TOKENIZER).exists()
}

fn make_engine() -> ZeroVecEngine {
    let config_str = std::fs::read_to_string(CONFIG).unwrap();
    ZeroVecEngine::new(SAFETENSORS, &config_str, TOKENIZER, Box::new(MeanPooling))
        .expect("Failed to initialize ZeroVecEngine")
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm_a * norm_b)
}

// --- Scenario A: Golden Path Parity ---

#[derive(Deserialize)]
struct Fixture {
    input: String,
    embedding: Vec<f32>,
}

#[derive(Deserialize)]
struct FixtureFile {
    fixtures: Vec<Fixture>,
}

#[test]
fn test_golden_parity_cosine_similarity() {
    if !assets_available() {
        eprintln!("Skipping golden parity test: model assets not found");
        return;
    }
    let engine = make_engine();
    let fixture_str = std::fs::read_to_string(FIXTURES)
        .expect("tests/fixtures/test_fixtures.json not found — run scripts/generate_fixtures.py");
    let data: FixtureFile = serde_json::from_str(&fixture_str).unwrap();

    for fixture in &data.fixtures {
        let rust_v = engine.embed(&fixture.input).unwrap();
        let sim = cosine_similarity(&rust_v, &fixture.embedding);
        assert!(
            sim > 0.999,
            "Cosine similarity {sim:.6} < 0.999 for {:?}",
            &fixture.input[..50.min(fixture.input.len())]
        );
    }
}

// --- Scenario C1: Missing Model Files ---

#[test]
fn test_missing_model_file_returns_model_load_error() {
    let result = ZeroVecEngine::new(
        "/nonexistent/model.safetensors",
        DUMMY_CONFIG,
        "/nonexistent/tokenizer.json",
        Box::new(MeanPooling),
    );
    assert!(result.is_err());
    assert!(
        matches!(result.unwrap_err(), LTEmbedError::ModelLoad(_)),
        "Expected ModelLoad error"
    );
}

/// --- Scenario C2: Context Length Overflow ---
// LTEmbed is a library — overlong input must return an explicit error, not be silently
// truncated or cause a panic. The caller decides how to handle the error.

#[test]
fn test_long_input_returns_input_too_long_error() {
    // Tier 1: does not require model assets — the error is raised in the tokenizer
    // before any model inference occurs.
    if !std::path::Path::new(TOKENIZER).exists() {
        eprintln!("Skipping: tokenizer asset not found");
        return;
    }
    // Engine init requires model.safetensors; if absent, use a tokenizer-only test
    // via HFTokenizer directly.
    use ltembed::traits::tokenizer::{HFTokenizer, Tokenizer};
    let tok = HFTokenizer::from_file(TOKENIZER).unwrap();
    let long_text = "hello world ".repeat(5000); // encodes to >> 512 tokens
    let result = tok.encode(&long_text, 512);
    assert!(result.is_err());
    match result.unwrap_err() {
        LTEmbedError::InputTooLong { tokens, max } => {
            assert!(tokens > 512, "tokens={tokens} should be > 512");
            assert_eq!(max, 512);
        }
        other => panic!("Expected InputTooLong, got {other:?}"),
    }
}

// --- Scenario C3: Malformed config.json ---

#[test]
fn test_malformed_config_returns_model_load_error() {
    let result = ZeroVecEngine::new(
        "/nonexistent/model.safetensors",
        "{ this is not valid json !!!",
        "/nonexistent/tokenizer.json",
        Box::new(MeanPooling),
    );
    assert!(result.is_err());
    assert!(
        matches!(result.unwrap_err(), LTEmbedError::ModelLoad(_)),
        "Expected ModelLoad error for malformed config"
    );
}

// --- Scenario B: Output properties ---

#[test]
fn test_embed_batch_consistency() {
    if !assets_available() {
        eprintln!("Skipping: model assets not found");
        return;
    }
    let engine = make_engine();
    let batch = engine
        .embed_batch(&["query: hello", "query: world"])
        .unwrap();
    let individual = engine.embed("query: hello").unwrap();
    assert_eq!(
        batch[0], individual,
        "embed_batch[0] must equal embed() for same input"
    );
}

// --- Output shape and L2 normalization ---

#[test]
fn test_output_is_l2_normalized() {
    if !assets_available() {
        eprintln!("Skipping: model assets not found");
        return;
    }
    let engine = make_engine();
    let v = engine.embed("query: normalization check").unwrap();
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert_relative_eq!(norm, 1.0, epsilon = 1e-5);
}

#[test]
fn test_output_dimension_is_384() {
    if !assets_available() {
        eprintln!("Skipping: model assets not found");
        return;
    }
    let engine = make_engine();
    let v = engine.embed("query: dimension check").unwrap();
    assert_eq!(v.len(), 384);
}

// ── Scenario D: LlamaCppEngine accuracy parity ────────────────────────────────

#[cfg(feature = "ggml-backend")]
#[test]
fn test_llamacpp_engine_parity_with_zerovec() {
    use ltembed::engine_llama::LlamaCppEngine;

    const GGUF: &str = "assets/model.gguf";
    if !assets_available() || !Path::new(GGUF).exists() {
        eprintln!("Skipping: assets/model.safetensors or assets/model.gguf not found");
        eprintln!("  Run: python scripts/convert_to_gguf.py");
        return;
    }

    let zerovec = make_engine();
    let llamacpp = LlamaCppEngine::new(Path::new(GGUF), 1).expect("Failed to load LlamaCppEngine");

    let texts = [
        "query: Hello, world!",
        "query: What is the capital of France?",
        "passage: The quick brown fox jumps over the lazy dog.",
    ];

    for text in &texts {
        let zv = zerovec.embed(text).unwrap();
        let lc = llamacpp.embed(text).unwrap();

        assert_eq!(zv.len(), lc.len(), "Dimension mismatch for {text:?}");

        // Cosine similarity of two L2-normalized vectors = dot product
        let cosine: f32 = zv.iter().zip(lc.iter()).map(|(a, b)| a * b).sum();
        assert!(
            cosine >= 0.999,
            "Parity failure for {text:?}: cosine similarity = {cosine:.6} (expected >= 0.999)"
        );
    }
}
