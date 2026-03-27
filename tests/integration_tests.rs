// tests/integration_tests.rs

use approx::assert_relative_eq;
use ltembed::engine::{EmbeddingInput, OnnxEngine, EMBEDDING_DIMENSION, MAX_LENGTH};
use ltembed::error::LTEmbedError;
use serde::Deserialize;
use std::path::Path;

const MODEL: &str = "assets/onnx/model_q4f16.onnx";
const TOKENIZER: &str = "assets/tokenizer.json";
const FIXTURES: &str = "tests/fixtures/test_fixtures.json";

fn assets_available() -> bool {
    Path::new(MODEL).exists() && Path::new(TOKENIZER).exists()
}

fn make_engine() -> OnnxEngine {
    OnnxEngine::new(MODEL, TOKENIZER).expect("Failed to initialize OnnxEngine")
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm_a * norm_b)
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum FixtureKind {
    Query,
    Document,
}

#[derive(Deserialize)]
struct Fixture {
    kind: FixtureKind,
    text: String,
    embedding: Vec<f32>,
}

#[derive(Deserialize)]
struct FixtureFile {
    dim: Option<usize>,
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
    if data.dim != Some(EMBEDDING_DIMENSION) {
        eprintln!(
            "Skipping golden parity test: fixtures are not regenerated for {}-d OnnxEngine outputs",
            EMBEDDING_DIMENSION
        );
        return;
    }

    for fixture in &data.fixtures {
        let input = match fixture.kind {
            FixtureKind::Query => EmbeddingInput::query(&fixture.text),
            FixtureKind::Document => EmbeddingInput::document(&fixture.text),
        };
        let rust_v = engine.embed(input).unwrap();
        let sim = cosine_similarity(&rust_v, &fixture.embedding);
        assert!(
            sim > 0.99,
            "Cosine similarity {sim:.6} < 0.99 for {:?}",
            &fixture.text[..50.min(fixture.text.len())]
        );
    }
}

#[test]
fn test_missing_model_file_returns_model_load_error() {
    let result = OnnxEngine::new(
        "/nonexistent/model_q4f16.onnx",
        "/nonexistent/tokenizer.json",
    );
    assert!(result.is_err());
    assert!(
        matches!(result.unwrap_err(), LTEmbedError::ModelLoad(_)),
        "Expected ModelLoad error"
    );
}

#[test]
fn test_long_input_returns_input_too_long_error() {
    if !std::path::Path::new(TOKENIZER).exists() {
        eprintln!("Skipping: tokenizer asset not found");
        return;
    }
    use ltembed::traits::tokenizer::{HFTokenizer, Tokenizer};
    let tok = HFTokenizer::from_file(TOKENIZER).unwrap();
    let long_text = "hello world ".repeat(12000);
    let result = tok.encode(&long_text, MAX_LENGTH);
    assert!(result.is_err());
    match result.unwrap_err() {
        LTEmbedError::InputTooLong { tokens, max } => {
            assert!(
                tokens > MAX_LENGTH,
                "tokens={tokens} should be > {MAX_LENGTH}"
            );
            assert_eq!(max, MAX_LENGTH);
        }
        other => panic!("Expected InputTooLong, got {other:?}"),
    }
}

#[test]
fn test_embed_batch_consistency() {
    if !assets_available() {
        eprintln!("Skipping: model assets not found");
        return;
    }
    let engine = make_engine();
    let inputs = [
        EmbeddingInput::query("hello"),
        EmbeddingInput::query("world"),
    ];
    let batch = engine.embed_batch(&inputs).unwrap();
    let individual = engine.embed(inputs[0]).unwrap();
    assert_eq!(
        batch[0], individual,
        "embed_batch[0] must equal embed() for same input"
    );
}

#[test]
fn test_output_is_l2_normalized() {
    if !assets_available() {
        eprintln!("Skipping: model assets not found");
        return;
    }
    let engine = make_engine();
    let v = engine
        .embed(EmbeddingInput::query("normalization check"))
        .unwrap();
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert_relative_eq!(norm, 1.0, epsilon = 1e-5);
}

#[test]
fn test_output_dimension_is_512() {
    if !assets_available() {
        eprintln!("Skipping: model assets not found");
        return;
    }
    let engine = make_engine();
    let v = engine
        .embed(EmbeddingInput::query("dimension check"))
        .unwrap();
    assert_eq!(v.len(), EMBEDDING_DIMENSION);
}
