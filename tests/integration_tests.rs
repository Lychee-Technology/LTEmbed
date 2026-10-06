use approx::assert_relative_eq;
use ltembed::engine::{
    EmbeddingEngine, EmbeddingInput, EngineConfig, EMBEDDING_DIMENSION, MAX_LENGTH,
};
use ltembed::error::{LTEmbedError, ModelLoadError};
use ltembed::traits::tokenizer::{HFTokenizer, Tokenizer, TokenizerOutput};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

const FIXTURES: &str = "tests/fixtures/test_fixtures.json";
const TOKEN_IDS_FIXTURE: &str = "tests/fixtures/token_ids.json";
const TOKENIZER: &str = "assets/tokenizer.json";
const TEST_BUNDLE_ENV: &str = "LTEMBED_TEST_BUNDLE_DIR";

fn bundle_dir() -> Option<PathBuf> {
    std::env::var_os(TEST_BUNDLE_ENV).map(PathBuf::from)
}

fn bundle_available() -> bool {
    bundle_dir()
        .map(|dir| dir.join("model.gguf").exists() && dir.join("tokenizer.json").exists())
        .unwrap_or(false)
}

fn make_engine() -> EmbeddingEngine {
    let bundle_dir = bundle_dir().expect("LTEMBED_TEST_BUNDLE_DIR must be set for bundle tests");
    EmbeddingEngine::from_gguf_bundle_dir(
        &bundle_dir,
        EngineConfig {
            output_dimension: EMBEDDING_DIMENSION,
            l2_normalize: true,
        },
    )
    .expect("Failed to initialize EmbeddingEngine from bundle")
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

/// `tests/fixtures/token_ids.json`, written by `scripts/generate_token_ids.py`.
#[derive(Deserialize)]
struct TokenIdFixtureFile {
    tokenizer: TokenizerSource,
    tokenizers_version: String,
    padding: BatchPadding,
    cases: Vec<TokenIdCase>,
}

/// The `tokenizer.json` the fixture was generated from.
#[derive(Deserialize)]
struct TokenizerSource {
    repo: String,
    revision: String,
    sha256: String,
}

/// Values of right-padding tokens. The generator rejects any other padding direction.
#[derive(Deserialize)]
struct BatchPadding {
    pad_id: u32,
    pad_type_id: u32,
}

#[derive(Deserialize)]
struct TokenIdCase {
    name: String,
    input: TokenIdInput,
    single: TokenIds,
    batch: PaddedTokenIds,
}

/// A literal text, or a long text described by a repeated unit.
#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum TokenIdInput {
    Text(String),
    Repeat {
        prefix: String,
        unit: String,
        count: usize,
        suffix: String,
    },
}

impl TokenIdInput {
    fn build(&self) -> String {
        match self {
            TokenIdInput::Text(text) => text.clone(),
            TokenIdInput::Repeat {
                prefix,
                unit,
                count,
                suffix,
            } => format!("{prefix}{}{suffix}", unit.repeat(*count)),
        }
    }
}

#[derive(Deserialize)]
struct TokenIds {
    input_ids: Vec<u32>,
    attention_mask: Vec<u32>,
    type_ids: Vec<u32>,
}

/// A batch row without its trailing padding, which is `padding` tokens long.
#[derive(Deserialize)]
struct PaddedTokenIds {
    #[serde(flatten)]
    unpadded: TokenIds,
    padding: usize,
}

impl PaddedTokenIds {
    fn padded(&self, pad: &BatchPadding) -> TokenIds {
        let pad_with = |values: &[u32], value: u32| -> Vec<u32> {
            values
                .iter()
                .copied()
                .chain(std::iter::repeat_n(value, self.padding))
                .collect()
        };
        TokenIds {
            input_ids: pad_with(&self.unpadded.input_ids, pad.pad_id),
            attention_mask: pad_with(&self.unpadded.attention_mask, 0),
            type_ids: pad_with(&self.unpadded.type_ids, pad.pad_type_id),
        }
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Describes how `actual` differs from `expected` without printing sequences that can be
/// thousands of tokens long.
fn sequence_mismatch(field: &str, actual: &[u32], expected: &[u32]) -> Option<String> {
    let lengths = format!("lengths {} vs {}", actual.len(), expected.len());
    match actual.iter().zip(expected).position(|(a, e)| a != e) {
        Some(i) => Some(format!(
            "{field}[{i}] is {}, expected {} ({lengths})",
            actual[i], expected[i]
        )),
        None => (actual.len() != expected.len()).then(|| format!("{field}: {lengths}")),
    }
}

fn token_id_mismatches(actual: &TokenizerOutput, expected: &TokenIds) -> Vec<String> {
    [
        ("input_ids", &actual.input_ids, &expected.input_ids),
        (
            "attention_mask",
            &actual.attention_mask,
            &expected.attention_mask,
        ),
        ("type_ids", &actual.token_type_ids, &expected.type_ids),
    ]
    .into_iter()
    .filter_map(|(field, actual, expected)| sequence_mismatch(field, actual, expected))
    .collect()
}

fn unique_temp_dir() -> PathBuf {
    static UNIQUE_TEMP_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let counter = UNIQUE_TEMP_DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("ltembed-bundle-tests-{nanos}-{counter}"))
}

fn write_build_info(dir: &Path, body: &str) {
    fs::write(dir.join("build-info.json"), body).unwrap();
}

fn write_tokenizer(dir: &Path) {
    fs::copy(TOKENIZER, dir.join("tokenizer.json")).unwrap();
}

fn write_model_stub(dir: &Path) {
    fs::write(dir.join("model.gguf"), "stub").unwrap();
}

fn valid_build_info_json() -> &'static str {
    r#"{
  "target_id": "jinaai/jina-embeddings-v5-text-nano-retrieval",
  "model_metadata": {
    "model_format": "gguf",
    "pooling": "last_token",
    "input_kind": "retrieval",
    "query_prefix": "Query: ",
    "document_prefix": "Document: ",
    "raw_embedding_dimension": 768,
    "output_embedding_dimension": 768,
    "max_length": 8192
  }
}"#
}

#[test]
fn test_golden_parity_cosine_similarity() {
    if !bundle_available() {
        eprintln!("Skipping golden parity test: {TEST_BUNDLE_ENV} not set");
        return;
    }
    let engine = make_engine();
    let fixture_str = std::fs::read_to_string(FIXTURES)
        .expect("tests/fixtures/test_fixtures.json not found — run scripts/generate_fixtures.py");
    let data: FixtureFile = serde_json::from_str(&fixture_str).unwrap();
    if data.dim != Some(EMBEDDING_DIMENSION) {
        eprintln!(
            "Skipping golden parity test: fixtures are not regenerated for {}-d engine outputs",
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

/// `HFTokenizer` must produce the same ids as the Python `tokenizers` behind the golden
/// fixtures. The fixture's inputs exercise the model's Oniguruma `Split` regex (a 1M-char
/// whitespace run that fancy-regex splits differently) and its added-token matcher.
#[test]
fn test_token_ids_match_python_tokenizers() {
    // Gated on the env var alone, not bundle_available(): only tokenizer.json is needed, and a
    // bundle dir without it should fail below rather than skip.
    let Some(bundle_dir) = bundle_dir() else {
        eprintln!("Skipping token-id parity test: {TEST_BUNDLE_ENV} not set");
        return;
    };
    let fixture_str = fs::read_to_string(TOKEN_IDS_FIXTURE)
        .expect("tests/fixtures/token_ids.json not found — run scripts/generate_token_ids.py");
    let fixture: TokenIdFixtureFile = serde_json::from_str(&fixture_str).unwrap();

    let tokenizer_path = bundle_dir.join("tokenizer.json");
    let tokenizer_json = fs::read(&tokenizer_path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", tokenizer_path.display()));
    let source = &fixture.tokenizer;
    assert_eq!(
        sha256_hex(&tokenizer_json),
        source.sha256,
        "{} is not the tokenizer.json that {TOKEN_IDS_FIXTURE} was generated from ({} at \
         revision {}). If the upstream tokenizer changed, regenerate the fixture: set REVISION \
         in scripts/generate_token_ids.py and HF_REVISION in .github/workflows/ci.yml to the \
         new revision, then run `python3 scripts/generate_token_ids.py`.",
        tokenizer_path.display(),
        source.repo,
        source.revision,
    );

    let tokenizer = HFTokenizer::from_file(&tokenizer_path.to_string_lossy()).unwrap();
    let texts: Vec<String> = fixture
        .cases
        .iter()
        .map(|case| case.input.build())
        .collect();
    let batch = tokenizer
        .encode_batch(&texts, MAX_LENGTH)
        .unwrap_or_else(|e| panic!("encode_batch failed: {e}"));
    assert_eq!(batch.len(), fixture.cases.len());

    let mut mismatches = Vec::new();
    for ((case, text), batch_row) in fixture.cases.iter().zip(&texts).zip(&batch) {
        let single = tokenizer
            .encode(text, MAX_LENGTH)
            .unwrap_or_else(|e| panic!("{}: encode failed: {e}", case.name));
        for mismatch in token_id_mismatches(&single, &case.single) {
            mismatches.push(format!("{} (encode): {mismatch}", case.name));
        }
        for mismatch in token_id_mismatches(batch_row, &case.batch.padded(&fixture.padding)) {
            mismatches.push(format!("{} (encode_batch): {mismatch}", case.name));
        }
    }
    assert!(
        mismatches.is_empty(),
        "HFTokenizer output differs from Python tokenizers {}:\n  {}",
        fixture.tokenizers_version,
        mismatches.join("\n  ")
    );
}

#[test]
fn test_missing_model_file_returns_model_load_error() {
    let temp_dir = unique_temp_dir();
    fs::create_dir_all(&temp_dir).unwrap();
    write_tokenizer(&temp_dir);
    write_build_info(&temp_dir, valid_build_info_json());

    let result = EmbeddingEngine::from_gguf_bundle_dir(&temp_dir, EngineConfig::default());
    assert!(matches!(
        result.unwrap_err(),
        LTEmbedError::ModelLoad(ModelLoadError::MissingFile { .. })
    ));

    fs::remove_dir_all(temp_dir).unwrap();
}

#[test]
fn test_missing_tokenizer_returns_model_load_error() {
    let temp_dir = unique_temp_dir();
    fs::create_dir_all(&temp_dir).unwrap();
    write_model_stub(&temp_dir);
    write_build_info(&temp_dir, valid_build_info_json());

    let result = EmbeddingEngine::from_gguf_bundle_dir(&temp_dir, EngineConfig::default());
    assert!(matches!(
        result.unwrap_err(),
        LTEmbedError::ModelLoad(ModelLoadError::MissingFile { .. })
    ));

    fs::remove_dir_all(temp_dir).unwrap();
}

#[test]
fn test_missing_build_info_returns_model_load_error() {
    let temp_dir = unique_temp_dir();
    fs::create_dir_all(&temp_dir).unwrap();
    write_tokenizer(&temp_dir);
    write_model_stub(&temp_dir);

    let result = EmbeddingEngine::from_gguf_bundle_dir(&temp_dir, EngineConfig::default());
    assert!(matches!(
        result.unwrap_err(),
        LTEmbedError::ModelLoad(ModelLoadError::MissingFile { .. })
    ));

    fs::remove_dir_all(temp_dir).unwrap();
}

#[test]
fn test_malformed_build_info_returns_model_load_error() {
    let temp_dir = unique_temp_dir();
    fs::create_dir_all(&temp_dir).unwrap();
    write_tokenizer(&temp_dir);
    write_model_stub(&temp_dir);
    write_build_info(&temp_dir, "{not-json");

    let result = EmbeddingEngine::from_gguf_bundle_dir(&temp_dir, EngineConfig::default());
    assert!(matches!(
        result.unwrap_err(),
        LTEmbedError::ModelLoad(ModelLoadError::Metadata(_))
    ));

    fs::remove_dir_all(temp_dir).unwrap();
}

#[test]
fn test_invalid_input_kind_returns_model_load_error() {
    let temp_dir = unique_temp_dir();
    fs::create_dir_all(&temp_dir).unwrap();
    write_tokenizer(&temp_dir);
    write_model_stub(&temp_dir);
    write_build_info(
        &temp_dir,
        r#"{
  "target_id": "bad-model",
  "model_metadata": {
    "model_format": "gguf",
    "pooling": "last_token",
    "input_kind": "classification",
    "query_prefix": "Query: ",
    "document_prefix": "Document: ",
    "raw_embedding_dimension": 768,
    "output_embedding_dimension": 768,
    "max_length": 8192
  }
}"#,
    );

    let result = EmbeddingEngine::from_gguf_bundle_dir(&temp_dir, EngineConfig::default());
    assert!(matches!(
        result.unwrap_err(),
        LTEmbedError::ModelLoad(ModelLoadError::UnsupportedInputKind { .. })
    ));
    fs::remove_dir_all(temp_dir).unwrap();
}

#[test]
fn test_invalid_pooling_returns_model_load_error() {
    let temp_dir = unique_temp_dir();
    fs::create_dir_all(&temp_dir).unwrap();
    write_tokenizer(&temp_dir);
    write_model_stub(&temp_dir);
    write_build_info(
        &temp_dir,
        r#"{
  "target_id": "bad-model",
  "model_metadata": {
    "model_format": "gguf",
    "pooling": "mean",
    "input_kind": "retrieval",
    "query_prefix": "Query: ",
    "document_prefix": "Document: ",
    "raw_embedding_dimension": 768,
    "output_embedding_dimension": 768,
    "max_length": 8192
  }
}"#,
    );

    let result = EmbeddingEngine::from_gguf_bundle_dir(&temp_dir, EngineConfig::default());
    assert!(matches!(
        result.unwrap_err(),
        LTEmbedError::ModelLoad(ModelLoadError::UnsupportedPooling { .. })
    ));

    fs::remove_dir_all(temp_dir).unwrap();
}

#[test]
fn test_output_dimension_larger_than_raw_returns_model_load_error() {
    let temp_dir = unique_temp_dir();
    fs::create_dir_all(&temp_dir).unwrap();
    write_tokenizer(&temp_dir);
    write_model_stub(&temp_dir);
    write_build_info(&temp_dir, valid_build_info_json());

    let result = EmbeddingEngine::from_gguf_bundle_dir(
        &temp_dir,
        EngineConfig {
            output_dimension: 769,
            l2_normalize: true,
        },
    );
    assert!(matches!(
        result.unwrap_err(),
        LTEmbedError::ModelLoad(ModelLoadError::Config(_))
    ));

    fs::remove_dir_all(temp_dir).unwrap();
}

#[test]
fn test_long_input_returns_input_too_long_error() {
    if !std::path::Path::new(TOKENIZER).exists() {
        eprintln!("Skipping: tokenizer asset not found");
        return;
    }
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
    if !bundle_available() {
        eprintln!("Skipping: {TEST_BUNDLE_ENV} not set");
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
    if !bundle_available() {
        eprintln!("Skipping: {TEST_BUNDLE_ENV} not set");
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
    if !bundle_available() {
        eprintln!("Skipping: {TEST_BUNDLE_ENV} not set");
        return;
    }
    let engine = make_engine();
    let v = engine
        .embed(EmbeddingInput::query("dimension check"))
        .unwrap();
    assert_eq!(v.len(), EMBEDDING_DIMENSION);
}
