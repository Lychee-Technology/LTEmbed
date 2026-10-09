use approx::assert_relative_eq;
use ltembed::engine::{
    EmbeddingEngine, EmbeddingInput, EngineConfig, EMBEDDING_DIMENSION, MAX_LENGTH,
};
use ltembed::error::{LTEmbedError, ModelLoadError};
use ltembed::traits::tokenizer::{HFTokenizer, Tokenizer, TokenizerOutput};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::env::VarError;
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

const FIXTURES: &str = "tests/fixtures/test_fixtures.json";
const TOKEN_IDS_FIXTURE: &str = "tests/fixtures/token_ids.json";
const TOKENIZER: &str = "assets/tokenizer.json";
const TEST_BUNDLE_ENV: &str = "LTEMBED_TEST_BUNDLE_DIR";
/// `1` turns every bundle-gated skip into a failure. CI sets it so that losing
/// `LTEMBED_TEST_BUNDLE_DIR` cannot leave the model-backed tests silently skipped.
const REQUIRE_TEST_BUNDLE_ENV: &str = "LTEMBED_REQUIRE_TEST_BUNDLE";
/// Bundle files the engine tests need. A missing `build-info.json` is not a skip: the engine
/// reports it as a load error and the test fails.
const ENGINE_BUNDLE_FILES: &[&str] = &["model.gguf", "tokenizer.json"];

/// The test bundle directory if it holds `files`. Otherwise the test should return early
/// (`None`), unless `LTEMBED_REQUIRE_TEST_BUNDLE=1`, which makes this panic instead.
/// Bundle-gated tests go through here, never through `LTEMBED_TEST_BUNDLE_DIR` directly.
fn test_bundle(files: &[&str]) -> Option<PathBuf> {
    bundle_or_skip(
        std::env::var_os(TEST_BUNDLE_ENV).map(PathBuf::from),
        test_bundle_required(),
        files,
    )
}

fn test_bundle_required() -> bool {
    match std::env::var(REQUIRE_TEST_BUNDLE_ENV).as_deref() {
        Err(VarError::NotPresent) | Ok("" | "0") => false,
        Ok("1") => true,
        // Fail closed: a typo such as `true` must not quietly turn the guard off.
        Ok(value) => panic!("{REQUIRE_TEST_BUNDLE_ENV} must be 0 or 1, got {value:?}"),
        Err(e) => panic!("{REQUIRE_TEST_BUNDLE_ENV}: {e}"),
    }
}

/// `test_bundle` with the environment passed in, so the tests below can check the decision.
fn bundle_or_skip(dir: Option<PathBuf>, required: bool, files: &[&str]) -> Option<PathBuf> {
    let reason = match dir {
        None => format!("{TEST_BUNDLE_ENV} is not set"),
        Some(dir) => {
            let missing: Vec<&str> = files
                .iter()
                .copied()
                .filter(|file| !dir.join(file).exists())
                .collect();
            if missing.is_empty() {
                return Some(dir);
            }
            format!("{} has no {}", dir.display(), missing.join(" or "))
        }
    };
    if required {
        panic!(
            "{REQUIRE_TEST_BUNDLE_ENV}=1 requires the test bundle, but {reason} \
             (see docs/testing.md)"
        );
    }
    eprintln!("Skipping: {reason}");
    None
}

fn make_engine(bundle_dir: &Path) -> EmbeddingEngine {
    EmbeddingEngine::from_gguf_bundle_dir(
        bundle_dir,
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
    dim: usize,
    fixtures: Vec<Fixture>,
}

/// `tests/fixtures/test_fixtures.json`, checked by `parse_golden_fixtures`.
fn golden_fixtures() -> FixtureFile {
    let fixture_str = fs::read_to_string(FIXTURES)
        .expect("tests/fixtures/test_fixtures.json not found — run scripts/generate_fixtures.py");
    parse_golden_fixtures(&fixture_str)
}

/// `golden_fixtures` with the file's contents passed in, so the tests below can check that a
/// bad fixture fails. Panics if `json` does not parse, or on any problem
/// `golden_fixture_problem` reports.
fn parse_golden_fixtures(json: &str) -> FixtureFile {
    let data: FixtureFile = serde_json::from_str(json).unwrap_or_else(|e| {
        panic!(
            "{FIXTURES} does not parse: {e}. Regenerate it with scripts/generate_fixtures.py \
             (see docs/testing.md)."
        )
    });
    if let Some(problem) = golden_fixture_problem(&data) {
        panic!(
            "{FIXTURES} {problem}. It must hold the PyTorch reference at the engine tests' \
             output dimension, so changing EMBEDDING_DIMENSION means regenerating it: set \
             OUTPUT_DIM in scripts/generate_fixtures.py to {EMBEDDING_DIMENSION} and run the \
             script (see docs/testing.md)."
        );
    }
    data
}

/// Why `data` cannot be compared with the `EMBEDDING_DIMENSION`-d vectors the engine tests
/// produce, or `None` if it can. A stale fixture is a problem in the repository, not a missing
/// local resource, so it fails every run instead of skipping the way a missing bundle does.
fn golden_fixture_problem(data: &FixtureFile) -> Option<String> {
    if data.dim != EMBEDDING_DIMENSION {
        return Some(format!(
            "records dim {}, but EMBEDDING_DIMENSION is {EMBEDDING_DIMENSION}",
            data.dim
        ));
    }
    if data.fixtures.is_empty() {
        return Some("has no fixtures".to_string());
    }
    data.fixtures
        .iter()
        .enumerate()
        .find(|(_, fixture)| fixture.embedding.len() != data.dim)
        .map(|(i, fixture)| {
            format!(
                "records dim {}, but fixture {i} has {} values",
                data.dim,
                fixture.embedding.len()
            )
        })
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
    std::env::temp_dir().join(unique_dir_name())
}

/// A directory name that no other call returns.
fn unique_dir_name() -> String {
    static UNIQUE_TEMP_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let counter = UNIQUE_TEMP_DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("ltembed-bundle-tests-{nanos}-{counter}")
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

/// Symlinks `files` from the test bundle into `dir`.
fn link_bundle_files(bundle_dir: &Path, dir: &Path, files: &[&str]) {
    for file in files {
        std::os::unix::fs::symlink(
            fs::canonicalize(bundle_dir.join(file)).unwrap(),
            dir.join(file),
        )
        .unwrap();
    }
}

/// A unique directory for large files, removed on drop even when the test panics. It is under
/// Cargo's `CARGO_TARGET_TMPDIR` (`target/tmp/`), not `std::env::temp_dir()` like
/// `unique_temp_dir`, because `/tmp` is often a small tmpfs.
struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Self {
        let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(unique_dir_name());
        fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// GGUF metadata value types (`enum gguf_type` in llama.cpp's `gguf.h`).
const GGUF_TYPE_UINT32: u32 = 4;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_ARRAY: u32 = 9;
/// Tensor data alignment of a GGUF without `general.alignment`.
const GGUF_DEFAULT_ALIGNMENT: u64 = 32;

/// Size in bytes of a fixed-size GGUF value type; `None` for strings and arrays.
fn gguf_fixed_size(value_type: u32) -> Option<u64> {
    match value_type {
        0 | 1 | 7 => Some(1), // uint8, int8, bool
        2 | 3 => Some(2),     // uint16, int16
        4..=6 => Some(4),     // uint32, int32, float32
        10..=12 => Some(8),   // uint64, int64, float64
        _ => None,
    }
}

/// Reads GGUF fields, which are little-endian.
struct GgufReader(BufReader<File>);

impl GgufReader {
    fn bytes<const N: usize>(&mut self) -> [u8; N] {
        let mut buf = [0; N];
        self.0.read_exact(&mut buf).unwrap();
        buf
    }

    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.bytes())
    }

    fn u64(&mut self) -> u64 {
        u64::from_le_bytes(self.bytes())
    }

    fn string(&mut self) -> String {
        let mut buf = vec![0; usize::try_from(self.u64()).unwrap()];
        self.0.read_exact(&mut buf).unwrap();
        String::from_utf8(buf).unwrap()
    }

    fn skip(&mut self, len: u64) {
        self.0.seek_relative(i64::try_from(len).unwrap()).unwrap();
    }

    fn skip_value(&mut self, value_type: u32) {
        if let Some(size) = gguf_fixed_size(value_type) {
            return self.skip(size);
        }
        match value_type {
            GGUF_TYPE_STRING => {
                let len = self.u64();
                self.skip(len);
            }
            GGUF_TYPE_ARRAY => {
                let element_type = self.u32();
                let len = self.u64();
                match gguf_fixed_size(element_type) {
                    Some(size) => self.skip(len * size),
                    None => (0..len).for_each(|_| self.skip_value(element_type)),
                }
            }
            _ => panic!("unknown GGUF value type {value_type}"),
        }
    }

    fn position(&mut self) -> u64 {
        self.0.stream_position().unwrap()
    }
}

/// A GGUF v3 file: a 24-byte header (magic, version, tensor count, key-value count), the
/// key-value pairs, the tensor infos, zero padding to `alignment`, then the tensor data. Holds
/// where those sections end and the metadata needed to add a key.
struct GgufFile {
    path: PathBuf,
    /// `general.architecture`, the prefix of the model's hyperparameter keys.
    architecture: String,
    keys: Vec<String>,
    kv_end: u64,
    tensor_infos_end: u64,
    alignment: u64,
}

impl GgufFile {
    fn read(path: &Path) -> Self {
        let mut reader = GgufReader(BufReader::new(File::open(path).unwrap()));
        assert_eq!(&reader.bytes(), b"GGUF", "{} is not a GGUF", path.display());
        let version = reader.u32();
        assert_eq!(version, 3, "{} is GGUF v{version}, not v3", path.display());
        let n_tensors = reader.u64();
        let n_kv = reader.u64();

        let mut architecture = None;
        let mut alignment = GGUF_DEFAULT_ALIGNMENT;
        let mut keys = Vec::new();
        for _ in 0..n_kv {
            let key = reader.string();
            let value_type = reader.u32();
            match (key.as_str(), value_type) {
                ("general.architecture", GGUF_TYPE_STRING) => {
                    architecture = Some(reader.string());
                }
                ("general.alignment", GGUF_TYPE_UINT32) => alignment = reader.u32().into(),
                _ => reader.skip_value(value_type),
            }
            keys.push(key);
        }
        let kv_end = reader.position();
        for _ in 0..n_tensors {
            reader.skip_value(GGUF_TYPE_STRING); // name
            let n_dims = reader.u32();
            reader.skip(8 * u64::from(n_dims) + 4 + 8); // dims, ggml type, data offset
        }
        Self {
            path: path.to_path_buf(),
            architecture: architecture
                .unwrap_or_else(|| panic!("{} has no general.architecture", path.display())),
            keys,
            kv_end,
            tensor_infos_end: reader.position(),
            alignment,
        }
    }

    /// Copies the file to `dst` with `key` appended to its metadata as a `uint32`. Tensor
    /// offsets are relative to the start of the tensor data, so only the key-value count and
    /// the padding before the data change.
    fn copy_with_u32_key(&self, dst: &Path, key: &str, value: u32) {
        assert!(
            !self.keys.iter().any(|k| k == key),
            "{} already has {key}",
            self.path.display()
        );
        let mut src = File::open(&self.path).unwrap();
        let mut header = vec![0; usize::try_from(self.tensor_infos_end).unwrap()];
        src.read_exact(&mut header).unwrap();
        let n_kv = u64::from_le_bytes(header[16..24].try_into().unwrap());
        header[16..24].copy_from_slice(&(n_kv + 1).to_le_bytes());

        let mut kv = Vec::new();
        kv.extend_from_slice(&u64::try_from(key.len()).unwrap().to_le_bytes());
        kv.extend_from_slice(key.as_bytes());
        kv.extend_from_slice(&GGUF_TYPE_UINT32.to_le_bytes());
        kv.extend_from_slice(&value.to_le_bytes());
        let kv_end = usize::try_from(self.kv_end).unwrap();
        header.splice(kv_end..kv_end, kv);
        let alignment = usize::try_from(self.alignment).unwrap();
        header.resize(header.len().next_multiple_of(alignment), 0);

        let mut out = BufWriter::new(File::create(dst).unwrap());
        out.write_all(&header).unwrap();
        let data_start = self.tensor_infos_end.next_multiple_of(self.alignment);
        src.seek(SeekFrom::Start(data_start)).unwrap();
        io::copy(&mut src, &mut out).unwrap();
        out.flush().unwrap();
    }
}

#[test]
fn test_golden_parity_cosine_similarity() {
    let Some(bundle_dir) = test_bundle(ENGINE_BUNDLE_FILES) else {
        return;
    };
    let data = golden_fixtures();
    let engine = make_engine(&bundle_dir);

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

/// Needs no bundle, so a fixture left stale by an output-dimension change fails every run,
/// not only the runs that can execute the parity test.
#[test]
fn test_golden_fixture_matches_engine_dimension() {
    golden_fixtures();
}

#[test]
fn test_golden_fixture_problems() {
    let fixture_file = |dim: usize, lengths: &[usize]| FixtureFile {
        dim,
        fixtures: lengths
            .iter()
            .map(|&len| Fixture {
                kind: FixtureKind::Query,
                text: "text".to_string(),
                embedding: vec![0.0; len],
            })
            .collect(),
    };
    let dim = EMBEDDING_DIMENSION;

    assert_eq!(
        golden_fixture_problem(&fixture_file(dim, &[dim, dim])),
        None
    );
    // The output dimension changed and the fixture was not regenerated.
    assert_eq!(
        golden_fixture_problem(&fixture_file(2 * dim, &[2 * dim])),
        Some(format!(
            "records dim {}, but EMBEDDING_DIMENSION is {dim}",
            2 * dim
        ))
    );
    assert_eq!(
        golden_fixture_problem(&fixture_file(dim, &[])),
        Some("has no fixtures".to_string())
    );
    assert_eq!(
        golden_fixture_problem(&fixture_file(dim, &[dim, dim / 2])),
        Some(format!(
            "records dim {dim}, but fixture 1 has {} values",
            dim / 2
        ))
    );
}

/// Detecting a stale fixture is not enough: loading it must fail the run. `dim` 0 can never
/// equal `EMBEDDING_DIMENSION`, and the file once shipped with it.
#[test]
#[should_panic(expected = "records dim 0, but EMBEDDING_DIMENSION is")]
fn test_stale_golden_fixture_fails() {
    parse_golden_fixtures(
        r#"{"dim": 0, "fixtures": [{"kind": "query", "text": "text", "embedding": []}]}"#,
    );
}

/// A fixture without `dim` fails too, rather than loading with its dimension unknown.
#[test]
#[should_panic(expected = "missing field `dim`")]
fn test_golden_fixture_without_dim_fails() {
    parse_golden_fixtures(r#"{"fixtures": [{"kind": "query", "text": "text", "embedding": []}]}"#);
}

/// `HFTokenizer` must produce the same ids as the Python `tokenizers` behind the golden
/// fixtures. The fixture's inputs exercise the model's Oniguruma `Split` regex (a 1M-char
/// whitespace run that fancy-regex splits differently) and its added-token matcher.
#[test]
fn test_token_ids_match_python_tokenizers() {
    // Needs only tokenizer.json, so a directory holding just that file is enough to run it.
    let Some(bundle_dir) = test_bundle(&["tokenizer.json"]) else {
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

/// The test bundle with only `raw_embedding_dimension` changed. The GGUF is 768 wide, so the
/// load itself must fail, not the first `embed`.
#[test]
fn test_gguf_width_mismatch_returns_model_load_error() {
    let Some(bundle_dir) = test_bundle(ENGINE_BUNDLE_FILES) else {
        return;
    };
    let temp_dir = unique_temp_dir();
    fs::create_dir_all(&temp_dir).unwrap();
    link_bundle_files(&bundle_dir, &temp_dir, ENGINE_BUNDLE_FILES);
    let build_info = fs::read_to_string(bundle_dir.join("build-info.json")).unwrap();
    let mut build_info: serde_json::Value = serde_json::from_str(&build_info).unwrap();
    build_info["model_metadata"]["raw_embedding_dimension"] = 1024.into();
    write_build_info(&temp_dir, &build_info.to_string());

    let result = EmbeddingEngine::from_gguf_bundle_dir(&temp_dir, EngineConfig::default());
    match result {
        Err(LTEmbedError::ModelLoad(ModelLoadError::Runtime(msg))) => {
            assert!(msg.contains("raw_embedding_dimension 1024"), "{msg}");
        }
        Err(other) => panic!("expected ModelLoad(Runtime), got {other:?}"),
        Ok(_) => panic!("a 1024-wide bundle loaded against the 768-wide GGUF"),
    }

    fs::remove_dir_all(temp_dir).unwrap();
}

/// The test bundle with `<arch>.embedding_length_out = 512` added to a copy of its GGUF.
/// llama.cpp then reports `n_embd` 768, which matches `raw_embedding_dimension`, but
/// `n_embd_out` 512, so pooled vectors are narrower than the 768 floats `embed` reads. The
/// load must fail on `n_embd_out`; the width test above never gets past `n_embd`.
#[test]
fn test_gguf_n_embd_out_mismatch_returns_model_load_error() {
    let Some(bundle_dir) = test_bundle(ENGINE_BUNDLE_FILES) else {
        return;
    };
    // The GGUF copy is about 170 MB.
    let temp_dir = TempDir::new();
    link_bundle_files(
        &bundle_dir,
        temp_dir.path(),
        &["tokenizer.json", "build-info.json"],
    );
    let gguf = GgufFile::read(&bundle_dir.join("model.gguf"));
    let key = format!("{}.embedding_length_out", gguf.architecture);
    gguf.copy_with_u32_key(&temp_dir.path().join("model.gguf"), &key, 512);

    let result = EmbeddingEngine::from_gguf_bundle_dir(temp_dir.path(), EngineConfig::default());
    match result {
        Err(LTEmbedError::ModelLoad(ModelLoadError::Runtime(msg))) => {
            assert!(msg.contains("n_embd_out 512"), "{msg}");
        }
        Err(other) => panic!("expected ModelLoad(Runtime), got {other:?}"),
        // Do not embed: the pooled buffer would be 512 floats, shorter than `embed` reads.
        Ok(_) => panic!(
            "the bundle loaded with {key} = 512 in its GGUF: either load no longer checks \
             n_embd_out, or llama.cpp no longer reads {key}"
        ),
    }
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
    let Some(bundle_dir) = test_bundle(ENGINE_BUNDLE_FILES) else {
        return;
    };
    let engine = make_engine(&bundle_dir);
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
    let Some(bundle_dir) = test_bundle(ENGINE_BUNDLE_FILES) else {
        return;
    };
    let engine = make_engine(&bundle_dir);
    let v = engine
        .embed(EmbeddingInput::query("normalization check"))
        .unwrap();
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert_relative_eq!(norm, 1.0, epsilon = 1e-5);
}

#[test]
fn test_output_dimension_is_512() {
    let Some(bundle_dir) = test_bundle(ENGINE_BUNDLE_FILES) else {
        return;
    };
    let engine = make_engine(&bundle_dir);
    let v = engine
        .embed(EmbeddingInput::query("dimension check"))
        .unwrap();
    assert_eq!(v.len(), EMBEDDING_DIMENSION);
}

#[test]
fn test_missing_bundle_skips_unless_required() {
    let temp_dir = unique_temp_dir();
    fs::create_dir_all(&temp_dir).unwrap();
    write_tokenizer(&temp_dir);

    assert_eq!(bundle_or_skip(None, false, ENGINE_BUNDLE_FILES), None);
    assert_eq!(
        bundle_or_skip(Some(temp_dir.clone()), false, ENGINE_BUNDLE_FILES),
        None
    );
    assert_eq!(
        bundle_or_skip(Some(temp_dir.clone()), true, &["tokenizer.json"]),
        Some(temp_dir.clone())
    );

    fs::remove_dir_all(temp_dir).unwrap();
}

#[test]
#[should_panic(
    expected = "LTEMBED_REQUIRE_TEST_BUNDLE=1 requires the test bundle, but LTEMBED_TEST_BUNDLE_DIR is not set"
)]
fn test_required_bundle_fails_when_dir_unset() {
    bundle_or_skip(None, true, ENGINE_BUNDLE_FILES);
}

#[test]
#[should_panic(expected = "has no model.gguf or tokenizer.json")]
fn test_required_bundle_fails_when_files_missing() {
    bundle_or_skip(Some(unique_temp_dir()), true, ENGINE_BUNDLE_FILES);
}
