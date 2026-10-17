use approx::assert_relative_eq;
use ltembed::engine::{
    EmbeddingEngine, EmbeddingInput, EngineConfig, DOCUMENT_PREFIX, EMBEDDING_DIMENSION,
    MAX_LENGTH, QUERY_PREFIX, RAW_EMBEDDING_DIMENSION,
};
use ltembed::error::{LTEmbedError, ModelLoadError};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::env::VarError;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Barrier;
use std::time::{SystemTime, UNIX_EPOCH};

const FIXTURES: &str = "tests/fixtures/test_fixtures.json";
const LLAMA_CPP_REFERENCE: &str = "tests/fixtures/llama_cpp_reference.json";
const TEST_BUNDLE_ENV: &str = "LTEMBED_TEST_BUNDLE_DIR";
/// `1` turns every bundle-gated skip into a failure. CI sets it so that losing
/// `LTEMBED_TEST_BUNDLE_DIR` cannot leave the model-backed tests silently skipped.
const REQUIRE_TEST_BUNDLE_ENV: &str = "LTEMBED_REQUIRE_TEST_BUNDLE";
/// Bundle files the engine tests need. A missing `build-info.json` is not a skip: the engine
/// reports it as a load error and the test fails.
const ENGINE_BUNDLE_FILES: &[&str] = &["model.gguf"];

/// Lowest cosine allowed between the engine's output and the F32 PyTorch golden, per input
/// and at every width. The release GGUF is 5-bit: measured against the golden, UD-Q5_K_XL
/// reaches 0.9969 at worst (a Rust snippet at 768 dimensions) and the BF16 GGUF of the same
/// revision 0.99999 through this same engine, so the gap is weight quantization. Semantic
/// mistakes fall far below the gate (docs/testing.md § "Golden fixture"). Every input must
/// pass it, so it also bounds the mean.
const GOLDEN_MIN_COSINE: f64 = 0.995;
/// Lowest cosine allowed between the engine's raw vector and upstream llama.cpp's own
/// embedding tool on the same GGUF, kernels and settings (measured: identical up to the
/// tool's 7-decimal output). Anything lower is an LTEmbed error, not quantization.
const LLAMA_CPP_MIN_COSINE: f64 = 0.99999;
/// Largest allowed relative difference between those two vectors' norms.
const LLAMA_CPP_MAX_NORM_ERROR: f64 = 1e-4;

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
    make_engine_with(bundle_dir, EMBEDDING_DIMENSION, true, 1)
}

fn make_engine_with(
    bundle_dir: &Path,
    output_dimension: usize,
    l2_normalize: bool,
    n_threads: usize,
) -> EmbeddingEngine {
    EmbeddingEngine::from_gguf_bundle_dir_with_threads(
        bundle_dir,
        EngineConfig {
            output_dimension,
            l2_normalize,
        },
        n_threads,
    )
    .expect("Failed to initialize EmbeddingEngine from bundle")
}

/// Threads for the tests that embed inputs of thousands of tokens.
fn long_input_threads() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get().min(8))
}

/// Cosine similarity, accumulated in f64.
fn cosine(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "cosine of vectors of different lengths");
    let dot: f64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| f64::from(x) * f64::from(y))
        .sum();
    dot / (norm(a) * norm(b))
}

fn norm(v: &[f32]) -> f64 {
    v.iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum::<f64>()
        .sqrt()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum FixtureKind {
    Query,
    Document,
}

#[derive(Deserialize)]
struct Fixture {
    name: String,
    kind: FixtureKind,
    text: String,
    /// Token count of the prompted text, BOS and EOS included.
    tokens: usize,
    /// For a query: the document fixture that must rank first.
    relevant: Option<String>,
    /// The `raw_dim`-wide, L2-normalized PyTorch embedding.
    embedding: Vec<f32>,
}

impl Fixture {
    fn input(&self) -> EmbeddingInput<'_> {
        match self.kind {
            FixtureKind::Query => EmbeddingInput::query(&self.text),
            FixtureKind::Document => EmbeddingInput::document(&self.text),
        }
    }
}

#[derive(Deserialize)]
struct Prompts {
    query: String,
    document: String,
}

/// `tests/fixtures/test_fixtures.json`, written by `scripts/generate_fixtures.py`.
#[derive(Deserialize)]
struct FixtureFile {
    prompts: Prompts,
    raw_dim: usize,
    /// The Matryoshka widths the tests check.
    dims: Vec<usize>,
    max_length: usize,
    /// Inputs of at most this many tokens are checked at every width.
    short_tokens: usize,
    fixtures: Vec<Fixture>,
}

impl FixtureFile {
    fn short_fixtures(&self) -> Vec<&Fixture> {
        self.fixtures
            .iter()
            .filter(|fixture| fixture.tokens <= self.short_tokens)
            .collect()
    }
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
            "{FIXTURES} {problem}. It must hold the PyTorch reference for the model the public \
             constants describe, so changing the model, its prompts or its widths means \
             regenerating it with scripts/generate_fixtures.py (see docs/testing.md)."
        );
    }
    data
}

/// Why `data` does not describe the model the public constants describe, or `None` if it
/// does. A stale fixture is a problem in the repository, not a missing local resource, so it
/// fails every run instead of skipping the way a missing bundle does.
fn golden_fixture_problem(data: &FixtureFile) -> Option<String> {
    if data.prompts.query != QUERY_PREFIX || data.prompts.document != DOCUMENT_PREFIX {
        return Some(format!(
            "records prompts {:?} / {:?}, but QUERY_PREFIX / DOCUMENT_PREFIX are {QUERY_PREFIX:?} \
             / {DOCUMENT_PREFIX:?}",
            data.prompts.query, data.prompts.document
        ));
    }
    if data.raw_dim != RAW_EMBEDDING_DIMENSION {
        return Some(format!(
            "records raw_dim {}, but RAW_EMBEDDING_DIMENSION is {RAW_EMBEDDING_DIMENSION}",
            data.raw_dim
        ));
    }
    if data.max_length != MAX_LENGTH {
        return Some(format!(
            "records max_length {}, but MAX_LENGTH is {MAX_LENGTH}",
            data.max_length
        ));
    }
    if !data.dims.contains(&EMBEDDING_DIMENSION) {
        return Some(format!(
            "checks widths {:?}, which leave out EMBEDDING_DIMENSION {EMBEDDING_DIMENSION}",
            data.dims
        ));
    }
    if let Some(dim) = data
        .dims
        .iter()
        .find(|&&dim| dim == 0 || dim > data.raw_dim)
    {
        return Some(format!("checks width {dim}, outside 1..={}", data.raw_dim));
    }
    if data.fixtures.is_empty() {
        return Some("has no fixtures".to_string());
    }
    for fixture in &data.fixtures {
        if fixture.embedding.len() != data.raw_dim {
            return Some(format!(
                "records raw_dim {}, but fixture {} has {} values",
                data.raw_dim,
                fixture.name,
                fixture.embedding.len()
            ));
        }
        if fixture.tokens > data.max_length {
            return Some(format!(
                "has fixture {} of {} tokens, more than max_length {}",
                fixture.name, fixture.tokens, data.max_length
            ));
        }
        if let Some(relevant) = &fixture.relevant {
            let is_short_document = data.short_fixtures().iter().any(|candidate| {
                candidate.name == *relevant && candidate.kind == FixtureKind::Document
            });
            if fixture.kind != FixtureKind::Query || !is_short_document {
                return Some(format!(
                    "has fixture {} whose relevant document {relevant} is not a short \
                     document fixture",
                    fixture.name
                ));
            }
        }
    }
    None
}

/// `tests/fixtures/llama_cpp_reference.json`, written by
/// `scripts/generate_llama_cpp_reference.py`: upstream llama.cpp's own embedding tool on the
/// release GGUF, run on the golden fixture's inputs.
#[derive(Deserialize)]
struct LlamaCppReference {
    gguf: GgufSource,
    raw_dim: usize,
    /// One entry per golden fixture, in the same order.
    inputs: Vec<ReferenceInput>,
    /// One section per static llama.cpp variant: their kernels round differently.
    variants: HashMap<String, ReferenceVariant>,
}

#[derive(Deserialize)]
struct GgufSource {
    sha256: String,
}

#[derive(Deserialize)]
struct ReferenceInput {
    name: String,
    /// The GGUF vocabulary's tokenization of the prompted input.
    token_ids: Vec<u32>,
}

#[derive(Deserialize)]
struct ReferenceVariant {
    embeddings: Vec<ReferenceEmbedding>,
}

#[derive(Deserialize)]
struct ReferenceEmbedding {
    name: String,
    /// The raw (un-normalized) pooled, projected vector.
    embedding: Vec<f32>,
}

/// The static llama.cpp variant this test binary links (see build.rs).
fn static_llama_variant() -> &'static str {
    if cfg!(target_arch = "x86_64") {
        "x86_64-v3"
    } else if cfg!(target_arch = "aarch64") {
        "aarch64-graviton2"
    } else {
        panic!("no static llama.cpp variant for this architecture")
    }
}

/// The reference, checked against the golden fixture: same inputs, in the same order.
fn llama_cpp_reference(golden: &FixtureFile) -> LlamaCppReference {
    let json = fs::read_to_string(LLAMA_CPP_REFERENCE).expect(
        "tests/fixtures/llama_cpp_reference.json not found — run \
         scripts/generate_llama_cpp_reference.py",
    );
    let reference: LlamaCppReference = serde_json::from_str(&json)
        .unwrap_or_else(|e| panic!("{LLAMA_CPP_REFERENCE} does not parse: {e}"));
    let golden_names: Vec<&str> = golden.fixtures.iter().map(|f| f.name.as_str()).collect();
    let input_names: Vec<&str> = reference.inputs.iter().map(|i| i.name.as_str()).collect();
    assert_eq!(
        input_names, golden_names,
        "{LLAMA_CPP_REFERENCE} was generated from other inputs than {FIXTURES}: rerun \
         scripts/generate_llama_cpp_reference.py for every variant (docs/testing.md)"
    );
    assert_eq!(reference.raw_dim, golden.raw_dim);
    for (variant, section) in &reference.variants {
        let names: Vec<&str> = section.embeddings.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(
            names, golden_names,
            "{LLAMA_CPP_REFERENCE} variant {variant}"
        );
        for entry in &section.embeddings {
            assert_eq!(
                entry.embedding.len(),
                reference.raw_dim,
                "{variant} {}",
                entry.name
            );
        }
    }
    reference
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
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

fn write_model_stub(dir: &Path) {
    fs::write(dir.join("model.gguf"), "stub").unwrap();
}

/// The `build-info.json` that CI and the release workflow write.
fn valid_build_info() -> serde_json::Value {
    serde_json::json!({
        "target_id": "unsloth/embeddinggemma-2-GGUF",
        "model_metadata": {
            "model_format": "gguf",
            "quant": "UD-Q5_K_XL",
            "pooling": "mean",
            "input_kind": "retrieval",
            "query_prefix": "task: search result | query: ",
            "document_prefix": "title: none | text: ",
            "raw_embedding_dimension": 768,
            "max_length": 8192
        }
    })
}

fn valid_build_info_json() -> String {
    valid_build_info().to_string()
}

/// The test bundle's `build-info.json` with `edit` applied, written to `dir`.
fn write_edited_bundle_build_info(
    bundle_dir: &Path,
    dir: &Path,
    edit: impl FnOnce(&mut serde_json::Value),
) {
    let build_info = fs::read_to_string(bundle_dir.join("build-info.json")).unwrap();
    let mut build_info: serde_json::Value = serde_json::from_str(&build_info).unwrap();
    edit(&mut build_info);
    write_build_info(dir, &build_info.to_string());
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
const GGUF_TYPE_BOOL: u32 = 7;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_ARRAY: u32 = 9;

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
                self.skip_array(element_type, len);
            }
            _ => panic!("unknown GGUF value type {value_type}"),
        }
    }

    /// Skips the `len` elements of an array whose element type and length were already read.
    fn skip_array(&mut self, element_type: u32, len: u64) {
        match gguf_fixed_size(element_type) {
            Some(size) => self.skip(len * size),
            None => (0..len).for_each(|_| self.skip_value(element_type)),
        }
    }

    fn position(&mut self) -> u64 {
        self.0.stream_position().unwrap()
    }
}

/// The metadata of a GGUF v3 file: a 24-byte header (magic, version, tensor count, key-value
/// count) followed by the key-value pairs. Holds what the tests below read or patch.
struct GgufFile {
    path: PathBuf,
    /// `general.architecture`, the prefix of the model's hyperparameter keys.
    architecture: String,
    /// File offset and type of the value of each `uint32` and `bool` key.
    fixed_offsets: HashMap<String, (u32, u64)>,
}

impl GgufFile {
    fn read(path: &Path) -> Self {
        let mut reader = GgufReader(BufReader::new(File::open(path).unwrap()));
        assert_eq!(&reader.bytes(), b"GGUF", "{} is not a GGUF", path.display());
        let version = reader.u32();
        assert_eq!(version, 3, "{} is GGUF v{version}, not v3", path.display());
        let _n_tensors = reader.u64();
        let n_kv = reader.u64();

        let mut architecture = None;
        let mut fixed_offsets = HashMap::new();
        for _ in 0..n_kv {
            let key = reader.string();
            let value_type = reader.u32();
            match (key.as_str(), value_type) {
                ("general.architecture", GGUF_TYPE_STRING) => {
                    architecture = Some(reader.string());
                }
                (_, GGUF_TYPE_UINT32 | GGUF_TYPE_BOOL) => {
                    fixed_offsets.insert(key, (value_type, reader.position()));
                    reader.skip_value(value_type);
                }
                _ => reader.skip_value(value_type),
            }
        }
        Self {
            path: path.to_path_buf(),
            architecture: architecture
                .unwrap_or_else(|| panic!("{} has no general.architecture", path.display())),
            fixed_offsets,
        }
    }

    /// Copies the file to `dst` with the value of `key`, which it must already have with type
    /// `value_type`, replaced by `value`. The value keeps its size, so nothing else in the file
    /// moves.
    fn copy_with_value(&self, dst: &Path, key: &str, value_type: u32, value: &[u8]) {
        let (found_type, offset) = *self.fixed_offsets.get(key).unwrap_or_else(|| {
            panic!("{} has no key {key}", self.path.display());
        });
        assert_eq!(found_type, value_type, "{} {key}", self.path.display());
        assert_eq!(gguf_fixed_size(value_type), Some(value.len() as u64));
        fs::copy(&self.path, dst).unwrap();
        let mut out = OpenOptions::new().write(true).open(dst).unwrap();
        out.seek(SeekFrom::Start(offset)).unwrap();
        out.write_all(value).unwrap();
        out.flush().unwrap();
    }

    fn copy_with_u32(&self, dst: &Path, key: &str, value: u32) {
        self.copy_with_value(dst, key, GGUF_TYPE_UINT32, &value.to_le_bytes());
    }

    fn copy_with_bool(&self, dst: &Path, key: &str, value: bool) {
        self.copy_with_value(dst, key, GGUF_TYPE_BOOL, &[u8::from(value)]);
    }
}

/// Needs no bundle, so a fixture left stale by a model, prompt or width change fails every
/// run, not only the runs that can execute the parity tests.
#[test]
fn test_golden_fixture_matches_engine_constants() {
    let data = golden_fixtures();
    // The golden is L2-normalized at full width.
    for fixture in &data.fixtures {
        assert_relative_eq!(norm(&fixture.embedding), 1.0, epsilon = 1e-5);
    }
}

#[test]
fn test_golden_fixture_problems() {
    let fixture = |name: &str, kind: FixtureKind, relevant: Option<&str>| Fixture {
        name: name.to_string(),
        kind,
        text: "text".to_string(),
        tokens: 4,
        relevant: relevant.map(str::to_string),
        embedding: vec![0.0; RAW_EMBEDDING_DIMENSION],
    };
    let valid = || FixtureFile {
        prompts: Prompts {
            query: QUERY_PREFIX.to_string(),
            document: DOCUMENT_PREFIX.to_string(),
        },
        raw_dim: RAW_EMBEDDING_DIMENSION,
        dims: vec![RAW_EMBEDDING_DIMENSION, EMBEDDING_DIMENSION],
        max_length: MAX_LENGTH,
        short_tokens: 512,
        fixtures: vec![
            fixture("q", FixtureKind::Query, Some("d")),
            fixture("d", FixtureKind::Document, None),
        ],
    };
    assert_eq!(golden_fixture_problem(&valid()), None);

    // The previous model's prompts.
    let mut data = valid();
    data.prompts.query = "Query: ".to_string();
    assert!(golden_fixture_problem(&data)
        .unwrap()
        .starts_with("records prompts \"Query: \""));

    let mut data = valid();
    data.raw_dim = 512;
    assert_eq!(
        golden_fixture_problem(&data),
        Some(format!(
            "records raw_dim 512, but RAW_EMBEDDING_DIMENSION is {RAW_EMBEDDING_DIMENSION}"
        ))
    );

    let mut data = valid();
    data.dims = vec![RAW_EMBEDDING_DIMENSION];
    assert!(golden_fixture_problem(&data)
        .unwrap()
        .contains("leave out EMBEDDING_DIMENSION"));

    let mut data = valid();
    data.fixtures.clear();
    assert_eq!(
        golden_fixture_problem(&data),
        Some("has no fixtures".to_string())
    );

    let mut data = valid();
    data.fixtures[1].embedding.truncate(EMBEDDING_DIMENSION);
    assert_eq!(
        golden_fixture_problem(&data),
        Some(format!(
            "records raw_dim {RAW_EMBEDDING_DIMENSION}, but fixture d has \
             {EMBEDDING_DIMENSION} values"
        ))
    );

    let mut data = valid();
    data.fixtures[0].relevant = Some("q".to_string());
    assert!(golden_fixture_problem(&data)
        .unwrap()
        .contains("relevant document q is not a short document fixture"));
}

/// Detecting a stale fixture is not enough: loading it must fail the run.
#[test]
#[should_panic(expected = "records prompts \"Query: \" / \"Document: \"")]
fn test_stale_golden_fixture_fails() {
    parse_golden_fixtures(
        r#"{"prompts": {"query": "Query: ", "document": "Document: "}, "raw_dim": 768,
            "dims": [512], "max_length": 8192, "short_tokens": 512,
            "fixtures": [{"name": "q", "kind": "query", "text": "text", "tokens": 4,
                          "embedding": []}]}"#,
    );
}

/// A fixture in the previous model's format (one `dim`, no prompts) fails too.
#[test]
#[should_panic(expected = "missing field `prompts`")]
fn test_golden_fixture_without_prompts_fails() {
    parse_golden_fixtures(r#"{"dim": 512, "fixtures": []}"#);
}

/// Needs no bundle: the reference must cover the golden inputs and this target's variant.
#[test]
fn test_llama_cpp_reference_matches_golden_inputs() {
    let golden = golden_fixtures();
    let reference = llama_cpp_reference(&golden);
    let variant = static_llama_variant();
    assert!(
        reference.variants.contains_key(variant),
        "{LLAMA_CPP_REFERENCE} has no {variant} section: run \
         scripts/generate_llama_cpp_reference.py --variant {variant} (docs/testing.md)"
    );
    for (fixture, input) in golden.fixtures.iter().zip(&reference.inputs) {
        assert_eq!(input.token_ids.len(), fixture.tokens, "{}", fixture.name);
    }
}

/// The engine against the F32 PyTorch golden at every width the fixture lists (768, 512, 256
/// and 128): each output has that width, unit norm and finite values, stays within quantization
/// error of the golden truncated to that width and renormalized, and ranks each query's
/// relevant document first. Only the short inputs: truncation does not depend on length, and
/// the long ones are checked at full width by the llama.cpp reference test.
#[test]
fn test_golden_parity_at_every_trained_dimension() {
    let Some(bundle_dir) = test_bundle(ENGINE_BUNDLE_FILES) else {
        return;
    };
    let data = golden_fixtures();
    let fixtures = data.short_fixtures();
    let inputs: Vec<EmbeddingInput<'_>> = fixtures.iter().map(|f| f.input()).collect();

    let mut failures = Vec::new();
    let mut report = Vec::new();
    for &dim in &data.dims {
        let engine = make_engine_with(&bundle_dir, dim, true, 1);
        let outputs = engine.embed_batch(&inputs).unwrap();
        assert_eq!(outputs.len(), fixtures.len());

        let mut cosines = Vec::new();
        for (fixture, output) in fixtures.iter().zip(&outputs) {
            assert_eq!(output.len(), dim, "{} at {dim}", fixture.name);
            assert!(
                output.iter().all(|v| v.is_finite()),
                "{} at {dim} is not finite",
                fixture.name
            );
            assert_relative_eq!(norm(output), 1.0, epsilon = 1e-5);
            // Cosine does not depend on scale, so truncating the golden is enough.
            let similarity = cosine(output, &fixture.embedding[..dim]);
            if similarity < GOLDEN_MIN_COSINE {
                failures.push(format!(
                    "{} at {dim}: cosine {similarity:.6} < {GOLDEN_MIN_COSINE}",
                    fixture.name
                ));
            }
            cosines.push(similarity);
        }
        let mean = cosines.iter().sum::<f64>() / cosines.len() as f64;
        let min = cosines.iter().copied().fold(f64::INFINITY, f64::min);
        report.push(format!("{dim}: min {min:.6} mean {mean:.6}"));

        let documents: Vec<(&Fixture, &Vec<f32>)> = fixtures
            .iter()
            .copied()
            .zip(&outputs)
            .filter(|(fixture, _)| fixture.kind == FixtureKind::Document)
            .collect();
        for (fixture, query) in fixtures.iter().zip(&outputs) {
            let Some(relevant) = &fixture.relevant else {
                continue;
            };
            let (best, _) = documents
                .iter()
                .map(|(document, output)| (&document.name, cosine(query, output)))
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap();
            if best != relevant {
                failures.push(format!(
                    "{} at {dim}: ranks {best} first, not {relevant}",
                    fixture.name
                ));
            }
        }
    }
    eprintln!("golden parity: {}", report.join("; "));
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Every golden input, up to the full `MAX_LENGTH` tokens, against upstream llama.cpp's own
/// embedding tool on the same GGUF and kernels: the raw vector must match the tool's (an
/// LTEmbed tokenization, prefix, pooling, projection or FFI error, not quantization, would show
/// here). The tool's token ids are compared with the engine's tokenization by
/// `test_gguf_vocabulary_matches_reference_token_ids` (src/engine/tokenizer_parity.rs). Each
/// raw vector is also checked against the F32 golden at full and default width.
///
/// The inputs run one after another on one engine, which holds the memory the longest needed
/// (about 1.4 GiB at 8192 tokens) until it is dropped (docs/architecture.md).
#[test]
fn test_llama_cpp_reference_parity() {
    let Some(bundle_dir) = test_bundle(ENGINE_BUNDLE_FILES) else {
        return;
    };
    let golden = golden_fixtures();
    let reference = llama_cpp_reference(&golden);
    let variant = static_llama_variant();
    let expected = &reference
        .variants
        .get(variant)
        .unwrap_or_else(|| panic!("{LLAMA_CPP_REFERENCE} has no {variant} section"))
        .embeddings;
    let model_sha256 = sha256_hex(&fs::read(bundle_dir.join("model.gguf")).unwrap());
    assert_eq!(
        model_sha256, reference.gguf.sha256,
        "the test bundle's model.gguf is not the GGUF {LLAMA_CPP_REFERENCE} was generated from"
    );

    let engine = make_engine_with(
        &bundle_dir,
        RAW_EMBEDDING_DIMENSION,
        false,
        long_input_threads(),
    );

    let mut failures = Vec::new();
    for (fixture, expected) in golden.fixtures.iter().zip(expected) {
        let name = &fixture.name;
        let raw = engine.embed(fixture.input()).unwrap();
        assert_eq!(raw.len(), RAW_EMBEDDING_DIMENSION, "{name}");
        assert!(raw.iter().all(|v| v.is_finite()), "{name} is not finite");

        let similarity = cosine(&raw, &expected.embedding);
        let norm_error = (norm(&raw) / norm(&expected.embedding) - 1.0).abs();
        if similarity < LLAMA_CPP_MIN_COSINE || norm_error > LLAMA_CPP_MAX_NORM_ERROR {
            failures.push(format!(
                "{name}: vs llama.cpp ({variant}) cosine {similarity:.9}, norm error \
                 {norm_error:.2e}"
            ));
        }
        for dim in [RAW_EMBEDDING_DIMENSION, EMBEDDING_DIMENSION] {
            let similarity = cosine(&raw[..dim], &fixture.embedding[..dim]);
            if similarity < GOLDEN_MIN_COSINE {
                failures.push(format!(
                    "{name} at {dim}: vs golden cosine {similarity:.6} < {GOLDEN_MIN_COSINE}"
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn test_missing_model_file_returns_model_load_error() {
    let temp_dir = unique_temp_dir();
    fs::create_dir_all(&temp_dir).unwrap();
    write_build_info(&temp_dir, &valid_build_info_json());

    let result = EmbeddingEngine::from_gguf_bundle_dir(&temp_dir, EngineConfig::default());
    assert!(matches!(
        result.unwrap_err(),
        LTEmbedError::ModelLoad(ModelLoadError::MissingFile { .. })
    ));

    fs::remove_dir_all(temp_dir).unwrap();
}

/// A bundle is `model.gguf` and `build-info.json`: the GGUF's vocabulary tokenizes, and a
/// `tokenizer.json` beside them, even one that does not parse, is not read.
#[test]
fn test_bundle_without_tokenizer_json_loads_and_ignores_one() {
    let Some(bundle_dir) = test_bundle(ENGINE_BUNDLE_FILES) else {
        return;
    };
    let temp_dir = unique_temp_dir();
    fs::create_dir_all(&temp_dir).unwrap();
    link_bundle_files(&bundle_dir, &temp_dir, &["model.gguf", "build-info.json"]);
    let input = EmbeddingInput::query("Which tokenizer splits this text?");
    let expected = make_engine(&bundle_dir).embed(input).unwrap();

    assert_eq!(make_engine(&temp_dir).embed(input).unwrap(), expected);
    fs::write(temp_dir.join("tokenizer.json"), "not a tokenizer").unwrap();
    assert_eq!(make_engine(&temp_dir).embed(input).unwrap(), expected);

    fs::remove_dir_all(temp_dir).unwrap();
}

#[test]
fn test_missing_build_info_returns_model_load_error() {
    let temp_dir = unique_temp_dir();
    fs::create_dir_all(&temp_dir).unwrap();
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
    write_model_stub(&temp_dir);
    let mut build_info = valid_build_info();
    build_info["model_metadata"]["input_kind"] = "classification".into();
    write_build_info(&temp_dir, &build_info.to_string());

    let result = EmbeddingEngine::from_gguf_bundle_dir(&temp_dir, EngineConfig::default());
    assert!(matches!(
        result.unwrap_err(),
        LTEmbedError::ModelLoad(ModelLoadError::UnsupportedInputKind { .. })
    ));
    fs::remove_dir_all(temp_dir).unwrap();
}

/// A bundle assembled for the previous model's last-token pooling.
#[test]
fn test_last_token_pooling_returns_model_load_error() {
    let temp_dir = unique_temp_dir();
    fs::create_dir_all(&temp_dir).unwrap();
    write_model_stub(&temp_dir);
    let mut build_info = valid_build_info();
    build_info["model_metadata"]["pooling"] = "last_token".into();
    write_build_info(&temp_dir, &build_info.to_string());

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
    write_model_stub(&temp_dir);
    write_build_info(&temp_dir, &valid_build_info_json());

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

/// Loads the test bundle with `raw_embedding_dimension` set to `raw` and returns the load
/// error's message. The GGUF's projected output is 768 wide, so the load itself must fail,
/// not the first `embed`.
fn load_with_raw_dimension(bundle_dir: &Path, raw: usize) -> String {
    let temp_dir = unique_temp_dir();
    fs::create_dir_all(&temp_dir).unwrap();
    link_bundle_files(bundle_dir, &temp_dir, ENGINE_BUNDLE_FILES);
    write_edited_bundle_build_info(bundle_dir, &temp_dir, |build_info| {
        build_info["model_metadata"]["raw_embedding_dimension"] = raw.into();
    });

    let result = EmbeddingEngine::from_gguf_bundle_dir(
        &temp_dir,
        EngineConfig {
            output_dimension: 128,
            l2_normalize: true,
        },
    );
    fs::remove_dir_all(temp_dir).unwrap();
    match result {
        Err(LTEmbedError::ModelLoad(ModelLoadError::Runtime(msg))) => msg,
        Err(other) => panic!("expected ModelLoad(Runtime), got {other:?}"),
        Ok(_) => panic!("a {raw}-wide bundle loaded against the 768-wide GGUF"),
    }
}

#[test]
fn test_gguf_width_mismatch_returns_model_load_error() {
    let Some(bundle_dir) = test_bundle(ENGINE_BUNDLE_FILES) else {
        return;
    };
    let msg = load_with_raw_dimension(&bundle_dir, 1024);
    assert!(msg.contains("raw_embedding_dimension 1024"), "{msg}");
}

/// The hidden width (512) is not the embedding width: the engine reads the projected output.
#[test]
fn test_hidden_width_as_raw_dimension_returns_model_load_error() {
    let Some(bundle_dir) = test_bundle(ENGINE_BUNDLE_FILES) else {
        return;
    };
    let msg = load_with_raw_dimension(&bundle_dir, 512);
    assert!(msg.contains("n_embd_out 768"), "{msg}");
    assert!(msg.contains("raw_embedding_dimension 512"), "{msg}");
    assert!(msg.contains("n_embd is 512"), "{msg}");
}

/// The test bundle with its GGUF's `<arch>.pooling_type` changed from mean (1) to last (3).
/// llama.cpp loads it, so the engine's own check of the declared pooling must reject it.
#[test]
fn test_gguf_without_mean_pooling_returns_model_load_error() {
    let Some(bundle_dir) = test_bundle(ENGINE_BUNDLE_FILES) else {
        return;
    };
    // The GGUF copy is about 210 MB.
    let temp_dir = TempDir::new();
    link_bundle_files(&bundle_dir, temp_dir.path(), &["build-info.json"]);
    let gguf = GgufFile::read(&bundle_dir.join("model.gguf"));
    let key = format!("{}.pooling_type", gguf.architecture);
    gguf.copy_with_u32(&temp_dir.path().join("model.gguf"), &key, 3);

    let result = EmbeddingEngine::from_gguf_bundle_dir(temp_dir.path(), EngineConfig::default());
    match result {
        Err(LTEmbedError::ModelLoad(ModelLoadError::Runtime(msg))) => {
            assert!(msg.contains(&format!("GGUF {key} is 3")), "{msg}");
        }
        Err(other) => panic!("expected ModelLoad(Runtime), got {other:?}"),
        Ok(_) => panic!("the bundle loaded with {key} = 3 in its GGUF"),
    }
}

/// The test bundle with its GGUF's `tokenizer.ggml.add_eos_token` set to false. llama.cpp
/// loads it and tokenizes without EOS, so every embedding would differ from the model's; the
/// engine's check of the GGUF's tokenization must reject it. (`add_bos_token = false` cannot
/// be tested this way: llama.cpp overrides it to true for a Gemma 4 tokenizer.)
#[test]
fn test_gguf_without_eos_returns_model_load_error() {
    let Some(bundle_dir) = test_bundle(ENGINE_BUNDLE_FILES) else {
        return;
    };
    // The GGUF copy is about 210 MB.
    let temp_dir = TempDir::new();
    link_bundle_files(&bundle_dir, temp_dir.path(), &["build-info.json"]);
    let key = "tokenizer.ggml.add_eos_token";
    GgufFile::read(&bundle_dir.join("model.gguf")).copy_with_bool(
        &temp_dir.path().join("model.gguf"),
        key,
        false,
    );

    let result = EmbeddingEngine::from_gguf_bundle_dir(temp_dir.path(), EngineConfig::default());
    match result {
        Err(LTEmbedError::ModelLoad(ModelLoadError::Runtime(msg))) => {
            assert!(
                msg.contains("the GGUF's tokenization adds no EOS:"),
                "{msg}"
            );
        }
        Err(other) => panic!("expected ModelLoad(Runtime), got {other:?}"),
        Ok(_) => panic!("the bundle loaded with {key} = false in its GGUF"),
    }
}

/// An input of more than `MAX_LENGTH` tokens is rejected before inference, with its full
/// token count, instead of being truncated.
#[test]
fn test_long_input_returns_input_too_long_error() {
    let Some(bundle_dir) = test_bundle(ENGINE_BUNDLE_FILES) else {
        return;
    };
    let engine = make_engine(&bundle_dir);
    let long_text = "hello world ".repeat(12000);
    match engine.embed(EmbeddingInput::document(&long_text)) {
        Err(LTEmbedError::InputTooLong {
            tokens: Some(tokens),
            bytes,
            max,
        }) => {
            assert!(
                tokens > MAX_LENGTH,
                "tokens={tokens} should be > {MAX_LENGTH}"
            );
            assert_eq!(bytes, DOCUMENT_PREFIX.len() + long_text.len());
            assert_eq!(max, MAX_LENGTH);
        }
        other => panic!("Expected InputTooLong with a token count, got {other:?}"),
    }
}

/// The release GGUF's longest token: 31 × U+2581 (`▁`), 93 bytes. The tokenizer parity tests
/// check that no token of the GGUF is longer. The engine rejects a text of more than
/// `max_length` times its length, prefix included, without tokenizing it.
fn longest_token() -> String {
    "\u{2581}".repeat(31)
}

/// The byte bound never rejects an input that fits: a document of the vocabulary's longest
/// token, as many copies as `MAX_LENGTH` tokens hold, embeds. With BOS, the six prompt tokens,
/// a `▁` for the prompt's last space and EOS, that is `MAX_LENGTH - 9` copies: 761,039 bytes,
/// 817 under the bound. One more copy is tokenized and counted, not rejected by its size.
#[test]
fn test_input_of_the_longest_tokens_that_fits_embeds() {
    let Some(bundle_dir) = test_bundle(ENGINE_BUNDLE_FILES) else {
        return;
    };
    let engine = make_engine_with(&bundle_dir, EMBEDDING_DIMENSION, true, long_input_threads());
    let fits = longest_token().repeat(MAX_LENGTH - 9);
    let embedding = engine.embed(EmbeddingInput::document(&fits)).unwrap();
    assert_eq!(embedding.len(), EMBEDDING_DIMENSION);
    assert!(embedding.iter().all(|value| value.is_finite()));

    let over = longest_token().repeat(MAX_LENGTH - 8);
    match engine.embed(EmbeddingInput::document(&over)) {
        Err(LTEmbedError::InputTooLong { tokens, bytes, max }) => {
            assert_eq!(
                (tokens, bytes, max),
                (
                    Some(MAX_LENGTH + 1),
                    DOCUMENT_PREFIX.len() + over.len(),
                    MAX_LENGTH
                )
            );
        }
        other => panic!("expected InputTooLong with a token count, got {other:?}"),
    }
}

/// A text one byte over the bound, `MAX_LENGTH` times the longest token's length with the
/// prefix, is rejected without being tokenized: its `InputTooLong` has no token count. One at
/// the bound is tokenized and counted.
#[test]
fn test_input_over_the_byte_bound_is_rejected_without_tokenizing() {
    let Some(bundle_dir) = test_bundle(ENGINE_BUNDLE_FILES) else {
        return;
    };
    let engine = make_engine(&bundle_dir);
    let bound = MAX_LENGTH * longest_token().len();
    let text_of = |bytes: usize| "a".repeat(bytes - DOCUMENT_PREFIX.len());

    for bytes in [bound + 1, 64 << 20] {
        let text = text_of(bytes);
        match engine.embed(EmbeddingInput::document(&text)) {
            Err(LTEmbedError::InputTooLong {
                tokens: None,
                bytes: reported,
                max,
            }) => assert_eq!((reported, max), (bytes, MAX_LENGTH)),
            other => panic!("{bytes} bytes: expected InputTooLong without a count, got {other:?}"),
        }
    }

    let at_bound = text_of(bound);
    match engine.embed(EmbeddingInput::document(&at_bound)) {
        Err(LTEmbedError::InputTooLong {
            tokens: Some(tokens),
            bytes,
            ..
        }) => {
            assert!(tokens > MAX_LENGTH, "{tokens} tokens");
            assert_eq!(bytes, bound);
        }
        other => panic!("expected InputTooLong with a token count, got {other:?}"),
    }
}

/// A bundle's `max_length` is the input limit the engine enforces and the context it sizes:
/// an input of exactly that many tokens embeds, one more token fails with `InputTooLong`
/// instead of being truncated. Lowering it is how a deployment caps peak memory.
#[test]
fn test_bundle_max_length_is_enforced_without_truncation() {
    let Some(bundle_dir) = test_bundle(ENGINE_BUNDLE_FILES) else {
        return;
    };
    const LIMIT: usize = 64;
    let temp_dir = unique_temp_dir();
    fs::create_dir_all(&temp_dir).unwrap();
    link_bundle_files(&bundle_dir, &temp_dir, ENGINE_BUNDLE_FILES);
    write_edited_bundle_build_info(&bundle_dir, &temp_dir, |build_info| {
        build_info["model_metadata"]["max_length"] = LIMIT.into();
    });

    // Texts whose prompted form is exactly LIMIT and LIMIT + 1 tokens. An engine whose
    // `max_length` is 8 counts tokens: each text below has more than 8 tokens (one word makes
    // 9) and fewer bytes than its byte bound (8 × 93), so it rejects each one after
    // tokenizing it, naming its token count.
    const COUNTER_LIMIT: usize = 8;
    let counter_dir = unique_temp_dir();
    fs::create_dir_all(&counter_dir).unwrap();
    link_bundle_files(&bundle_dir, &counter_dir, ENGINE_BUNDLE_FILES);
    write_edited_bundle_build_info(&bundle_dir, &counter_dir, |build_info| {
        build_info["model_metadata"]["max_length"] = COUNTER_LIMIT.into();
    });
    let counter = make_engine(&counter_dir);
    let text_of = |tokens: usize| {
        (1..LIMIT)
            .map(|words| vec!["alpha"; words].join(" "))
            .find(|text| match counter.embed(EmbeddingInput::document(text)) {
                Err(LTEmbedError::InputTooLong {
                    tokens: Some(count),
                    ..
                }) => count == tokens,
                other => panic!(
                    "expected InputTooLong with a token count at max_length {COUNTER_LIMIT}, \
                     got {other:?}"
                ),
            })
            .unwrap_or_else(|| panic!("no text of {tokens} tokens"))
    };
    let (fits, too_long) = (text_of(LIMIT), text_of(LIMIT + 1));
    fs::remove_dir_all(counter_dir).unwrap();

    let engine = make_engine(&temp_dir);
    let embedding = engine.embed(EmbeddingInput::document(&fits)).unwrap();
    assert_eq!(embedding.len(), EMBEDDING_DIMENSION);
    match engine.embed(EmbeddingInput::document(&too_long)) {
        Err(LTEmbedError::InputTooLong { tokens, max, .. }) => {
            assert_eq!((tokens, max), (Some(LIMIT + 1), LIMIT));
        }
        other => panic!("expected InputTooLong, got {other:?}"),
    }

    fs::remove_dir_all(temp_dir).unwrap();
}

/// `embed_batch` returns one embedding per input, in input order, each identical to embedding
/// that input alone, and repeating a call returns identical vectors.
#[test]
fn test_embed_batch_matches_single_embeds_and_repeats_exactly() {
    let Some(bundle_dir) = test_bundle(ENGINE_BUNDLE_FILES) else {
        return;
    };
    let engine = make_engine(&bundle_dir);
    let long_document = "Mean pooling averages every token of the sequence. ".repeat(40);
    let inputs = [
        EmbeddingInput::query("hello"),
        EmbeddingInput::document(&long_document),
        EmbeddingInput::query("世界"),
        EmbeddingInput::document(""),
        EmbeddingInput::query("hello"),
    ];
    let batch = engine.embed_batch(&inputs).unwrap();
    assert_eq!(batch.len(), inputs.len());
    for (input, embedding) in inputs.iter().zip(&batch) {
        assert_eq!(
            *embedding,
            engine.embed(*input).unwrap(),
            "embed_batch must equal embed() for the same input"
        );
    }
    assert_eq!(batch[0], batch[4], "the same input twice in one batch");
    assert_ne!(
        batch[0], batch[2],
        "different inputs must not share an embedding"
    );
    assert_eq!(
        batch,
        engine.embed_batch(&inputs).unwrap(),
        "a repeated call"
    );
}

/// What one `embed_batch` call returned, as `test_concurrent_calls_match_a_single_thread`
/// compares it: the vectors, or the `InputTooLong` of an input over `MAX_LENGTH` tokens.
#[derive(Debug, PartialEq)]
enum CallResult {
    Embeddings(Vec<Vec<f32>>),
    TooLong {
        tokens: Option<usize>,
        bytes: usize,
        max: usize,
    },
}

fn call_result(engine: &EmbeddingEngine, inputs: &[EmbeddingInput<'_>]) -> CallResult {
    match engine.embed_batch(inputs) {
        Ok(embeddings) => CallResult::Embeddings(embeddings),
        Err(LTEmbedError::InputTooLong { tokens, bytes, max }) => {
            CallResult::TooLong { tokens, bytes, max }
        }
        Err(err) => panic!("unexpected error: {err}"),
    }
}

/// Threads that share one engine in `test_concurrent_calls_match_a_single_thread`.
const CONCURRENT_THREADS: usize = 8;

/// Calls on one engine from `CONCURRENT_THREADS` threads at once return exactly what they
/// return one after another on one thread. Only inference takes the engine's lock, so threads
/// tokenize short inputs, a document of about 100 tokens and inputs of about 24,000 tokens,
/// which are rejected after tokenizing, while another thread encodes: a data race in
/// tokenizing, or between tokenizing and inference, would show as a different vector or token
/// count. Thread `t` starts at call `t` (modulo the number of calls), so different inputs
/// overlap.
#[test]
fn test_concurrent_calls_match_a_single_thread() {
    let Some(bundle_dir) = test_bundle(ENGINE_BUNDLE_FILES) else {
        return;
    };
    const ROUNDS: usize = 2;
    let engine = make_engine(&bundle_dir);
    let document = "Mean pooling averages every token of the sequence. ".repeat(10);
    let too_long = "hello world ".repeat(12000);
    let calls = [
        vec![EmbeddingInput::query("hello")],
        vec![EmbeddingInput::document(&too_long)],
        vec![EmbeddingInput::document(&document)],
        vec![
            EmbeddingInput::query("世界"),
            EmbeddingInput::document("在登录页面点击忘记密码"),
        ],
        vec![EmbeddingInput::query(&too_long)],
    ];
    let expected: Vec<CallResult> = calls
        .iter()
        .map(|inputs| call_result(&engine, inputs))
        .collect();
    for call in [1, 4] {
        assert!(
            matches!(expected[call], CallResult::TooLong { tokens: Some(tokens), .. } if tokens > MAX_LENGTH),
            "call {call} must be tokenized and rejected: {:?}",
            expected[call]
        );
    }

    let start = Barrier::new(CONCURRENT_THREADS);
    std::thread::scope(|scope| {
        for thread in 0..CONCURRENT_THREADS {
            let (engine, calls, expected, start) = (&engine, &calls, &expected, &start);
            scope.spawn(move || {
                start.wait();
                for round in 0..ROUNDS {
                    for offset in 0..calls.len() {
                        let call = (thread + round + offset) % calls.len();
                        let result = call_result(engine, &calls[call]);
                        // Not `assert_eq!`: a mismatch would print every vector.
                        assert!(
                            result == expected[call],
                            "thread {thread}, round {round}: call {call} differs from one \
                             thread's result"
                        );
                    }
                }
            });
        }
    });
}

/// The same text as a query and as a document gets different prompts, so different vectors.
#[test]
fn test_query_and_document_prompts_differ() {
    let Some(bundle_dir) = test_bundle(ENGINE_BUNDLE_FILES) else {
        return;
    };
    let engine = make_engine(&bundle_dir);
    let query = engine
        .embed(EmbeddingInput::query("password reset"))
        .unwrap();
    let document = engine
        .embed(EmbeddingInput::document("password reset"))
        .unwrap();
    let similarity = cosine(&query, &document);
    assert!(similarity < 0.999, "query and document cosine {similarity}");
}

#[test]
fn test_default_config_output_is_normalized_and_512_wide() {
    let Some(bundle_dir) = test_bundle(ENGINE_BUNDLE_FILES) else {
        return;
    };
    let engine = EmbeddingEngine::from_gguf_bundle_dir(&bundle_dir, EngineConfig::default())
        .expect("Failed to initialize EmbeddingEngine from bundle");
    let v = engine
        .embed(EmbeddingInput::query("normalization check"))
        .unwrap();
    assert_eq!(v.len(), EMBEDDING_DIMENSION);
    assert!(v.iter().all(|x| x.is_finite()));
    assert_relative_eq!(norm(&v), 1.0, epsilon = 1e-5);
}

#[test]
fn test_missing_bundle_skips_unless_required() {
    let temp_dir = unique_temp_dir();
    fs::create_dir_all(&temp_dir).unwrap();
    write_build_info(&temp_dir, &valid_build_info_json());

    assert_eq!(bundle_or_skip(None, false, ENGINE_BUNDLE_FILES), None);
    assert_eq!(
        bundle_or_skip(Some(temp_dir.clone()), false, ENGINE_BUNDLE_FILES),
        None
    );
    assert_eq!(
        bundle_or_skip(Some(temp_dir.clone()), true, &["build-info.json"]),
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
#[should_panic(expected = "has no model.gguf")]
fn test_required_bundle_fails_when_files_missing() {
    bundle_or_skip(Some(unique_temp_dir()), true, ENGINE_BUNDLE_FILES);
}
