use std::path::Path;
use std::sync::OnceLock;
use std::time::Instant;

use crate::error::{InferenceError, LTEmbedError, ModelLoadError};

mod backend;
mod bundle;
mod config;
mod inference;
mod input;
mod llama;
mod space;
#[cfg(test)]
mod tokenizer_parity;

pub use config::EngineConfig;
pub use inference::EmbedBatchProfile;
pub use input::{EmbeddingInput, EmbeddingInputKind};
pub use space::EmbeddingSpace;

use backend::EmbeddingBackend;
use bundle::{require_file, ModelSpec};
use inference::{postprocess_embedding, prefix, prefixed_text};
use llama::LlamaBackend;
use space::ModelFile;

/// Width of EmbeddingGemma 2's pooled, projected output (hidden width 512, projected to 768).
pub const RAW_EMBEDDING_DIMENSION: usize = 768;
/// `EngineConfig::default()` output size: the first 512 of the 768 Matryoshka dimensions.
pub const EMBEDDING_DIMENSION: usize = 512;
pub const MAX_LENGTH: usize = 8192;
/// EmbeddingGemma 2's search-query task prompt (its `Retrieval-query` / `SearchQuery` prompt).
pub const QUERY_PREFIX: &str = "task: search result | query: ";
/// EmbeddingGemma 2's untitled-document prompt (its `Retrieval-document` / `Document` prompt).
pub const DOCUMENT_PREFIX: &str = "title: none | text: ";

const MODEL_FILE: &str = "model.gguf";
const BUILD_INFO_FILE: &str = "build-info.json";

/// A loaded embedding model. Backed by an internal backend (currently llama.cpp/GGUF);
/// owns the shared prefix → tokenize → backend → truncate/normalize pipeline. Inputs are
/// tokenized with the GGUF's own vocabulary.
pub struct EmbeddingEngine {
    backend: Box<dyn EmbeddingBackend>,
    spec: ModelSpec,
    config: EngineConfig,
    /// No text of more bytes, prefix included, fits in `spec.max_length` tokens
    /// ([`EmbeddingBackend::max_input_bytes`]).
    max_input_bytes: usize,
    /// The `model.gguf` the backend loaded, hashed by [`EmbeddingEngine::embedding_space`].
    model_file: ModelFile,
    /// Set by the first [`EmbeddingEngine::embedding_space`] call that succeeds.
    embedding_space: OnceLock<EmbeddingSpace>,
}

impl std::fmt::Debug for EmbeddingEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddingEngine").finish_non_exhaustive()
    }
}

impl EmbeddingEngine {
    /// Load from a bundle directory containing `model.gguf` and `build-info.json`, using a
    /// single inference thread. The GGUF's own vocabulary tokenizes the inputs; a
    /// `tokenizer.json` in the directory is not read.
    pub fn from_gguf_bundle_dir(
        bundle_dir: impl AsRef<Path>,
        config: EngineConfig,
    ) -> Result<Self, LTEmbedError> {
        Self::from_gguf_bundle_dir_with_threads(bundle_dir, config, 1)
    }

    /// Like [`EmbeddingEngine::from_gguf_bundle_dir`] but sets the number of llama.cpp
    /// inference threads.
    ///
    /// # Errors
    ///
    /// Returns `LTEmbedError::ModelLoad(ModelLoadError::Config)` if `n_threads` is `0`, in
    /// addition to the bundle-validation and model-load errors.
    pub fn from_gguf_bundle_dir_with_threads(
        bundle_dir: impl AsRef<Path>,
        config: EngineConfig,
        n_threads: usize,
    ) -> Result<Self, LTEmbedError> {
        let bundle_dir = bundle_dir.as_ref();
        let model_path = bundle_dir.join(MODEL_FILE);
        let build_info_path = bundle_dir.join(BUILD_INFO_FILE);

        require_file(&model_path, "GGUF model")?;
        require_file(&build_info_path, "build-info metadata")?;

        let spec = ModelSpec::from_build_info(&build_info_path)?;
        Self::build(&model_path, spec, config, n_threads)
    }

    fn validate_threads(n_threads: usize) -> Result<(), LTEmbedError> {
        if n_threads == 0 {
            return Err(LTEmbedError::ModelLoad(ModelLoadError::Config(
                "n_threads must be greater than zero".into(),
            )));
        }
        Ok(())
    }

    fn build(
        model_path: &Path,
        spec: ModelSpec,
        config: EngineConfig,
        n_threads: usize,
    ) -> Result<Self, LTEmbedError> {
        config.validate(spec.raw_embedding_dimension)?;
        Self::validate_threads(n_threads)?;

        // Opened before llama.cpp opens the same path, and kept open, so that the embedding
        // space names the file the backend loaded even if the path is replaced later.
        let model_file = ModelFile::open(model_path)?;
        // Loading checks that the GGUF declares mean pooling, the only pooling the bundle
        // metadata accepts (`ModelSpec::from_build_info`), and that its tokenization adds
        // BOS and EOS.
        let backend = LlamaBackend::load(
            model_path,
            spec.raw_embedding_dimension,
            spec.max_length,
            n_threads,
        )?;
        // llama.cpp opened the path again. A file renamed over it in between may be the one
        // the backend loaded, and the space would name another file.
        model_file.check_still_at_path()?;

        Ok(Self::with_backend(
            Box::new(backend),
            spec,
            config,
            model_file,
        ))
    }

    fn with_backend(
        backend: Box<dyn EmbeddingBackend>,
        spec: ModelSpec,
        config: EngineConfig,
        model_file: ModelFile,
    ) -> Self {
        let max_input_bytes = backend.max_input_bytes(spec.max_length);
        Self {
            backend,
            spec,
            config,
            max_input_bytes,
            model_file,
            embedding_space: OnceLock::new(),
        }
    }

    /// The vector space this engine embeds into: the sha256 of the `model.gguf` it loaded, the
    /// query and document prefixes it adds, and its [`EngineConfig`]. Vectors from engines
    /// whose spaces are equal are comparable. Store the space's string form beside an index
    /// and compare it with the engine's before adding to or searching the index
    /// (`docs/model-migration.md` § "Moving an index").
    ///
    /// The first call reads and hashes the model file, through the handle the engine opened
    /// when it loaded the file: about 0.11 s for the release GGUF on x86-64-v3. Later calls
    /// return the same value without reading the file again.
    ///
    /// # Errors
    ///
    /// Returns `LTEmbedError::Io` if the model file cannot be read. A later call tries again.
    pub fn embedding_space(&self) -> Result<&EmbeddingSpace, LTEmbedError> {
        if let Some(space) = self.embedding_space.get() {
            return Ok(space);
        }
        // Threads that make the first call at the same time each hash the file; the first
        // result is kept, and the others are equal to it.
        let model_sha256 = self.model_file.sha256()?;
        Ok(self
            .embedding_space
            .get_or_init(|| EmbeddingSpace::new(model_sha256, &self.spec, self.config)))
    }

    /// The longest input [`EmbeddingEngine::embed`] accepts, in tokens as
    /// [`EmbeddingEngine::token_count`] counts them: the bundle's `max_length` from
    /// `build-info.json`, at most [`MAX_LENGTH`]. An input of more tokens is rejected with
    /// `LTEmbedError::InputTooLong`, never truncated.
    pub fn max_length(&self) -> usize {
        self.spec.max_length
    }

    /// The number of tokens `input` takes: its text with the query or document prefix the
    /// engine adds, tokenized with the GGUF's own vocabulary, BOS and EOS included. This is the
    /// count [`EmbeddingEngine::embed`] compares with [`EmbeddingEngine::max_length`], and it
    /// is returned for an input over that limit too, so that text can be split to fit before
    /// it is embedded: `embed` rejects an input of more than `max_length()` tokens with
    /// `InputTooLong { tokens: Some(count), .. }`, where `count` is this value. An input too
    /// large to fit at all has no count: it is the `InputTooLong` error under "Errors".
    ///
    /// Takes no lock: it does not wait for another call's inference on the same engine
    /// (`docs/architecture.md` § "Concurrency").
    ///
    /// # Errors
    ///
    /// Returns `LTEmbedError::InputTooLong { tokens: None, bytes, max }` without tokenizing
    /// `input` if its prefixed text is over the engine's byte bound: no text of `bytes` bytes
    /// fits in `max` tokens of the vocabulary, and tokenizing one costs time and memory in
    /// proportion to its size (`docs/architecture.md` § "Inputs over the byte bound"). `embed`
    /// rejects such an input the same way. Returns `LTEmbedError::Tokenization` if the text
    /// cannot be tokenized.
    pub fn token_count(&self, input: EmbeddingInput<'_>) -> Result<usize, LTEmbedError> {
        self.check_input_size(input)?;
        let text = prefixed_text(input, &self.spec);
        Ok(self.backend.tokenize(&text)?.len())
    }

    /// Rejects `input` if its prefixed text is too long to fit in the bundle's `max_length`
    /// tokens, without building or tokenizing that text.
    fn check_input_size(&self, input: EmbeddingInput<'_>) -> Result<(), LTEmbedError> {
        let bytes = prefix(input.kind, &self.spec)
            .len()
            .saturating_add(input.text.len());
        check_input_bytes(bytes, self.max_input_bytes, self.spec.max_length)
    }

    /// The GGUF vocabulary's token ids for `text`, BOS and EOS included. A text of more than
    /// the bundle's `max_length` tokens is an error; it is never truncated.
    fn tokenize(&self, text: &str) -> Result<Vec<u32>, LTEmbedError> {
        let ids = self.backend.tokenize(text)?;
        if ids.len() > self.spec.max_length {
            return Err(LTEmbedError::InputTooLong {
                tokens: Some(ids.len()),
                bytes: text.len(),
                max: self.spec.max_length,
            });
        }
        Ok(ids)
    }

    pub fn embed(&self, input: EmbeddingInput<'_>) -> Result<Vec<f32>, LTEmbedError> {
        let embeddings = self.embed_batch(&[input])?;
        embeddings
            .into_iter()
            .next()
            .ok_or(LTEmbedError::Inference(InferenceError::Internal(
                "expected one embedding".into(),
            )))
    }

    pub fn embed_batch(
        &self,
        inputs: &[EmbeddingInput<'_>],
    ) -> Result<Vec<Vec<f32>>, LTEmbedError> {
        let (embeddings, _) = self.embed_batch_impl(inputs, false)?;
        Ok(embeddings)
    }

    pub fn embed_batch_profiled(
        &self,
        inputs: &[EmbeddingInput<'_>],
    ) -> Result<(Vec<Vec<f32>>, EmbedBatchProfile), LTEmbedError> {
        let (embeddings, profile) = self.embed_batch_impl(inputs, true)?;
        let profile = profile.ok_or(LTEmbedError::Inference(InferenceError::Internal(
            "profiling requested but no profile was collected".into(),
        )))?;
        Ok((embeddings, profile))
    }

    fn embed_batch_impl(
        &self,
        inputs: &[EmbeddingInput<'_>],
        collect_profile: bool,
    ) -> Result<(Vec<Vec<f32>>, Option<EmbedBatchProfile>), LTEmbedError> {
        if inputs.is_empty() {
            let profile = collect_profile.then(EmbedBatchProfile::empty);
            return Ok((Vec::new(), profile));
        }

        let overall_start = Instant::now();

        for input in inputs {
            self.check_input_size(*input)?;
        }

        let prefix_start = Instant::now();
        let prefixed_inputs = inputs
            .iter()
            .copied()
            .map(|input| prefixed_text(input, &self.spec))
            .collect::<Vec<_>>();
        let prefix_ms = prefix_start.elapsed().as_secs_f64() * 1_000.0;

        let tokenize_start = Instant::now();
        let encoded = prefixed_inputs
            .iter()
            .map(|text| self.tokenize(text))
            .collect::<Result<Vec<_>, _>>()?;
        let tokenize_ms = tokenize_start.elapsed().as_secs_f64() * 1_000.0;

        let batch_size = encoded.len();
        let sequence_length = encoded.iter().map(Vec::len).max().unwrap_or(0);

        let (raw_embeddings, backend_profile) = self.backend.embed(&encoded, collect_profile)?;

        let raw_dim = self.backend.raw_embedding_dimension();
        let postprocess_start = Instant::now();
        let embeddings = raw_embeddings
            .iter()
            .map(|raw| postprocess_embedding(raw, raw_dim, self.config))
            .collect::<Result<Vec<_>, _>>()?;
        let postprocess_ms = postprocess_start.elapsed().as_secs_f64() * 1_000.0;
        let total_ms = overall_start.elapsed().as_secs_f64() * 1_000.0;

        // A backend asked for a profile that returns none leaves this call without one, which
        // `embed_batch_profiled` reports as an internal error, instead of a run time of zero.
        let profile = backend_profile.map(|backend| EmbedBatchProfile {
            batch_size,
            sequence_length,
            prefix_ms,
            tokenize_ms,
            run_ms: backend.run_ms,
            postprocess_ms,
            total_ms,
        });

        Ok((embeddings, profile))
    }
}

/// `InputTooLong`, without a token count, if a text of `bytes` bytes is over
/// `max_input_bytes`: it cannot fit in `max_length` tokens, so it is not tokenized.
fn check_input_bytes(
    bytes: usize,
    max_input_bytes: usize,
    max_length: usize,
) -> Result<(), LTEmbedError> {
    if bytes > max_input_bytes {
        return Err(LTEmbedError::InputTooLong {
            tokens: None,
            bytes,
            max: max_length,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use super::backend::BackendRunProfile;
    use super::*;

    /// Bytes each token of [`CountingBackend`] covers at most.
    const TOKEN_BYTES: usize = 4;

    /// A backend whose tokens cover up to `TOKEN_BYTES` bytes each, and that counts the texts
    /// it tokenizes.
    struct CountingBackend {
        tokenized: Arc<AtomicUsize>,
    }

    impl EmbeddingBackend for CountingBackend {
        fn raw_embedding_dimension(&self) -> usize {
            RAW_EMBEDDING_DIMENSION
        }

        fn tokenize(&self, text: &str) -> Result<Vec<u32>, LTEmbedError> {
            self.tokenized.fetch_add(1, Ordering::SeqCst);
            // BOS, the text, EOS.
            Ok(vec![0; text.len().div_ceil(TOKEN_BYTES) + 2])
        }

        fn max_input_bytes(&self, max_tokens: usize) -> usize {
            max_tokens * TOKEN_BYTES
        }

        fn embed(
            &self,
            tokenized: &[Vec<u32>],
            _collect_profile: bool,
        ) -> Result<(Vec<Vec<f32>>, Option<BackendRunProfile>), LTEmbedError> {
            let raw = vec![1.0; RAW_EMBEDDING_DIMENSION];
            Ok((vec![raw; tokenized.len()], None))
        }
    }

    const MAX_TOKENS: usize = 16;

    /// A [`ModelFile`] holding `contents`. Its path is removed at once: the engine reads the
    /// file through the handle it keeps.
    fn model_file(contents: &[u8]) -> ModelFile {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "ltembed-engine-unit-tests-{}-{}.gguf",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        fs::write(&path, contents).unwrap();
        let file = ModelFile::open(&path).unwrap();
        fs::remove_file(&path).unwrap();
        file
    }

    /// An engine of `MAX_TOKENS` tokens on a [`CountingBackend`], whose byte bound is 64
    /// bytes, and the count of texts it has tokenized.
    fn counting_engine() -> (EmbeddingEngine, Arc<AtomicUsize>) {
        let spec = ModelSpec {
            max_length: MAX_TOKENS,
            ..ModelSpec::defaults()
        };
        let tokenized = Arc::new(AtomicUsize::new(0));
        let backend = Box::new(CountingBackend {
            tokenized: Arc::clone(&tokenized),
        });
        let engine = EmbeddingEngine::with_backend(
            backend,
            spec,
            EngineConfig::default(),
            model_file(b"abc"),
        );
        (engine, tokenized)
    }

    /// A document text whose prefixed form is `bytes` bytes long.
    fn document_text(bytes: usize) -> String {
        "x".repeat(bytes - DOCUMENT_PREFIX.len())
    }

    /// A text over the bound fails the whole batch before any input of it is tokenized, and
    /// the error reports its prefixed size instead of a token count.
    #[test]
    fn test_text_over_the_byte_bound_is_rejected_before_tokenizing() {
        let (engine, tokenized) = counting_engine();
        let bound = MAX_TOKENS * TOKEN_BYTES;
        let over = document_text(bound + 1);
        let batch = [
            EmbeddingInput::query("fits"),
            EmbeddingInput::document(&over),
        ];
        match engine.embed_batch(&batch) {
            Err(LTEmbedError::InputTooLong {
                tokens: None,
                bytes,
                max,
            }) => assert_eq!((bytes, max), (bound + 1, MAX_TOKENS)),
            other => panic!("expected InputTooLong without a token count, got {other:?}"),
        }
        assert_eq!(tokenized.load(Ordering::SeqCst), 0);
    }

    /// A text at the bound is tokenized; one over `max_length` tokens reports its count.
    #[test]
    fn test_text_at_the_byte_bound_is_tokenized() {
        let (engine, tokenized) = counting_engine();
        let bound = MAX_TOKENS * TOKEN_BYTES;
        let at = document_text(bound);
        match engine.embed(EmbeddingInput::document(&at)) {
            Err(LTEmbedError::InputTooLong { tokens, bytes, max }) => {
                assert_eq!(
                    (tokens, bytes, max),
                    (Some(MAX_TOKENS + 2), bound, MAX_TOKENS)
                );
            }
            other => panic!("expected InputTooLong with a token count, got {other:?}"),
        }
        assert_eq!(tokenized.load(Ordering::SeqCst), 1);

        let fits = document_text(bound - 2 * TOKEN_BYTES);
        let embedding = engine.embed(EmbeddingInput::document(&fits)).unwrap();
        assert_eq!(embedding.len(), EMBEDDING_DIMENSION);
        assert_eq!(tokenized.load(Ordering::SeqCst), 2);
    }

    /// `token_count` counts the prefixed text's tokens, BOS and EOS included: the count `embed`
    /// compares with `max_length`, which `max_length()` reports. A text of more than
    /// `max_length` tokens is counted too, and `embed` rejects it naming that count.
    #[test]
    fn test_token_count_is_the_count_embed_compares_with_max_length() {
        let (engine, tokenized) = counting_engine();
        assert_eq!(engine.max_length(), MAX_TOKENS);

        let tokens_of = |text: &str| text.len().div_ceil(TOKEN_BYTES) + 2;
        assert_eq!(
            engine.token_count(EmbeddingInput::query("hello")).unwrap(),
            tokens_of(&format!("{QUERY_PREFIX}hello"))
        );
        assert_eq!(
            engine
                .token_count(EmbeddingInput::document("hello"))
                .unwrap(),
            tokens_of(&format!("{DOCUMENT_PREFIX}hello"))
        );
        assert_eq!(tokenized.load(Ordering::SeqCst), 2);

        let at_bound = document_text(MAX_TOKENS * TOKEN_BYTES);
        let count = engine
            .token_count(EmbeddingInput::document(&at_bound))
            .unwrap();
        assert_eq!(count, MAX_TOKENS + 2);
        match engine.embed(EmbeddingInput::document(&at_bound)) {
            Err(LTEmbedError::InputTooLong { tokens, .. }) => assert_eq!(tokens, Some(count)),
            other => panic!("expected InputTooLong with a token count, got {other:?}"),
        }
    }

    /// A text over the byte bound cannot fit, so `token_count` rejects it as `embed` does,
    /// without tokenizing it.
    #[test]
    fn test_token_count_rejects_a_text_over_the_byte_bound_without_tokenizing() {
        let (engine, tokenized) = counting_engine();
        let bound = MAX_TOKENS * TOKEN_BYTES;
        let over = document_text(bound + 1);
        match engine.token_count(EmbeddingInput::document(&over)) {
            Err(LTEmbedError::InputTooLong {
                tokens: None,
                bytes,
                max,
            }) => assert_eq!((bytes, max), (bound + 1, MAX_TOKENS)),
            other => panic!("expected InputTooLong without a token count, got {other:?}"),
        }
        assert_eq!(tokenized.load(Ordering::SeqCst), 0);
    }

    /// [`CountingBackend`] measures nothing, so a profiled call fails instead of reporting a
    /// run time of zero. An empty batch never reaches the backend and is profiled as empty.
    #[test]
    fn test_profile_without_the_backends_run_time_is_an_error() {
        let (engine, _) = counting_engine();
        let result = engine.embed_batch_profiled(&[EmbeddingInput::query("hello")]);
        match result.map(|(_, profile)| profile) {
            Err(LTEmbedError::Inference(InferenceError::Internal(_))) => {}
            other => panic!("expected Inference(Internal), got {other:?}"),
        }
        assert_eq!(
            engine.embed_batch_profiled(&[]).unwrap(),
            (Vec::new(), EmbedBatchProfile::empty())
        );
    }

    /// The space names the contents of the model file the engine holds, the prefixes it adds
    /// and its config. The first call's result is kept.
    #[test]
    fn test_embedding_space_names_the_model_file_prefixes_and_config() {
        let spec = ModelSpec {
            query_prefix: "Query: ".into(),
            document_prefix: "Document: ".into(),
            ..ModelSpec::defaults()
        };
        let config = EngineConfig {
            output_dimension: 256,
            l2_normalize: false,
        };
        let backend = Box::new(CountingBackend {
            tokenized: Arc::default(),
        });
        let engine = EmbeddingEngine::with_backend(backend, spec, config, model_file(b"abc"));

        let space = engine.embedding_space().unwrap();
        assert_eq!(
            space.model_sha256,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            (space.query_prefix.as_str(), space.document_prefix.as_str()),
            ("Query: ", "Document: ")
        );
        assert_eq!((space.output_dimension, space.l2_normalize), (256, false));
        assert!(std::ptr::eq(space, engine.embedding_space().unwrap()));
    }
}
