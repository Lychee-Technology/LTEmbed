use std::path::Path;
use std::time::Instant;

use crate::error::{InferenceError, LTEmbedError, ModelLoadError};
use crate::traits::tokenizer::HFTokenizer;

mod backend;
mod bundle;
mod config;
mod inference;
mod input;
mod llama;

pub use config::EngineConfig;
pub use inference::EmbedBatchProfile;
pub use input::{EmbeddingInput, EmbeddingInputKind};

use backend::EmbeddingBackend;
use bundle::{require_file, ModelSpec};
use inference::{postprocess_embedding, prefixed_text};
use llama::LlamaBackend;

pub const RAW_EMBEDDING_DIMENSION: usize = 768;
pub const EMBEDDING_DIMENSION: usize = 512;
pub const MAX_LENGTH: usize = 8192;
pub const QUERY_PREFIX: &str = "Query: ";
pub const DOCUMENT_PREFIX: &str = "Document: ";

const MODEL_FILE: &str = "model.gguf";
const TOKENIZER_FILE: &str = "tokenizer.json";
const BUILD_INFO_FILE: &str = "build-info.json";

/// A loaded embedding model. Backed by an [`EmbeddingBackend`] (currently llama.cpp/GGUF);
/// owns the shared prefix → tokenize → backend → truncate/normalize pipeline.
pub struct EmbeddingEngine {
    backend: Box<dyn EmbeddingBackend>,
    tokenizer: HFTokenizer,
    spec: ModelSpec,
    config: EngineConfig,
}

impl std::fmt::Debug for EmbeddingEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddingEngine").finish_non_exhaustive()
    }
}

impl EmbeddingEngine {
    /// Load from a bundle directory containing `model.gguf`, `tokenizer.json`, and
    /// `build-info.json`, using a single inference thread.
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
        let tokenizer_path = bundle_dir.join(TOKENIZER_FILE);
        let build_info_path = bundle_dir.join(BUILD_INFO_FILE);

        require_file(&model_path, "GGUF model")?;
        require_file(&tokenizer_path, "tokenizer")?;
        require_file(&build_info_path, "build-info metadata")?;

        let spec = ModelSpec::from_build_info(&build_info_path)?;
        Self::build(&model_path, &tokenizer_path, spec, config, n_threads)
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
        tokenizer_path: &Path,
        spec: ModelSpec,
        config: EngineConfig,
        n_threads: usize,
    ) -> Result<Self, LTEmbedError> {
        config.validate(spec.raw_embedding_dimension)?;
        Self::validate_threads(n_threads)?;

        let tokenizer = HFTokenizer::from_file(&tokenizer_path.to_string_lossy())?;
        let backend = LlamaBackend::load(
            model_path,
            spec.raw_embedding_dimension,
            spec.max_length,
            n_threads,
        )?;

        Ok(Self {
            backend: Box::new(backend),
            tokenizer,
            spec,
            config,
        })
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

        let prefix_start = Instant::now();
        let prefixed_inputs = inputs
            .iter()
            .copied()
            .map(|input| prefixed_text(input, &self.spec))
            .collect::<Vec<_>>();
        let prefix_ms = prefix_start.elapsed().as_secs_f64() * 1_000.0;

        let tokenize_start = Instant::now();
        let encoded = self
            .tokenizer
            .encode_batch(&prefixed_inputs, self.spec.max_length)?;
        let tokenize_ms = tokenize_start.elapsed().as_secs_f64() * 1_000.0;

        let batch_size = encoded.len();
        let sequence_length = encoded
            .iter()
            .map(|item| item.input_ids.len())
            .max()
            .unwrap_or(0);

        let (raw_embeddings, backend_profile) = self.backend.embed(&encoded, collect_profile)?;

        let raw_dim = self.backend.raw_embedding_dimension();
        let postprocess_start = Instant::now();
        let embeddings = raw_embeddings
            .iter()
            .map(|raw| postprocess_embedding(raw, raw_dim, self.config))
            .collect::<Result<Vec<_>, _>>()?;
        let postprocess_ms = postprocess_start.elapsed().as_secs_f64() * 1_000.0;
        let total_ms = overall_start.elapsed().as_secs_f64() * 1_000.0;

        let profile = collect_profile.then(|| {
            let backend = backend_profile.unwrap_or_default();
            EmbedBatchProfile {
                batch_size,
                sequence_length,
                prefix_ms,
                tokenize_ms,
                tensorize_ms: backend.tensorize_ms,
                run_ms: backend.run_ms,
                extract_ms: backend.extract_ms,
                postprocess_ms,
                total_ms,
            }
        });

        Ok((embeddings, profile))
    }
}
