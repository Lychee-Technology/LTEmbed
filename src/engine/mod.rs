use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;

use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::TensorRef;

use crate::error::{InferenceError, LTEmbedError, ModelLoadError};
use crate::traits::tokenizer::HFTokenizer;

mod bundle;
mod config;
mod inference;
mod input;
mod ort_init;
mod session_io;

pub use config::OnnxEngineConfig;
pub use inference::EmbedBatchProfile;
pub use input::{EmbeddingInput, EmbeddingInputKind};

use bundle::{require_file, resolve_dylib_path, ModelSpec};
use inference::{pack_batch, pool_last_token, prefixed_text, validate_hidden_shape};
use ort_init::ensure_ort_initialized;
use session_io::{effective_sequence_length, SessionIo};

pub const RAW_EMBEDDING_DIMENSION: usize = 768;
pub const EMBEDDING_DIMENSION: usize = 512;
pub const MAX_LENGTH: usize = 8192;
pub const QUERY_PREFIX: &str = "Query: ";
pub const DOCUMENT_PREFIX: &str = "Document: ";

const TOKENIZER_FILE: &str = "tokenizer.json";
const BUILD_INFO_FILE: &str = "build-info.json";

pub struct OnnxEngine {
    session: Mutex<Session>,
    tokenizer: HFTokenizer,
    io: SessionIo,
    spec: ModelSpec,
    config: OnnxEngineConfig,
}

impl std::fmt::Debug for OnnxEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnnxEngine").finish_non_exhaustive()
    }
}

impl OnnxEngine {
    pub fn new(model_path: &str, tokenizer_path: &str) -> Result<Self, LTEmbedError> {
        let model_path = Path::new(model_path);
        let tokenizer_path = Path::new(tokenizer_path);
        require_file(model_path, "model")?;
        require_file(tokenizer_path, "tokenizer")?;

        let spec = ModelSpec::jina_defaults();
        let config = OnnxEngineConfig::default();
        Self::build(model_path, tokenizer_path, None, spec, config, 1)
    }

    pub fn from_bundle_dir(
        bundle_dir: impl AsRef<Path>,
        model_path: impl AsRef<Path>,
        config: OnnxEngineConfig,
    ) -> Result<Self, LTEmbedError> {
        Self::from_bundle_dir_with_intra_threads(bundle_dir, model_path, config, 1)
    }

    pub fn from_bundle_dir_with_intra_threads(
        bundle_dir: impl AsRef<Path>,
        model_path: impl AsRef<Path>,
        config: OnnxEngineConfig,
        intra_threads: usize,
    ) -> Result<Self, LTEmbedError> {
        let bundle_dir = bundle_dir.as_ref();
        let model_path = model_path.as_ref();
        let tokenizer_path = bundle_dir.join(TOKENIZER_FILE);
        let build_info_path = bundle_dir.join(BUILD_INFO_FILE);

        require_file(model_path, "ORT model")?;
        require_file(&tokenizer_path, "tokenizer")?;
        require_file(&build_info_path, "build-info metadata")?;

        let dylib_path = resolve_dylib_path(bundle_dir);

        let spec = ModelSpec::from_build_info(&build_info_path)?;
        Self::build(
            model_path,
            &tokenizer_path,
            dylib_path.as_deref(),
            spec,
            config,
            intra_threads,
        )
    }

    fn validate_intra_threads(intra_threads: usize) -> Result<(), LTEmbedError> {
        if intra_threads == 0 {
            return Err(LTEmbedError::ModelLoad(ModelLoadError::Config(
                "intra_threads must be greater than zero".into(),
            )));
        }
        Ok(())
    }

    fn build(
        model_path: &Path,
        tokenizer_path: &Path,
        dylib_path: Option<&Path>,
        spec: ModelSpec,
        config: OnnxEngineConfig,
        intra_threads: usize,
    ) -> Result<Self, LTEmbedError> {
        config.validate(spec.raw_embedding_dimension)?;
        Self::validate_intra_threads(intra_threads)?;

        ensure_ort_initialized(dylib_path)?;
        let tokenizer = HFTokenizer::from_file(&tokenizer_path.to_string_lossy())?;
        let session = Session::builder()
            .map_err(|err| {
                LTEmbedError::ModelLoad(ModelLoadError::Runtime(format!(
                    "Failed to create ORT session builder: {err}"
                )))
            })?
            .with_intra_threads(intra_threads)
            .map_err(|err| {
                LTEmbedError::ModelLoad(ModelLoadError::Runtime(format!(
                    "Failed to configure ORT session: {err}"
                )))
            })?
            .with_optimization_level(GraphOptimizationLevel::Disable)
            .map_err(|err| {
                LTEmbedError::ModelLoad(ModelLoadError::Runtime(format!(
                    "Failed to disable ORT runtime optimization: {err}"
                )))
            })?
            .commit_from_file(model_path)
            .map_err(|err| {
                LTEmbedError::ModelLoad(ModelLoadError::Runtime(format!(
                    "Failed to load ORT model: {err}"
                )))
            })?;
        let io = SessionIo::from_session(&session, spec.raw_embedding_dimension)?;

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            io,
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
        let batch_max_seq_len = encoded
            .iter()
            .map(|item| item.input_ids.len())
            .max()
            .unwrap_or(0);
        let seq_len = effective_sequence_length(self.io.sequence_length(), batch_max_seq_len)?;

        let tensorize_start = Instant::now();
        let (input_ids, attention_mask) = pack_batch(&encoded, seq_len);

        let input_ids_tensor = TensorRef::from_array_view(([batch_size, seq_len], &input_ids[..]))
            .map_err(|err| {
                LTEmbedError::Inference(InferenceError::Tensor(format!(
                    "Failed to convert input_ids tensor: {err}"
                )))
            })?;
        let attention_mask_tensor =
            TensorRef::from_array_view(([batch_size, seq_len], &attention_mask[..])).map_err(
                |err| {
                    LTEmbedError::Inference(InferenceError::Tensor(format!(
                        "Failed to convert attention_mask tensor: {err}"
                    )))
                },
            )?;
        let tensorize_ms = tensorize_start.elapsed().as_secs_f64() * 1_000.0;

        let mut session = self
            .session
            .lock()
            .map_err(|_| LTEmbedError::Inference(InferenceError::MutexPoisoned))?;
        let run_start = Instant::now();
        let outputs = session
            .run(ort::inputs! {
                self.io.input_ids_name() => input_ids_tensor,
                self.io.attention_mask_name() => attention_mask_tensor,
            })
            .map_err(|err| {
                LTEmbedError::Inference(InferenceError::OrtRun(format!(
                    "ORT inference failed: {err}"
                )))
            })?;
        let run_ms = run_start.elapsed().as_secs_f64() * 1_000.0;
        let extract_start = Instant::now();
        let (hidden_shape, hidden_data) = outputs[self.io.output_name()]
            .try_extract_tensor::<f32>()
            .map_err(|err| {
                LTEmbedError::Inference(InferenceError::Tensor(format!(
                    "Failed to extract ORT output tensor: {err}"
                )))
            })?;
        validate_hidden_shape(
            hidden_shape,
            batch_size,
            seq_len,
            self.spec.raw_embedding_dimension,
        )?;
        let extract_ms = extract_start.elapsed().as_secs_f64() * 1_000.0;

        let postprocess_start = Instant::now();
        let embeddings = pool_last_token(
            hidden_data,
            &attention_mask,
            batch_size,
            seq_len,
            self.spec.raw_embedding_dimension,
            self.config,
        )?;
        let postprocess_ms = postprocess_start.elapsed().as_secs_f64() * 1_000.0;
        let total_ms = overall_start.elapsed().as_secs_f64() * 1_000.0;
        let profile = collect_profile.then_some(EmbedBatchProfile {
            batch_size,
            sequence_length: seq_len,
            prefix_ms,
            tokenize_ms,
            tensorize_ms,
            run_ms,
            extract_ms,
            postprocess_ms,
            total_ms,
        });

        Ok((embeddings, profile))
    }
}
