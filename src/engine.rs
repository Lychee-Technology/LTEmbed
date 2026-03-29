use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::TensorRef;
use serde::Deserialize;

use crate::error::LTEmbedError;
use crate::traits::tokenizer::HFTokenizer;

pub const RAW_EMBEDDING_DIMENSION: usize = 768;
pub const EMBEDDING_DIMENSION: usize = 512;
pub const MAX_LENGTH: usize = 8192;
pub const QUERY_PREFIX: &str = "Query: ";
pub const DOCUMENT_PREFIX: &str = "Document: ";

const MODEL_FILE: &str = "model.ort";
const TOKENIZER_FILE: &str = "tokenizer.json";
const ORT_DYLIB_FILE: &str = "libonnxruntime.so";
const BUILD_INFO_FILE: &str = "build-info.json";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingInputKind {
    Query,
    Document,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmbeddingInput<'a> {
    pub text: &'a str,
    pub kind: EmbeddingInputKind,
}

impl<'a> EmbeddingInput<'a> {
    pub fn query(text: &'a str) -> Self {
        Self {
            text,
            kind: EmbeddingInputKind::Query,
        }
    }

    pub fn document(text: &'a str) -> Self {
        Self {
            text,
            kind: EmbeddingInputKind::Document,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OnnxEngineConfig {
    pub output_dimension: usize,
    pub l2_normalize: bool,
}

impl Default for OnnxEngineConfig {
    fn default() -> Self {
        Self {
            output_dimension: EMBEDDING_DIMENSION,
            l2_normalize: true,
        }
    }
}

#[derive(Debug)]
struct SessionIo {
    input_ids: String,
    attention_mask: String,
    last_hidden_state: String,
    sequence_length: Option<usize>,
}

#[derive(Debug, Clone)]
struct ModelSpec {
    query_prefix: String,
    document_prefix: String,
    raw_embedding_dimension: usize,
    max_length: usize,
}

#[derive(Debug, Deserialize)]
struct BuildInfo {
    target_id: String,
    model_metadata: BuildMetadata,
}

#[derive(Debug, Deserialize)]
struct BuildMetadata {
    #[allow(dead_code)]
    model_format: Option<String>,
    pooling: String,
    input_kind: String,
    query_prefix: String,
    document_prefix: String,
    raw_embedding_dimension: usize,
    #[allow(dead_code)]
    output_embedding_dimension: usize,
    max_length: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum OrtInitSource {
    System,
    DynamicLibrary(PathBuf),
}

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
        let spec = ModelSpec::jina_defaults();
        let config = OnnxEngineConfig::default();
        Self::build(
            Path::new(model_path),
            Path::new(tokenizer_path),
            None,
            spec,
            config,
        )
    }

    pub fn from_bundle_dir(
        bundle_dir: impl AsRef<Path>,
        config: OnnxEngineConfig,
    ) -> Result<Self, LTEmbedError> {
        let bundle_dir = bundle_dir.as_ref();
        let model_path = bundle_dir.join(MODEL_FILE);
        let tokenizer_path = bundle_dir.join(TOKENIZER_FILE);
        let dylib_path = bundle_dir.join(ORT_DYLIB_FILE);
        let build_info_path = bundle_dir.join(BUILD_INFO_FILE);

        require_file(&model_path, "ORT model")?;
        require_file(&tokenizer_path, "tokenizer")?;
        require_file(&dylib_path, "ORT dynamic library")?;
        require_file(&build_info_path, "build-info metadata")?;

        let spec = ModelSpec::from_build_info(&build_info_path)?;
        Self::build(
            &model_path,
            &tokenizer_path,
            Some(&dylib_path),
            spec,
            config,
        )
    }

    fn build(
        model_path: &Path,
        tokenizer_path: &Path,
        dylib_path: Option<&Path>,
        spec: ModelSpec,
        config: OnnxEngineConfig,
    ) -> Result<Self, LTEmbedError> {
        require_file(model_path, "model")?;
        require_file(tokenizer_path, "tokenizer")?;
        config.validate(spec.raw_embedding_dimension)?;

        ensure_ort_initialized(dylib_path)?;
        let tokenizer = HFTokenizer::from_file(&tokenizer_path.to_string_lossy())?;
        let session = Session::builder()
            .map_err(|err| {
                LTEmbedError::ModelLoad(format!("Failed to create ORT session builder: {err}"))
            })?
            .with_intra_threads(1)
            .map_err(|err| {
                LTEmbedError::ModelLoad(format!("Failed to configure ORT session: {err}"))
            })?
            .with_optimization_level(GraphOptimizationLevel::Disable)
            .map_err(|err| {
                LTEmbedError::ModelLoad(format!(
                    "Failed to disable ORT runtime optimization: {err}"
                ))
            })?
            .commit_from_file(model_path)
            .map_err(|err| LTEmbedError::ModelLoad(format!("Failed to load ORT model: {err}")))?;
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
            .ok_or_else(|| LTEmbedError::Inference("expected one embedding".into()))
    }

    pub fn embed_batch(
        &self,
        inputs: &[EmbeddingInput<'_>],
    ) -> Result<Vec<Vec<f32>>, LTEmbedError> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let prefixed_inputs = inputs
            .iter()
            .copied()
            .map(|input| prefixed_text(input, &self.spec))
            .collect::<Vec<_>>();
        let encoded = self
            .tokenizer
            .encode_batch(&prefixed_inputs, self.spec.max_length)?;
        let batch_size = encoded.len();
        let batch_max_seq_len = encoded
            .iter()
            .map(|item| item.input_ids.len())
            .max()
            .unwrap_or(0);
        let seq_len = effective_sequence_length(self.io.sequence_length, batch_max_seq_len)?;

        let mut input_ids = vec![0_i64; batch_size * seq_len];
        let mut attention_mask = vec![0_i64; batch_size * seq_len];

        for (batch_idx, item) in encoded.iter().enumerate() {
            for (token_idx, (&token, &mask)) in item
                .input_ids
                .iter()
                .zip(item.attention_mask.iter())
                .enumerate()
            {
                let offset = batch_idx * seq_len + token_idx;
                input_ids[offset] = token as i64;
                attention_mask[offset] = mask as i64;
            }
        }

        let input_ids_tensor = TensorRef::from_array_view(([batch_size, seq_len], &input_ids[..]))
            .map_err(|err| {
                LTEmbedError::Inference(format!("Failed to convert input_ids tensor: {err}"))
            })?;
        let attention_mask_tensor = TensorRef::from_array_view((
            [batch_size, seq_len],
            &attention_mask[..],
        ))
        .map_err(|err| {
            LTEmbedError::Inference(format!("Failed to convert attention_mask tensor: {err}"))
        })?;

        let mut session = self
            .session
            .lock()
            .map_err(|_| LTEmbedError::Inference("ORT session mutex poisoned".into()))?;
        let outputs = session
            .run(ort::inputs! {
                self.io.input_ids.as_str() => input_ids_tensor,
                self.io.attention_mask.as_str() => attention_mask_tensor,
            })
            .map_err(|err| LTEmbedError::Inference(format!("ORT inference failed: {err}")))?;
        let (hidden_shape, hidden_data) = outputs[self.io.last_hidden_state.as_str()]
            .try_extract_tensor::<f32>()
            .map_err(|err| {
                LTEmbedError::Inference(format!("Failed to extract ORT output tensor: {err}"))
            })?;
        if hidden_shape.len() != 3 {
            return Err(LTEmbedError::Inference(format!(
                "expected rank-3 hidden states, got shape {hidden_shape:?}"
            )));
        }
        if hidden_shape[0] as usize != batch_size || hidden_shape[1] as usize != seq_len {
            return Err(LTEmbedError::Inference(format!(
                "unexpected hidden state shape {hidden_shape:?}, expected [{batch_size}, {seq_len}, {}]",
                self.spec.raw_embedding_dimension
            )));
        }
        if hidden_shape[2] as usize != self.spec.raw_embedding_dimension {
            return Err(LTEmbedError::Inference(format!(
                "expected raw embedding dimension {}, got {}",
                self.spec.raw_embedding_dimension, hidden_shape[2]
            )));
        }

        let mut embeddings = Vec::with_capacity(batch_size);
        for batch_idx in 0..batch_size {
            let mask_start = batch_idx * seq_len;
            let mask_end = mask_start + seq_len;
            let mask_slice = &attention_mask[mask_start..mask_end];
            let last_token_idx =
                mask_slice
                    .iter()
                    .rposition(|mask| *mask == 1)
                    .ok_or_else(|| {
                        LTEmbedError::Inference("attention mask contains only padding".into())
                    })?;
            let hidden_offset =
                (batch_idx * seq_len + last_token_idx) * self.spec.raw_embedding_dimension;
            let raw =
                &hidden_data[hidden_offset..hidden_offset + self.spec.raw_embedding_dimension];
            embeddings.push(postprocess_embedding(
                raw,
                self.spec.raw_embedding_dimension,
                self.config,
            )?);
        }

        Ok(embeddings)
    }
}

impl OnnxEngineConfig {
    fn validate(self, raw_embedding_dimension: usize) -> Result<(), LTEmbedError> {
        if self.output_dimension == 0 {
            return Err(LTEmbedError::ModelLoad(
                "output_dimension must be greater than zero".into(),
            ));
        }
        if self.output_dimension > raw_embedding_dimension {
            return Err(LTEmbedError::ModelLoad(format!(
                "output_dimension {} exceeds raw embedding dimension {}",
                self.output_dimension, raw_embedding_dimension
            )));
        }
        Ok(())
    }
}

impl ModelSpec {
    fn jina_defaults() -> Self {
        Self {
            query_prefix: QUERY_PREFIX.to_string(),
            document_prefix: DOCUMENT_PREFIX.to_string(),
            raw_embedding_dimension: RAW_EMBEDDING_DIMENSION,
            max_length: MAX_LENGTH,
        }
    }

    fn from_build_info(path: &Path) -> Result<Self, LTEmbedError> {
        let raw = fs::read_to_string(path).map_err(|err| {
            LTEmbedError::ModelLoad(format!(
                "Failed to read build-info metadata at {}: {err}",
                path.display()
            ))
        })?;
        let build_info: BuildInfo = serde_json::from_str(&raw).map_err(|err| {
            LTEmbedError::ModelLoad(format!(
                "Failed to parse build-info metadata at {}: {err}",
                path.display()
            ))
        })?;

        let metadata = build_info.model_metadata;
        if metadata.input_kind != "retrieval" {
            return Err(LTEmbedError::ModelLoad(format!(
                "Unsupported input_kind '{}' for bundle target '{}'",
                metadata.input_kind, build_info.target_id
            )));
        }
        if metadata.pooling != "last_token" {
            return Err(LTEmbedError::ModelLoad(format!(
                "Unsupported pooling '{}' for bundle target '{}'",
                metadata.pooling, build_info.target_id
            )));
        }

        Ok(Self {
            query_prefix: metadata.query_prefix,
            document_prefix: metadata.document_prefix,
            raw_embedding_dimension: metadata.raw_embedding_dimension,
            max_length: metadata.max_length,
        })
    }
}

fn prefixed_text(input: EmbeddingInput<'_>, spec: &ModelSpec) -> String {
    match input.kind {
        EmbeddingInputKind::Query => format!("{}{}", spec.query_prefix, input.text),
        EmbeddingInputKind::Document => format!("{}{}", spec.document_prefix, input.text),
    }
}

fn postprocess_embedding(
    raw_embedding: &[f32],
    raw_embedding_dimension: usize,
    config: OnnxEngineConfig,
) -> Result<Vec<f32>, LTEmbedError> {
    if raw_embedding.len() != raw_embedding_dimension {
        return Err(LTEmbedError::Inference(format!(
            "expected raw embedding dimension {raw_embedding_dimension}, got {}",
            raw_embedding.len()
        )));
    }

    let mut output = raw_embedding[..config.output_dimension].to_vec();
    if config.l2_normalize {
        let norm = output.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for value in &mut output {
                *value /= norm;
            }
        }
    }
    Ok(output)
}

fn require_file(path: &Path, label: &str) -> Result<(), LTEmbedError> {
    if path.exists() {
        return Ok(());
    }

    Err(LTEmbedError::ModelLoad(format!(
        "{label} file not found: {}",
        path.display()
    )))
}

fn ensure_ort_initialized(dylib_path: Option<&Path>) -> Result<(), LTEmbedError> {
    static INIT: OnceLock<Result<OrtInitSource, String>> = OnceLock::new();

    let requested = resolve_ort_source(dylib_path);
    let initialized = INIT.get_or_init(|| match &requested {
        OrtInitSource::DynamicLibrary(path) => ort::init_from(path)
            .map_err(|err| format!("Failed to load ORT dynamic library: {err}"))
            .map(|builder| {
                let _ = builder.commit();
                requested.clone()
            }),
        OrtInitSource::System => Ok(OrtInitSource::System),
    });

    match initialized {
        Ok(source) if *source == requested => Ok(()),
        Ok(OrtInitSource::DynamicLibrary(_)) if requested == OrtInitSource::System => Ok(()),
        Ok(source) => Err(LTEmbedError::ModelLoad(format!(
            "ORT is already initialized for {:?}, cannot reinitialize with {:?}",
            source, requested
        ))),
        Err(message) => Err(LTEmbedError::ModelLoad(message.clone())),
    }
}

fn resolve_ort_source(dylib_path: Option<&Path>) -> OrtInitSource {
    if let Some(path) = dylib_path {
        return OrtInitSource::DynamicLibrary(path.to_path_buf());
    }

    match std::env::var("ORT_DYLIB_PATH") {
        Ok(path) if !path.is_empty() => OrtInitSource::DynamicLibrary(PathBuf::from(path)),
        _ => OrtInitSource::System,
    }
}

fn model_sequence_length(shape: &[i64]) -> Result<Option<usize>, LTEmbedError> {
    if shape.len() != 2 {
        return Err(LTEmbedError::ModelLoad(format!(
            "ORT model input must be rank-2, got shape {shape:?}"
        )));
    }

    match shape[1] {
        dim if dim < 0 => Ok(None),
        dim => usize::try_from(dim)
            .map(Some)
            .map_err(|_| LTEmbedError::ModelLoad(format!("Invalid ORT input shape {shape:?}"))),
    }
}

fn resolved_model_sequence_length(
    input_ids_shape: &[i64],
    attention_mask_shape: &[i64],
) -> Result<Option<usize>, LTEmbedError> {
    let input_sequence_length = model_sequence_length(input_ids_shape)?;
    let attention_mask_sequence_length = model_sequence_length(attention_mask_shape)?;

    match (input_sequence_length, attention_mask_sequence_length) {
        (Some(input_len), Some(mask_len)) if input_len != mask_len => Err(LTEmbedError::ModelLoad(
            format!(
                "ORT model inputs `input_ids` and `attention_mask` must expose compatible sequence lengths, got {input_ids_shape:?} and {attention_mask_shape:?}"
            ),
        )),
        (Some(input_len), Some(_)) => Ok(Some(input_len)),
        (Some(input_len), None) => Ok(Some(input_len)),
        (None, Some(mask_len)) => Ok(Some(mask_len)),
        (None, None) => Ok(None),
    }
}

fn effective_sequence_length(
    model_sequence_length: Option<usize>,
    batch_max_sequence_length: usize,
) -> Result<usize, LTEmbedError> {
    match model_sequence_length {
        Some(model_len) if batch_max_sequence_length > model_len => Err(LTEmbedError::Inference(
            format!(
                "encoded input length {batch_max_sequence_length} exceeds ORT model sequence length {model_len}"
            ),
        )),
        Some(model_len) => Ok(model_len),
        None => Ok(batch_max_sequence_length),
    }
}

impl SessionIo {
    fn from_session(
        session: &Session,
        raw_embedding_dimension: usize,
    ) -> Result<Self, LTEmbedError> {
        let input_ids = session
            .inputs()
            .iter()
            .find(|input| input.name() == "input_ids")
            .ok_or_else(|| {
                LTEmbedError::ModelLoad("ORT model is missing required input `input_ids`".into())
            })?;
        let attention_mask = session
            .inputs()
            .iter()
            .find(|input| input.name() == "attention_mask")
            .ok_or_else(|| {
                LTEmbedError::ModelLoad(
                    "ORT model is missing required input `attention_mask`".into(),
                )
            })?;
        let last_hidden_state = session
            .outputs()
            .iter()
            .find(|output| output.name() == "last_hidden_state")
            .ok_or_else(|| {
                LTEmbedError::ModelLoad(
                    "ORT model is missing required output `last_hidden_state`".into(),
                )
            })?;
        let raw_dim = last_hidden_state
            .dtype()
            .tensor_shape()
            .and_then(|shape| shape.last().copied());
        if raw_dim != Some(raw_embedding_dimension as i64) {
            return Err(LTEmbedError::ModelLoad(format!(
                "ORT model output `last_hidden_state` must expose raw embedding dimension {raw_embedding_dimension}, got {raw_dim:?}"
            )));
        }
        let input_ids_shape = input_ids.dtype().tensor_shape().ok_or_else(|| {
            LTEmbedError::ModelLoad("ORT model input `input_ids` must be a tensor".into())
        })?;
        let attention_mask_shape = attention_mask.dtype().tensor_shape().ok_or_else(|| {
            LTEmbedError::ModelLoad("ORT model input `attention_mask` must be a tensor".into())
        })?;
        let sequence_length =
            resolved_model_sequence_length(input_ids_shape, attention_mask_shape)?;

        Ok(Self {
            input_ids: input_ids.name().to_string(),
            attention_mask: attention_mask.name().to_string(),
            last_hidden_state: last_hidden_state.name().to_string(),
            sequence_length,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_embedding_input_query_constructor() {
        assert_eq!(
            EmbeddingInput::query("hello"),
            EmbeddingInput {
                text: "hello",
                kind: EmbeddingInputKind::Query,
            }
        );
    }

    #[test]
    fn test_embedding_input_document_constructor() {
        assert_eq!(
            EmbeddingInput::document("hello"),
            EmbeddingInput {
                text: "hello",
                kind: EmbeddingInputKind::Document,
            }
        );
    }

    #[test]
    fn test_prefixed_text_applies_query_prefix() {
        let spec = ModelSpec::jina_defaults();
        assert_eq!(
            prefixed_text(EmbeddingInput::query("hello"), &spec),
            "Query: hello"
        );
    }

    #[test]
    fn test_prefixed_text_applies_document_prefix() {
        let spec = ModelSpec::jina_defaults();
        assert_eq!(
            prefixed_text(EmbeddingInput::document("hello"), &spec),
            "Document: hello"
        );
    }

    #[test]
    fn test_postprocess_embedding_truncates_and_normalizes() {
        let mut raw = vec![0.0; RAW_EMBEDDING_DIMENSION];
        raw[0] = 3.0;
        raw[1] = 4.0;
        raw[600] = 10.0;

        let embedding =
            postprocess_embedding(&raw, RAW_EMBEDDING_DIMENSION, OnnxEngineConfig::default())
                .unwrap();

        assert_eq!(embedding.len(), EMBEDDING_DIMENSION);
        let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert_relative_eq!(norm, 1.0, epsilon = 1e-6);
        assert_eq!(embedding[0], 3.0 / 5.0);
        assert_eq!(embedding[1], 4.0 / 5.0);
    }

    #[test]
    fn test_postprocess_embedding_respects_normalize_flag() {
        let raw = vec![1.0; RAW_EMBEDDING_DIMENSION];
        let embedding = postprocess_embedding(
            &raw,
            RAW_EMBEDDING_DIMENSION,
            OnnxEngineConfig {
                output_dimension: 4,
                l2_normalize: false,
            },
        )
        .unwrap();
        assert_eq!(embedding, vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_postprocess_embedding_rejects_non_matching_raw_dimension() {
        let err = postprocess_embedding(
            &vec![0.0; EMBEDDING_DIMENSION],
            RAW_EMBEDDING_DIMENSION,
            OnnxEngineConfig::default(),
        )
        .unwrap_err();
        assert!(matches!(err, LTEmbedError::Inference(_)));
    }

    #[test]
    fn test_config_rejects_output_dimension_larger_than_raw() {
        let err = OnnxEngineConfig {
            output_dimension: RAW_EMBEDDING_DIMENSION + 1,
            l2_normalize: true,
        }
        .validate(RAW_EMBEDDING_DIMENSION)
        .unwrap_err();
        assert!(matches!(err, LTEmbedError::ModelLoad(_)));
    }

    #[test]
    fn test_model_sequence_length_is_dynamic_when_second_dim_is_negative() {
        assert_eq!(model_sequence_length(&[-1, -1]).unwrap(), None);
    }

    #[test]
    fn test_model_sequence_length_uses_fixed_second_dim() {
        assert_eq!(model_sequence_length(&[-1, 8192]).unwrap(), Some(8192));
    }

    #[test]
    fn test_effective_sequence_length_uses_batch_max_for_dynamic_models() {
        assert_eq!(effective_sequence_length(None, 7).unwrap(), 7);
    }

    #[test]
    fn test_effective_sequence_length_uses_fixed_model_length() {
        assert_eq!(effective_sequence_length(Some(8192), 7).unwrap(), 8192);
    }

    #[test]
    fn test_resolved_model_sequence_length_uses_fixed_input_ids_shape() {
        assert_eq!(
            resolved_model_sequence_length(&[-1, 8192], &[-1, -1]).unwrap(),
            Some(8192)
        );
    }

    #[test]
    fn test_resolved_model_sequence_length_uses_fixed_attention_mask_shape() {
        assert_eq!(
            resolved_model_sequence_length(&[-1, -1], &[-1, 8192]).unwrap(),
            Some(8192)
        );
    }

    #[test]
    fn test_resolved_model_sequence_length_rejects_mismatched_fixed_shapes() {
        let err = resolved_model_sequence_length(&[-1, 8192], &[-1, 4096]).unwrap_err();
        assert!(matches!(err, LTEmbedError::ModelLoad(_)));
    }
}
