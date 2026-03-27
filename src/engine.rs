// src/engine.rs

use std::path::Path;
use std::sync::{Mutex, OnceLock};

use ort::session::Session;
use ort::value::TensorRef;

use crate::error::LTEmbedError;
use crate::traits::tokenizer::HFTokenizer;

pub const RAW_EMBEDDING_DIMENSION: usize = 768;
pub const EMBEDDING_DIMENSION: usize = 512;
pub const MAX_LENGTH: usize = 8192;
pub const QUERY_PREFIX: &str = "Query: ";
pub const DOCUMENT_PREFIX: &str = "Document: ";

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

#[derive(Debug)]
struct SessionIo {
    input_ids: String,
    attention_mask: String,
    last_hidden_state: String,
}

pub struct OnnxEngine {
    session: Mutex<Session>,
    tokenizer: HFTokenizer,
    io: SessionIo,
}

impl std::fmt::Debug for OnnxEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnnxEngine").finish_non_exhaustive()
    }
}

impl OnnxEngine {
    pub fn new(model_path: &str, tokenizer_path: &str) -> Result<Self, LTEmbedError> {
        if !Path::new(model_path).exists() {
            return Err(LTEmbedError::ModelLoad(format!(
                "ONNX model file not found: {model_path}"
            )));
        }
        if !Path::new(tokenizer_path).exists() {
            return Err(LTEmbedError::ModelLoad(format!(
                "tokenizer file not found: {tokenizer_path}"
            )));
        }

        ensure_ort_initialized()?;
        let tokenizer = HFTokenizer::from_file(tokenizer_path)?;
        let session = Session::builder()
            .map_err(|err| {
                LTEmbedError::ModelLoad(format!("Failed to create ORT session builder: {err}"))
            })?
            .with_intra_threads(1)
            .map_err(|err| {
                LTEmbedError::ModelLoad(format!("Failed to configure ORT session: {err}"))
            })?
            .commit_from_file(model_path)
            .map_err(|err| LTEmbedError::ModelLoad(format!("Failed to load ONNX model: {err}")))?;
        let io = SessionIo::from_session(&session)?;

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            io,
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
            .map(prefixed_text)
            .collect::<Vec<_>>();
        let encoded = self.tokenizer.encode_batch(&prefixed_inputs, MAX_LENGTH)?;
        let batch_size = encoded.len();
        let seq_len = encoded
            .iter()
            .map(|item| item.input_ids.len())
            .max()
            .unwrap_or(0);

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
                LTEmbedError::Inference(format!("Failed to extract ONNX output tensor: {err}"))
            })?;
        if hidden_shape.len() != 3 {
            return Err(LTEmbedError::Inference(format!(
                "expected rank-3 hidden states, got shape {hidden_shape:?}"
            )));
        }
        if hidden_shape[0] as usize != batch_size || hidden_shape[1] as usize != seq_len {
            return Err(LTEmbedError::Inference(format!(
                "unexpected hidden state shape {hidden_shape:?}, expected [{batch_size}, {seq_len}, {RAW_EMBEDDING_DIMENSION}]"
            )));
        }
        if hidden_shape[2] as usize != RAW_EMBEDDING_DIMENSION {
            return Err(LTEmbedError::Inference(format!(
                "expected raw embedding dimension {RAW_EMBEDDING_DIMENSION}, got {}",
                hidden_shape[2]
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
            let hidden_offset = (batch_idx * seq_len + last_token_idx) * RAW_EMBEDDING_DIMENSION;
            let raw = &hidden_data[hidden_offset..hidden_offset + RAW_EMBEDDING_DIMENSION];
            embeddings.push(truncate_and_normalize(raw)?);
        }

        Ok(embeddings)
    }
}

fn prefixed_text(input: EmbeddingInput<'_>) -> String {
    match input.kind {
        EmbeddingInputKind::Query => format!("{QUERY_PREFIX}{}", input.text),
        EmbeddingInputKind::Document => format!("{DOCUMENT_PREFIX}{}", input.text),
    }
}

fn truncate_and_normalize(raw_embedding: &[f32]) -> Result<Vec<f32>, LTEmbedError> {
    if raw_embedding.len() != RAW_EMBEDDING_DIMENSION {
        return Err(LTEmbedError::Inference(format!(
            "expected raw embedding dimension {RAW_EMBEDDING_DIMENSION}, got {}",
            raw_embedding.len()
        )));
    }

    let mut truncated = raw_embedding[..EMBEDDING_DIMENSION].to_vec();
    let norm = truncated.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut truncated {
            *value /= norm;
        }
    }
    Ok(truncated)
}

fn ensure_ort_initialized() -> Result<(), LTEmbedError> {
    static INIT: OnceLock<Result<(), String>> = OnceLock::new();

    INIT.get_or_init(|| match std::env::var("ORT_DYLIB_PATH") {
        Ok(path) if !path.is_empty() => ort::init_from(path)
            .map_err(|err| format!("Failed to load ORT dynamic library: {err}"))
            .map(|builder| {
                let _ = builder.commit();
            }),
        _ => Ok(()),
    })
    .clone()
    .map_err(LTEmbedError::ModelLoad)
}

impl SessionIo {
    fn from_session(session: &Session) -> Result<Self, LTEmbedError> {
        let input_ids = session
            .inputs()
            .iter()
            .find(|input| input.name() == "input_ids")
            .ok_or_else(|| {
                LTEmbedError::ModelLoad("ONNX model is missing required input `input_ids`".into())
            })?;
        let attention_mask = session
            .inputs()
            .iter()
            .find(|input| input.name() == "attention_mask")
            .ok_or_else(|| {
                LTEmbedError::ModelLoad(
                    "ONNX model is missing required input `attention_mask`".into(),
                )
            })?;
        let last_hidden_state = session
            .outputs()
            .iter()
            .find(|output| output.name() == "last_hidden_state")
            .ok_or_else(|| {
                LTEmbedError::ModelLoad(
                    "ONNX model is missing required output `last_hidden_state`".into(),
                )
            })?;
        let raw_dim = last_hidden_state
            .dtype()
            .tensor_shape()
            .and_then(|shape| shape.last().copied());
        if raw_dim != Some(RAW_EMBEDDING_DIMENSION as i64) {
            return Err(LTEmbedError::ModelLoad(format!(
                "ONNX model output `last_hidden_state` must expose raw embedding dimension {RAW_EMBEDDING_DIMENSION}, got {raw_dim:?}"
            )));
        }

        Ok(Self {
            input_ids: input_ids.name().to_string(),
            attention_mask: attention_mask.name().to_string(),
            last_hidden_state: last_hidden_state.name().to_string(),
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
        assert_eq!(
            prefixed_text(EmbeddingInput::query("hello")),
            "Query: hello"
        );
    }

    #[test]
    fn test_prefixed_text_applies_document_prefix() {
        assert_eq!(
            prefixed_text(EmbeddingInput::document("hello")),
            "Document: hello"
        );
    }

    #[test]
    fn test_truncate_and_normalize_returns_512_dim_unit_vector() {
        let mut raw = vec![0.0; RAW_EMBEDDING_DIMENSION];
        raw[0] = 3.0;
        raw[1] = 4.0;
        raw[600] = 10.0;

        let embedding = truncate_and_normalize(&raw).unwrap();

        assert_eq!(embedding.len(), EMBEDDING_DIMENSION);
        let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert_relative_eq!(norm, 1.0, epsilon = 1e-6);
        assert_eq!(embedding[0], 3.0 / 5.0);
        assert_eq!(embedding[1], 4.0 / 5.0);
    }

    #[test]
    fn test_truncate_and_normalize_rejects_non_768_raw_embedding() {
        let err = truncate_and_normalize(&vec![0.0; EMBEDDING_DIMENSION]).unwrap_err();
        assert!(matches!(err, LTEmbedError::Inference(_)));
    }
}
