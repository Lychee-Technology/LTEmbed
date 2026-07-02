use crate::error::{InferenceError, LTEmbedError};
use crate::traits::tokenizer::TokenizerOutput;

use super::bundle::ModelSpec;
use super::config::OnnxEngineConfig;
use super::input::{EmbeddingInput, EmbeddingInputKind};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EmbedBatchProfile {
    pub batch_size: usize,
    pub sequence_length: usize,
    pub prefix_ms: f64,
    pub tokenize_ms: f64,
    pub tensorize_ms: f64,
    pub run_ms: f64,
    pub extract_ms: f64,
    pub postprocess_ms: f64,
    pub total_ms: f64,
}

impl EmbedBatchProfile {
    /// Zeroed profile returned for empty batches.
    pub(crate) fn empty() -> Self {
        Self {
            batch_size: 0,
            sequence_length: 0,
            prefix_ms: 0.0,
            tokenize_ms: 0.0,
            tensorize_ms: 0.0,
            run_ms: 0.0,
            extract_ms: 0.0,
            postprocess_ms: 0.0,
            total_ms: 0.0,
        }
    }
}

pub(crate) fn prefixed_text(input: EmbeddingInput<'_>, spec: &ModelSpec) -> String {
    match input.kind {
        EmbeddingInputKind::Query => format!("{}{}", spec.query_prefix, input.text),
        EmbeddingInputKind::Document => format!("{}{}", spec.document_prefix, input.text),
    }
}

/// Pack a tokenized batch into row-major `[batch_size, seq_len]` `input_ids` and
/// `attention_mask` tensors, padding shorter sequences with zeros.
pub(crate) fn pack_batch(encoded: &[TokenizerOutput], seq_len: usize) -> (Vec<i64>, Vec<i64>) {
    let batch_size = encoded.len();
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

    (input_ids, attention_mask)
}

/// Validate that the model's `last_hidden_state` output has the expected
/// rank-3 `[batch_size, seq_len, raw_embedding_dimension]` shape.
pub(crate) fn validate_hidden_shape(
    hidden_shape: &[i64],
    batch_size: usize,
    seq_len: usize,
    raw_embedding_dimension: usize,
) -> Result<(), LTEmbedError> {
    if hidden_shape.len() != 3 {
        return Err(LTEmbedError::Inference(InferenceError::OutputShape(
            format!("expected rank-3 hidden states, got shape {hidden_shape:?}"),
        )));
    }
    if hidden_shape[0] as usize != batch_size || hidden_shape[1] as usize != seq_len {
        return Err(LTEmbedError::Inference(InferenceError::OutputShape(
            format!(
                "unexpected hidden state shape {hidden_shape:?}, expected [{batch_size}, {seq_len}, {raw_embedding_dimension}]"
            ),
        )));
    }
    if hidden_shape[2] as usize != raw_embedding_dimension {
        return Err(LTEmbedError::Inference(InferenceError::OutputShape(
            format!(
                "expected raw embedding dimension {}, got {}",
                raw_embedding_dimension, hidden_shape[2]
            ),
        )));
    }
    Ok(())
}

/// Extract one embedding per batch row from the last non-padding token, then
/// truncate and optionally L2-normalize each via [`postprocess_embedding`].
pub(crate) fn pool_last_token(
    hidden_data: &[f32],
    attention_mask: &[i64],
    batch_size: usize,
    seq_len: usize,
    raw_embedding_dimension: usize,
    config: OnnxEngineConfig,
) -> Result<Vec<Vec<f32>>, LTEmbedError> {
    let mut embeddings = Vec::with_capacity(batch_size);
    for batch_idx in 0..batch_size {
        let mask_start = batch_idx * seq_len;
        let mask_end = mask_start + seq_len;
        let mask_slice = &attention_mask[mask_start..mask_end];
        let last_token_idx = mask_slice
            .iter()
            .rposition(|mask| *mask == 1)
            .ok_or(LTEmbedError::Inference(InferenceError::AllPadding))?;
        let hidden_offset = (batch_idx * seq_len + last_token_idx) * raw_embedding_dimension;
        let raw = &hidden_data[hidden_offset..hidden_offset + raw_embedding_dimension];
        embeddings.push(postprocess_embedding(raw, raw_embedding_dimension, config)?);
    }
    Ok(embeddings)
}

pub(crate) fn postprocess_embedding(
    raw_embedding: &[f32],
    raw_embedding_dimension: usize,
    config: OnnxEngineConfig,
) -> Result<Vec<f32>, LTEmbedError> {
    if raw_embedding.len() != raw_embedding_dimension {
        return Err(LTEmbedError::Inference(InferenceError::OutputShape(
            format!(
                "expected raw embedding dimension {raw_embedding_dimension}, got {}",
                raw_embedding.len()
            ),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{EMBEDDING_DIMENSION, RAW_EMBEDDING_DIMENSION};
    use approx::assert_relative_eq;

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
        assert!(matches!(
            err,
            LTEmbedError::Inference(InferenceError::OutputShape(_))
        ));
    }

    fn tokenizer_output(input_ids: Vec<u32>, attention_mask: Vec<u32>) -> TokenizerOutput {
        TokenizerOutput {
            input_ids,
            attention_mask,
            token_type_ids: vec![0; 0],
        }
    }

    #[test]
    fn test_pack_batch_pads_shorter_sequences_with_zeros() {
        let encoded = vec![
            tokenizer_output(vec![5, 6, 7], vec![1, 1, 1]),
            tokenizer_output(vec![9], vec![1]),
        ];
        let (input_ids, attention_mask) = pack_batch(&encoded, 3);
        assert_eq!(input_ids, vec![5, 6, 7, 9, 0, 0]);
        assert_eq!(attention_mask, vec![1, 1, 1, 1, 0, 0]);
    }

    #[test]
    fn test_validate_hidden_shape_accepts_expected_shape() {
        assert!(validate_hidden_shape(&[2, 3, 768], 2, 3, 768).is_ok());
    }

    #[test]
    fn test_validate_hidden_shape_rejects_wrong_rank() {
        let err = validate_hidden_shape(&[2, 3], 2, 3, 768).unwrap_err();
        assert!(matches!(
            err,
            LTEmbedError::Inference(InferenceError::OutputShape(_))
        ));
    }

    #[test]
    fn test_validate_hidden_shape_rejects_wrong_dimension() {
        let err = validate_hidden_shape(&[2, 3, 512], 2, 3, 768).unwrap_err();
        assert!(matches!(
            err,
            LTEmbedError::Inference(InferenceError::OutputShape(_))
        ));
    }

    #[test]
    fn test_pool_last_token_selects_last_unmasked_token() {
        // batch_size=1, seq_len=2, raw_dim=2; second token is padding.
        let hidden_data = vec![3.0, 4.0, 9.0, 9.0];
        let attention_mask = vec![1_i64, 0];
        let pooled = pool_last_token(
            &hidden_data,
            &attention_mask,
            1,
            2,
            2,
            OnnxEngineConfig {
                output_dimension: 2,
                l2_normalize: true,
            },
        )
        .unwrap();
        assert_eq!(pooled.len(), 1);
        assert_relative_eq!(pooled[0][0], 3.0 / 5.0, epsilon = 1e-6);
        assert_relative_eq!(pooled[0][1], 4.0 / 5.0, epsilon = 1e-6);
    }

    #[test]
    fn test_pool_last_token_rejects_all_padding_rows() {
        let hidden_data = vec![1.0, 2.0];
        let attention_mask = vec![0_i64];
        let err = pool_last_token(
            &hidden_data,
            &attention_mask,
            1,
            1,
            2,
            OnnxEngineConfig::default(),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            LTEmbedError::Inference(InferenceError::AllPadding)
        ));
    }
}
