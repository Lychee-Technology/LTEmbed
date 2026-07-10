use crate::error::{InferenceError, LTEmbedError};

use super::bundle::ModelSpec;
use super::config::EngineConfig;
use super::input::{EmbeddingInput, EmbeddingInputKind};

/// Per-stage timing for one `embed_batch` call (backend-agnostic).
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

/// Truncate a raw pooled embedding to `config.output_dimension` (Matryoshka) and
/// optionally L2-normalize it. Shared across backends — a backend is responsible only
/// for producing the raw, un-normalized pooled `raw_embedding_dimension` vector.
pub(crate) fn postprocess_embedding(
    raw_embedding: &[f32],
    raw_embedding_dimension: usize,
    config: EngineConfig,
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
            postprocess_embedding(&raw, RAW_EMBEDDING_DIMENSION, EngineConfig::default()).unwrap();

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
            EngineConfig {
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
            EngineConfig::default(),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            LTEmbedError::Inference(InferenceError::OutputShape(_))
        ));
    }

    #[test]
    fn test_postprocess_embedding_preserves_zero_vector_after_normalization() {
        let raw = vec![0.0_f32; RAW_EMBEDDING_DIMENSION];
        let embedding = postprocess_embedding(
            &raw,
            RAW_EMBEDDING_DIMENSION,
            EngineConfig {
                output_dimension: 512,
                l2_normalize: true,
            },
        )
        .unwrap();
        // A zero vector stays zero: the norm>0 guard skips division.
        assert_eq!(embedding, vec![0.0_f32; 512]);
    }
}
