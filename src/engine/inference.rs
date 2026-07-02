use crate::error::LTEmbedError;

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

pub(crate) fn prefixed_text(input: EmbeddingInput<'_>, spec: &ModelSpec) -> String {
    match input.kind {
        EmbeddingInputKind::Query => format!("{}{}", spec.query_prefix, input.text),
        EmbeddingInputKind::Document => format!("{}{}", spec.document_prefix, input.text),
    }
}

pub(crate) fn postprocess_embedding(
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
        assert!(matches!(err, LTEmbedError::Inference(_)));
    }
}
