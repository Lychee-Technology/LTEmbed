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
///
/// A raw vector with a NaN or infinite value is an inference failure, not an embedding:
/// normalizing it would spread the NaN, or turn an infinity into NaN, without an error. The
/// norm is accumulated in `f64`, so squaring large finite values cannot overflow it. An
/// all-zero vector stays zero.
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
    if let Some(index) = raw_embedding.iter().position(|value| !value.is_finite()) {
        return Err(LTEmbedError::Inference(InferenceError::Backend(format!(
            "raw embedding value {index} is {}, not a finite number",
            raw_embedding[index]
        ))));
    }

    let mut output = raw_embedding[..config.output_dimension].to_vec();
    if config.l2_normalize {
        let norm = output
            .iter()
            .map(|&x| f64::from(x) * f64::from(x))
            .sum::<f64>()
            .sqrt();
        if norm > 0.0 {
            for value in &mut output {
                *value = (f64::from(*value) / norm) as f32;
            }
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{
        DOCUMENT_PREFIX, EMBEDDING_DIMENSION, QUERY_PREFIX, RAW_EMBEDDING_DIMENSION,
    };
    use approx::assert_relative_eq;

    #[test]
    fn test_prefixed_text_applies_query_prefix() {
        let spec = ModelSpec::defaults();
        assert_eq!(
            prefixed_text(EmbeddingInput::query("如何重置密码？"), &spec),
            "task: search result | query: 如何重置密码？"
        );
    }

    #[test]
    fn test_prefixed_text_applies_document_prefix() {
        let spec = ModelSpec::defaults();
        assert_eq!(
            prefixed_text(EmbeddingInput::document("在登录页面点击忘记密码"), &spec),
            "title: none | text: 在登录页面点击忘记密码"
        );
    }

    /// The prefix is added once per call and never stripped or deduplicated: text that already
    /// starts with it is the caller's text, so it is embedded with the prefix twice.
    #[test]
    fn test_prefixed_text_adds_prefix_exactly_once() {
        let spec = ModelSpec::defaults();
        let query = prefixed_text(EmbeddingInput::query("hello"), &spec);
        assert_eq!(query.matches(QUERY_PREFIX).count(), 1);
        assert!(!query.contains(DOCUMENT_PREFIX));
        let document = prefixed_text(EmbeddingInput::document("hello"), &spec);
        assert_eq!(document.matches(DOCUMENT_PREFIX).count(), 1);
        assert!(!document.contains(QUERY_PREFIX));
        assert_eq!(
            prefixed_text(EmbeddingInput::query(&query), &spec),
            format!("{QUERY_PREFIX}{QUERY_PREFIX}hello")
        );
        assert_eq!(
            prefixed_text(EmbeddingInput::query(""), &spec),
            QUERY_PREFIX
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

    /// Every Matryoshka width EmbeddingGemma 2 was trained for keeps the leading values and
    /// renormalizes them.
    #[test]
    fn test_postprocess_embedding_truncates_to_each_matryoshka_width() {
        let raw: Vec<f32> = (0..RAW_EMBEDDING_DIMENSION)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.5 + 0.1)
            .collect();
        for dim in [768, 512, 256, 128] {
            let embedding = postprocess_embedding(
                &raw,
                RAW_EMBEDDING_DIMENSION,
                EngineConfig {
                    output_dimension: dim,
                    l2_normalize: true,
                },
            )
            .unwrap();
            assert_eq!(embedding.len(), dim);
            let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert_relative_eq!(norm, 1.0, epsilon = 1e-6);
            let raw_norm = raw[..dim].iter().map(|x| x * x).sum::<f32>().sqrt();
            for (value, raw_value) in embedding.iter().zip(&raw[..dim]) {
                assert_relative_eq!(*value, raw_value / raw_norm, epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_postprocess_embedding_rejects_non_finite_values() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            // Past the 512 kept dimensions: a non-finite raw value fails the whole vector.
            for index in [0, 600] {
                let mut raw = vec![0.5_f32; RAW_EMBEDDING_DIMENSION];
                raw[index] = bad;
                for l2_normalize in [true, false] {
                    let err = postprocess_embedding(
                        &raw,
                        RAW_EMBEDDING_DIMENSION,
                        EngineConfig {
                            output_dimension: EMBEDDING_DIMENSION,
                            l2_normalize,
                        },
                    )
                    .unwrap_err();
                    match err {
                        LTEmbedError::Inference(InferenceError::Backend(msg)) => {
                            assert!(msg.contains(&format!("value {index} is")), "{msg}");
                        }
                        other => panic!("expected Inference(Backend), got {other:?}"),
                    }
                }
            }
        }
    }

    /// Values whose squares overflow `f32` still normalize to a finite unit vector.
    #[test]
    fn test_postprocess_embedding_normalizes_large_values_without_overflow() {
        let raw = vec![1e30_f32; RAW_EMBEDDING_DIMENSION];
        let embedding =
            postprocess_embedding(&raw, RAW_EMBEDDING_DIMENSION, EngineConfig::default()).unwrap();
        assert!(embedding.iter().all(|x| x.is_finite()));
        let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert_relative_eq!(norm, 1.0, epsilon = 1e-5);
    }
}
