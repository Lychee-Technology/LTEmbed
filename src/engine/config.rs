use crate::error::{LTEmbedError, ModelLoadError};

use super::EMBEDDING_DIMENSION;

/// Post-processing applied to each raw pooled embedding.
///
/// The engine keeps the first `output_dimension` values of the model's raw embedding
/// (Matryoshka truncation), then L2-normalizes them if `l2_normalize` is set.
/// EmbeddingGemma 2's raw embedding has 768 values, and the model was trained to keep its
/// quality when truncated to 768, 512, 256 or 128 of them. Any width from 1 to the raw
/// dimension is accepted, but only those four are trained, and only they are covered by the
/// reference tests. The default is 512, normalized.
///
/// Truncating a unit vector does not keep it unit length, so leave `l2_normalize` on when the
/// embeddings are compared by cosine or dot product. Query and document embeddings must use
/// the same `output_dimension` and normalization to be comparable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EngineConfig {
    /// Number of leading raw-embedding values to keep: `1..=768` for EmbeddingGemma 2.
    pub output_dimension: usize,
    /// L2-normalize the truncated vector. An all-zero vector stays zero.
    pub l2_normalize: bool,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            output_dimension: EMBEDDING_DIMENSION,
            l2_normalize: true,
        }
    }
}

impl EngineConfig {
    pub(crate) fn validate(self, raw_embedding_dimension: usize) -> Result<(), LTEmbedError> {
        if self.output_dimension == 0 {
            return Err(LTEmbedError::ModelLoad(ModelLoadError::Config(
                "output_dimension must be greater than zero".into(),
            )));
        }
        if self.output_dimension > raw_embedding_dimension {
            return Err(LTEmbedError::ModelLoad(ModelLoadError::Config(format!(
                "output_dimension {} exceeds raw embedding dimension {}",
                self.output_dimension, raw_embedding_dimension
            ))));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::RAW_EMBEDDING_DIMENSION;

    #[test]
    fn test_config_accepts_every_width_up_to_raw() {
        for output_dimension in [1, 128, 256, 300, 512, 768] {
            EngineConfig {
                output_dimension,
                l2_normalize: true,
            }
            .validate(RAW_EMBEDDING_DIMENSION)
            .unwrap();
        }
    }

    #[test]
    fn test_config_rejects_zero_output_dimension() {
        let err = EngineConfig {
            output_dimension: 0,
            l2_normalize: true,
        }
        .validate(RAW_EMBEDDING_DIMENSION)
        .unwrap_err();
        assert!(matches!(
            err,
            LTEmbedError::ModelLoad(ModelLoadError::Config(_))
        ));
    }

    #[test]
    fn test_config_rejects_output_dimension_larger_than_raw() {
        let err = EngineConfig {
            output_dimension: RAW_EMBEDDING_DIMENSION + 1,
            l2_normalize: true,
        }
        .validate(RAW_EMBEDDING_DIMENSION)
        .unwrap_err();
        assert!(matches!(
            err,
            LTEmbedError::ModelLoad(ModelLoadError::Config(_))
        ));
    }
}
