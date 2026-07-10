use crate::error::{LTEmbedError, ModelLoadError};

use super::EMBEDDING_DIMENSION;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EngineConfig {
    pub output_dimension: usize,
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
