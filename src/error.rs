// src/error.rs
use thiserror::Error;

#[derive(Debug, Error)]
pub enum LTEmbedError {
    #[error("Model load failed: {0}")]
    ModelLoad(String),

    #[error("Tokenization failed: {0}")]
    Tokenization(String),

    #[error("Inference failed: {0}")]
    Inference(String),

    #[error("Input too long: {tokens} tokens exceeds the {max} token limit")]
    InputTooLong { tokens: usize, max: usize },

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_load_error_display() {
        let e = LTEmbedError::ModelLoad("file not found".to_string());
        assert_eq!(e.to_string(), "Model load failed: file not found");
    }

    #[test]
    fn test_io_from_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "no file");
        let e: LTEmbedError = io_err.into();
        assert!(e.to_string().contains("I/O error"));
    }

    #[test]
    fn test_input_too_long_display() {
        let e = LTEmbedError::InputTooLong { tokens: 600, max: 512 };
        assert_eq!(e.to_string(), "Input too long: 600 tokens exceeds the 512 token limit");
    }
}
