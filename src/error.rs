// src/error.rs
use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum LTEmbedError {
    #[error("Model load failed: {0}")]
    ModelLoad(#[from] ModelLoadError),

    #[error("Tokenization failed: {0}")]
    Tokenization(String),

    #[error("Inference failed: {0}")]
    Inference(#[from] InferenceError),

    #[error("Input too long: {tokens} tokens exceeds the {max} token limit")]
    InputTooLong { tokens: usize, max: usize },

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Failure classes surfaced while loading a bundle and building an engine.
///
/// The high-value, caller-matchable cases (`MissingFile`, `UnsupportedModelFormat`,
/// `UnsupportedInputKind`, `UnsupportedPooling`) are modeled explicitly; the
/// remaining metadata / config / runtime failures are grouped into broader buckets.
#[derive(Debug, Error)]
pub enum ModelLoadError {
    #[error("{label} file not found: {path}")]
    MissingFile { label: String, path: PathBuf },

    #[error(
        "Unsupported model_format '{model_format}' for bundle target '{target}' (expected 'gguf')"
    )]
    UnsupportedModelFormat {
        model_format: String,
        target: String,
    },

    #[error("Unsupported input_kind '{input_kind}' for bundle target '{target}'")]
    UnsupportedInputKind { input_kind: String, target: String },

    #[error("Unsupported pooling '{pooling}' for bundle target '{target}'")]
    UnsupportedPooling { pooling: String, target: String },

    /// build-info read or parse failure.
    #[error("{0}")]
    Metadata(String),

    /// `EngineConfig` validation failure.
    #[error("{0}")]
    Config(String),

    /// Backend model load (GGUF load, context creation) or tokenizer load failure.
    #[error("{0}")]
    Runtime(String),
}

/// Failure classes surfaced while running inference on an already-loaded engine.
///
/// `SequenceTooLong`, `AllPadding`, and `OutputShape` are the cases most useful
/// to match precisely; the remainder are broader buckets.
#[derive(Debug, Error)]
pub enum InferenceError {
    #[error("encoded input length {encoded} exceeds model context length {model}")]
    SequenceTooLong { encoded: usize, model: usize },

    #[error("attention mask contains only padding")]
    AllPadding,

    /// Model output rank / dimension mismatch.
    #[error("{0}")]
    OutputShape(String),

    /// Tensor conversion or extraction failure.
    #[error("{0}")]
    Tensor(String),

    /// Backend inference (e.g. `llama_decode`) failure.
    #[error("{0}")]
    Backend(String),

    #[error("backend context mutex poisoned")]
    MutexPoisoned,

    /// Internal invariant violation.
    #[error("{0}")]
    Internal(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_load_error_display() {
        let e = LTEmbedError::ModelLoad(ModelLoadError::Runtime("file not found".to_string()));
        assert_eq!(e.to_string(), "Model load failed: file not found");
    }

    #[test]
    fn test_missing_file_error_display() {
        let e = LTEmbedError::ModelLoad(ModelLoadError::MissingFile {
            label: "tokenizer".to_string(),
            path: PathBuf::from("/tmp/tokenizer.json"),
        });
        assert_eq!(
            e.to_string(),
            "Model load failed: tokenizer file not found: /tmp/tokenizer.json"
        );
    }

    #[test]
    fn test_inference_error_display() {
        let e = LTEmbedError::Inference(InferenceError::SequenceTooLong {
            encoded: 9000,
            model: 8192,
        });
        assert_eq!(
            e.to_string(),
            "Inference failed: encoded input length 9000 exceeds model context length 8192"
        );
    }

    #[test]
    fn test_io_from_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "no file");
        let e: LTEmbedError = io_err.into();
        assert!(e.to_string().contains("I/O error"));
    }

    #[test]
    fn test_input_too_long_display() {
        let e = LTEmbedError::InputTooLong {
            tokens: 600,
            max: 512,
        };
        assert_eq!(
            e.to_string(),
            "Input too long: 600 tokens exceeds the 512 token limit"
        );
    }
}
