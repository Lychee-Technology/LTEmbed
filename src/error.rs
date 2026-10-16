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

    /// Backend model load (GGUF load, context creation) failure, or a GGUF the engine cannot
    /// run as this model: its embedding width differs from `build-info.json`, it declares
    /// another pooling, or its tokenization does not add BOS and EOS.
    #[error("{0}")]
    Runtime(String),
}

/// Failure classes surfaced while running inference on an already-loaded engine.
///
/// An input that is too long is not one of them: it is `LTEmbedError::InputTooLong`, returned
/// before the backend runs.
#[derive(Debug, Error)]
pub enum InferenceError {
    /// Model output rank / dimension mismatch.
    #[error("{0}")]
    OutputShape(String),

    /// Tensor conversion or extraction failure.
    #[error("{0}")]
    Tensor(String),

    /// Backend inference (e.g. `llama_encode`) failure.
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
            label: "GGUF model".to_string(),
            path: PathBuf::from("/tmp/model.gguf"),
        });
        assert_eq!(
            e.to_string(),
            "Model load failed: GGUF model file not found: /tmp/model.gguf"
        );
    }

    #[test]
    fn test_inference_error_display() {
        let e = LTEmbedError::Inference(InferenceError::Backend(
            "llama_encode failed (rc=-3)".to_string(),
        ));
        assert_eq!(
            e.to_string(),
            "Inference failed: llama_encode failed (rc=-3)"
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
