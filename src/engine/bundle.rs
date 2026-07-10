use std::fs;
use std::path::Path;

use serde::Deserialize;

use crate::error::{LTEmbedError, ModelLoadError};

use super::{DOCUMENT_PREFIX, MAX_LENGTH, QUERY_PREFIX, RAW_EMBEDDING_DIMENSION};

#[derive(Debug, Clone)]
pub(crate) struct ModelSpec {
    pub(crate) query_prefix: String,
    pub(crate) document_prefix: String,
    pub(crate) raw_embedding_dimension: usize,
    pub(crate) max_length: usize,
}

#[derive(Debug, Deserialize)]
struct BuildInfo {
    target_id: String,
    model_metadata: BuildMetadata,
}

#[derive(Debug, Deserialize)]
struct BuildMetadata {
    model_format: Option<String>,
    pooling: String,
    input_kind: String,
    query_prefix: String,
    document_prefix: String,
    raw_embedding_dimension: usize,
    #[allow(dead_code)]
    output_embedding_dimension: usize,
    max_length: usize,
}

impl ModelSpec {
    /// Fallback spec matching the jina-v5-nano defaults (used by tests and as a sane
    /// default when a bundle omits build-info-derived fields).
    #[allow(dead_code)]
    pub(crate) fn jina_defaults() -> Self {
        Self {
            query_prefix: QUERY_PREFIX.to_string(),
            document_prefix: DOCUMENT_PREFIX.to_string(),
            raw_embedding_dimension: RAW_EMBEDDING_DIMENSION,
            max_length: MAX_LENGTH,
        }
    }

    pub(crate) fn from_build_info(path: &Path) -> Result<Self, LTEmbedError> {
        let raw = fs::read_to_string(path).map_err(|err| {
            LTEmbedError::ModelLoad(ModelLoadError::Metadata(format!(
                "Failed to read build-info metadata at {}: {err}",
                path.display()
            )))
        })?;
        let build_info: BuildInfo = serde_json::from_str(&raw).map_err(|err| {
            LTEmbedError::ModelLoad(ModelLoadError::Metadata(format!(
                "Failed to parse build-info metadata at {}: {err}",
                path.display()
            )))
        })?;

        let metadata = build_info.model_metadata;
        // GGUF-only loader: reject stale ORT (or otherwise non-GGUF) build metadata. A missing
        // `model_format` is tolerated for backward compatibility with minimal bundles.
        if let Some(model_format) = &metadata.model_format {
            if model_format != "gguf" {
                return Err(LTEmbedError::ModelLoad(
                    ModelLoadError::UnsupportedModelFormat {
                        model_format: model_format.clone(),
                        target: build_info.target_id,
                    },
                ));
            }
        }
        if metadata.input_kind != "retrieval" && metadata.input_kind != "text" {
            return Err(LTEmbedError::ModelLoad(
                ModelLoadError::UnsupportedInputKind {
                    input_kind: metadata.input_kind,
                    target: build_info.target_id,
                },
            ));
        }
        if metadata.pooling != "last_token" && metadata.pooling != "lasttoken" {
            return Err(LTEmbedError::ModelLoad(
                ModelLoadError::UnsupportedPooling {
                    pooling: metadata.pooling,
                    target: build_info.target_id,
                },
            ));
        }

        Ok(Self {
            query_prefix: metadata.query_prefix,
            document_prefix: metadata.document_prefix,
            raw_embedding_dimension: metadata.raw_embedding_dimension,
            max_length: metadata.max_length,
        })
    }
}

pub(crate) fn require_file(path: &Path, label: &str) -> Result<(), LTEmbedError> {
    if path.exists() {
        return Ok(());
    }

    Err(LTEmbedError::ModelLoad(ModelLoadError::MissingFile {
        label: label.to_string(),
        path: path.to_path_buf(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_dir() -> PathBuf {
        static UNIQUE_TEMP_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let counter = UNIQUE_TEMP_DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("ltembed-bundle-unit-tests-{nanos}-{counter}"))
    }

    fn write_build_info(dir: &Path, body: &str) -> PathBuf {
        let path = dir.join("build-info.json");
        fs::write(&path, body).unwrap();
        path
    }

    #[test]
    fn test_legacy_metadata_aliases_are_accepted() {
        // `pooling: "lasttoken"` and `input_kind: "text"` are format-independent aliases
        // that remain valid for GGUF bundles.
        let temp_dir = unique_temp_dir();
        fs::create_dir_all(&temp_dir).unwrap();
        let build_info_path = write_build_info(
            &temp_dir,
            r#"{
  "target_id": "gguf-aliases",
  "model_metadata": {
    "model_format": "gguf",
    "pooling": "lasttoken",
    "input_kind": "text",
    "query_prefix": "Query: ",
    "document_prefix": "Document: ",
    "raw_embedding_dimension": 768,
    "output_embedding_dimension": 768,
    "max_length": 8192
  }
}"#,
        );

        let spec = ModelSpec::from_build_info(&build_info_path).unwrap();

        assert_eq!(spec.query_prefix, "Query: ");
        assert_eq!(spec.document_prefix, "Document: ");
        assert_eq!(spec.raw_embedding_dimension, RAW_EMBEDDING_DIMENSION);
        assert_eq!(spec.max_length, MAX_LENGTH);

        fs::remove_dir_all(temp_dir).unwrap();
    }

    #[test]
    fn test_non_gguf_model_format_is_rejected() {
        let temp_dir = unique_temp_dir();
        fs::create_dir_all(&temp_dir).unwrap();
        let build_info_path = write_build_info(
            &temp_dir,
            r#"{
  "target_id": "stale-ort",
  "model_metadata": {
    "model_format": "ort",
    "pooling": "last_token",
    "input_kind": "retrieval",
    "query_prefix": "Query: ",
    "document_prefix": "Document: ",
    "raw_embedding_dimension": 768,
    "output_embedding_dimension": 768,
    "max_length": 8192
  }
}"#,
        );

        let err = ModelSpec::from_build_info(&build_info_path).unwrap_err();
        assert!(matches!(
            err,
            LTEmbedError::ModelLoad(ModelLoadError::UnsupportedModelFormat { .. })
        ));

        fs::remove_dir_all(temp_dir).unwrap();
    }
}
