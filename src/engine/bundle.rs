use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::error::{LTEmbedError, ModelLoadError};

use super::ort_init::env_dylib_path;
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
    #[allow(dead_code)]
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

pub(crate) fn resolve_dylib_path(bundle_dir: &Path) -> Option<PathBuf> {
    env_dylib_path().or_else(|| {
        let bundle_dylib = bundle_dir.join("libonnxruntime.so");
        bundle_dylib.exists().then_some(bundle_dylib)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_dir() -> PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let counter = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("ltembed-bundle-tests-{nanos}-{counter}"))
    }

    #[test]
    fn test_legacy_q4f16_metadata_aliases_are_accepted() {
        let temp_dir = unique_temp_dir();
        fs::create_dir_all(&temp_dir).unwrap();
        let build_info_path = temp_dir.join("build-info.json");

        fs::write(
            &build_info_path,
            r#"{
  "target_id": "legacy-q4f16",
  "model_metadata": {
    "model_format": "ort",
    "pooling": "lasttoken",
    "input_kind": "text",
    "query_prefix": "Query: ",
    "document_prefix": "Document: ",
    "raw_embedding_dimension": 768,
    "output_embedding_dimension": 768,
    "max_length": 8192
  }
}"#,
        )
        .unwrap();

        let spec = ModelSpec::from_build_info(&build_info_path).unwrap();

        assert_eq!(spec.query_prefix, "Query: ");
        assert_eq!(spec.document_prefix, "Document: ");
        assert_eq!(spec.raw_embedding_dimension, RAW_EMBEDDING_DIMENSION);
        assert_eq!(spec.max_length, MAX_LENGTH);

        fs::remove_dir_all(temp_dir).unwrap();
    }
}
