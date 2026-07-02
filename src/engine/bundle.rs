use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::error::LTEmbedError;

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
            LTEmbedError::ModelLoad(format!(
                "Failed to read build-info metadata at {}: {err}",
                path.display()
            ))
        })?;
        let build_info: BuildInfo = serde_json::from_str(&raw).map_err(|err| {
            LTEmbedError::ModelLoad(format!(
                "Failed to parse build-info metadata at {}: {err}",
                path.display()
            ))
        })?;

        let metadata = build_info.model_metadata;
        if metadata.input_kind != "retrieval" && metadata.input_kind != "text" {
            return Err(LTEmbedError::ModelLoad(format!(
                "Unsupported input_kind '{}' for bundle target '{}'",
                metadata.input_kind, build_info.target_id
            )));
        }
        if metadata.pooling != "last_token" && metadata.pooling != "lasttoken" {
            return Err(LTEmbedError::ModelLoad(format!(
                "Unsupported pooling '{}' for bundle target '{}'",
                metadata.pooling, build_info.target_id
            )));
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

    Err(LTEmbedError::ModelLoad(format!(
        "{label} file not found: {}",
        path.display()
    )))
}

pub(crate) fn resolve_dylib_path(bundle_dir: &Path) -> Option<PathBuf> {
    if let Ok(path) = std::env::var("ORT_DYLIB_PATH") {
        if !path.is_empty() {
            return Some(PathBuf::from(path));
        }
    }
    let bundle_dylib = bundle_dir.join("libonnxruntime.so");
    if bundle_dylib.exists() {
        return Some(bundle_dylib);
    }
    None
}
