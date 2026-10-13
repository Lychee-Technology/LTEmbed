use std::fs;
use std::path::Path;

use serde::Deserialize;

use crate::error::{LTEmbedError, ModelLoadError};

use super::MAX_LENGTH;
#[cfg(test)]
use super::{DOCUMENT_PREFIX, QUERY_PREFIX, RAW_EMBEDDING_DIMENSION};

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
    /// The spec of the release bundle's `build-info.json`, built from the public constants.
    /// Only unit tests use it; a loaded engine always takes its spec from the bundle.
    #[cfg(test)]
    pub(crate) fn defaults() -> Self {
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
        // The backend always runs llama.cpp MEAN pooling (and checks that the GGUF declares
        // it), so a bundle that asks for anything else, such as a last-token bundle from the
        // previous model, is rejected rather than embedded with the wrong pooling.
        if metadata.pooling != "mean" {
            return Err(LTEmbedError::ModelLoad(
                ModelLoadError::UnsupportedPooling {
                    pooling: metadata.pooling,
                    target: build_info.target_id,
                },
            ));
        }

        // `max_length` sizes the llama.cpp context and is the longest input the engine
        // accepts. A smaller one lowers the memory a long input can take (docs/architecture.md
        // § "Long inputs and memory"); a larger one would run the model past the 8192-token
        // context it was trained for.
        if !(1..=MAX_LENGTH).contains(&metadata.max_length) {
            return Err(LTEmbedError::ModelLoad(ModelLoadError::Metadata(format!(
                "build-info.json max_length {} is outside 1..={MAX_LENGTH}, the model's context \
                 length",
                metadata.max_length
            ))));
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

    /// The `build-info.json` that CI and the release workflow write.
    fn release_build_info() -> serde_json::Value {
        serde_json::json!({
            "target_id": "unsloth/embeddinggemma-2-GGUF",
            "model_metadata": {
                "model_format": "gguf",
                "quant": "UD-Q5_K_XL",
                "pooling": "mean",
                "input_kind": "retrieval",
                "query_prefix": "task: search result | query: ",
                "document_prefix": "title: none | text: ",
                "raw_embedding_dimension": 768,
                "output_embedding_dimension": 768,
                "max_length": 8192
            }
        })
    }

    /// Writes `build_info` to a fresh directory and parses it.
    fn parse(build_info: &serde_json::Value) -> Result<ModelSpec, LTEmbedError> {
        let temp_dir = unique_temp_dir();
        fs::create_dir_all(&temp_dir).unwrap();
        let path = temp_dir.join("build-info.json");
        fs::write(&path, build_info.to_string()).unwrap();
        let spec = ModelSpec::from_build_info(&path);
        fs::remove_dir_all(temp_dir).unwrap();
        spec
    }

    /// The release metadata yields exactly the spec the public constants describe.
    #[test]
    fn test_release_build_info_matches_constants() {
        let spec = parse(&release_build_info()).unwrap();
        let defaults = ModelSpec::defaults();
        assert_eq!(spec.query_prefix, defaults.query_prefix);
        assert_eq!(spec.document_prefix, defaults.document_prefix);
        assert_eq!(
            spec.raw_embedding_dimension,
            defaults.raw_embedding_dimension
        );
        assert_eq!(spec.max_length, defaults.max_length);
        assert_eq!(QUERY_PREFIX, "task: search result | query: ");
        assert_eq!(DOCUMENT_PREFIX, "title: none | text: ");
        assert_eq!(RAW_EMBEDDING_DIMENSION, 768);
        assert_eq!(MAX_LENGTH, 8192);
    }

    /// `input_kind: "text"` is a format-independent alias that remains valid.
    #[test]
    fn test_text_input_kind_alias_is_accepted() {
        let mut build_info = release_build_info();
        build_info["model_metadata"]["input_kind"] = "text".into();
        parse(&build_info).unwrap();
    }

    /// The backend runs mean pooling only. A bundle assembled for the previous model's
    /// last-token pooling must fail to load instead of being embedded with mean pooling.
    #[test]
    fn test_non_mean_pooling_is_rejected() {
        for pooling in ["last_token", "lasttoken", "cls", "none", "Mean", ""] {
            let mut build_info = release_build_info();
            build_info["model_metadata"]["pooling"] = pooling.into();
            match parse(&build_info) {
                Err(LTEmbedError::ModelLoad(ModelLoadError::UnsupportedPooling {
                    pooling: rejected,
                    target,
                })) => {
                    assert_eq!(rejected, pooling);
                    assert_eq!(target, "unsloth/embeddinggemma-2-GGUF");
                }
                other => panic!("pooling {pooling:?}: expected UnsupportedPooling, got {other:?}"),
            }
        }
    }

    #[test]
    fn test_release_provenance_is_informational_only() {
        let mut build_info = release_build_info();
        build_info["model_source"] = serde_json::json!({
            "gguf_repository": "unsloth/embeddinggemma-2-GGUF",
            "gguf_revision": "031f0d4b35536f69ab3509d4893c923264fcf253",
            "model_license": "Apache-2.0",
            "note": "informational; not checked by the loader"
        });
        let spec = parse(&build_info).unwrap();
        assert_eq!(spec.query_prefix, QUERY_PREFIX);
        assert_eq!(spec.document_prefix, DOCUMENT_PREFIX);
        assert_eq!(spec.raw_embedding_dimension, RAW_EMBEDDING_DIMENSION);
        assert_eq!(spec.max_length, MAX_LENGTH);
    }

    /// A smaller `max_length` is a supported way to bound memory; zero or more than the
    /// model's context length is rejected.
    #[test]
    fn test_max_length_must_be_within_the_model_context() {
        for max_length in [1, 64, 2048, MAX_LENGTH] {
            let mut build_info = release_build_info();
            build_info["model_metadata"]["max_length"] = max_length.into();
            assert_eq!(parse(&build_info).unwrap().max_length, max_length);
        }
        for max_length in [0, MAX_LENGTH + 1, 262_144] {
            let mut build_info = release_build_info();
            build_info["model_metadata"]["max_length"] = max_length.into();
            match parse(&build_info) {
                Err(LTEmbedError::ModelLoad(ModelLoadError::Metadata(msg))) => {
                    assert!(msg.contains(&format!("max_length {max_length} ")), "{msg}");
                }
                other => panic!("max_length {max_length}: expected Metadata, got {other:?}"),
            }
        }
    }

    #[test]
    fn test_non_gguf_model_format_is_rejected() {
        let mut build_info = release_build_info();
        build_info["model_metadata"]["model_format"] = "ort".into();
        let err = parse(&build_info).unwrap_err();
        assert!(matches!(
            err,
            LTEmbedError::ModelLoad(ModelLoadError::UnsupportedModelFormat { .. })
        ));
    }
}
