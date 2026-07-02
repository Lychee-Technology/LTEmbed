use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::error::{LTEmbedError, ModelLoadError};

#[derive(Debug, Clone, PartialEq, Eq)]
enum OrtInitSource {
    System,
    DynamicLibrary(PathBuf),
}

pub(crate) fn ensure_ort_initialized(dylib_path: Option<&Path>) -> Result<(), LTEmbedError> {
    static INIT: OnceLock<Result<OrtInitSource, String>> = OnceLock::new();

    let requested = resolve_ort_source(dylib_path);
    let initialized = INIT.get_or_init(|| match &requested {
        OrtInitSource::DynamicLibrary(path) => ort::init_from(path)
            .map_err(|err| format!("Failed to load ORT dynamic library: {err}"))
            .map(|builder| {
                let _ = builder.commit();
                requested.clone()
            }),
        OrtInitSource::System => Ok(OrtInitSource::System),
    });

    match initialized {
        Ok(source) if *source == requested => Ok(()),
        Ok(OrtInitSource::DynamicLibrary(_)) if requested == OrtInitSource::System => Ok(()),
        Ok(source) => Err(LTEmbedError::ModelLoad(ModelLoadError::Runtime(format!(
            "ORT is already initialized for {:?}, cannot reinitialize with {:?}",
            source, requested
        )))),
        Err(message) => Err(LTEmbedError::ModelLoad(ModelLoadError::Runtime(
            message.clone(),
        ))),
    }
}

fn resolve_ort_source(dylib_path: Option<&Path>) -> OrtInitSource {
    if let Some(path) = dylib_path {
        return OrtInitSource::DynamicLibrary(path.to_path_buf());
    }

    match env_dylib_path() {
        Some(path) => OrtInitSource::DynamicLibrary(path),
        None => OrtInitSource::System,
    }
}

pub(crate) fn env_dylib_path() -> Option<PathBuf> {
    match std::env::var("ORT_DYLIB_PATH") {
        Ok(path) if !path.is_empty() => Some(PathBuf::from(path)),
        _ => None,
    }
}
