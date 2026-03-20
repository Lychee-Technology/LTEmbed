// src/cross_encoder.rs — Safe Rust wrapper for the llama.cpp cross-encoder backend.
// Only compiled when the `ggml-backend` feature is active.

#[cfg(feature = "ggml-backend")]
pub use inner::CrossEncoderEngine;

#[cfg(feature = "ggml-backend")]
mod inner {
    use std::ffi::{CString, NulError};
    use std::path::Path;

    use crate::error::LTEmbedError;
    use crate::ffi::cross_encoder as ffi;

    /// Cross-encoder reranking engine backed by llama.cpp GGUF models.
    ///
    /// Takes a (query, document) pair and returns a raw relevance logit.
    /// Apply `1.0 / (1.0 + (-score).exp())` for a [0, 1] probability.
    pub struct CrossEncoderEngine {
        ptr: *mut ffi::LtCrossEncoderOpaque,
    }

    impl std::fmt::Debug for CrossEncoderEngine {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("CrossEncoderEngine").finish_non_exhaustive()
        }
    }

    // Safety: llama.cpp context is thread-safe for read-only inference.
    unsafe impl Send for CrossEncoderEngine {}
    unsafe impl Sync for CrossEncoderEngine {}

    impl CrossEncoderEngine {
        /// Load a GGUF cross-encoder model from `model_path`.
        pub fn new(model_path: &Path, n_threads: u32) -> Result<Self, LTEmbedError> {
            let path_str = model_path
                .to_str()
                .ok_or_else(|| LTEmbedError::ModelLoad("model path is not valid UTF-8".into()))?;
            let c_path = CString::new(path_str)
                .map_err(|_| LTEmbedError::ModelLoad("model path contains null byte".into()))?;

            let ptr = unsafe { ffi::lt_ce_load(c_path.as_ptr(), n_threads as i32) };
            if ptr.is_null() {
                return Err(LTEmbedError::ModelLoad(format!(
                    "llama.cpp failed to load cross-encoder model: {}",
                    path_str
                )));
            }

            Ok(Self { ptr })
        }

        /// Score a (query, document) pair.
        ///
        /// Returns the raw relevance logit. Higher is more relevant.
        pub fn score(&self, query: &str, document: &str) -> Result<f32, LTEmbedError> {
            let c_query = CString::new(query).map_err(|_: NulError| {
                LTEmbedError::Tokenization("query contains null byte".into())
            })?;
            let c_doc = CString::new(document).map_err(|_: NulError| {
                LTEmbedError::Tokenization("document contains null byte".into())
            })?;

            let score = unsafe { ffi::lt_ce_score(self.ptr, c_query.as_ptr(), c_doc.as_ptr()) };

            if score.is_nan() {
                return Err(LTEmbedError::Inference(
                    "llama.cpp cross-encoder inference failed".into(),
                ));
            }

            Ok(score)
        }
    }

    impl Drop for CrossEncoderEngine {
        fn drop(&mut self) {
            unsafe { ffi::lt_ce_free(self.ptr) }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use static_assertions::assert_impl_all;

        assert_impl_all!(CrossEncoderEngine: Send, Sync);

        #[test]
        fn test_missing_gguf_returns_model_load_error() {
            let result = CrossEncoderEngine::new(Path::new("/nonexistent/ce.gguf"), 1);
            assert!(result.is_err());
            assert!(matches!(result.unwrap_err(), LTEmbedError::ModelLoad(_)));
        }
    }
}
