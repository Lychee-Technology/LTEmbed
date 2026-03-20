// src/engine_llama.rs — Safe Rust wrapper for the llama.cpp embedding backend.
// Only compiled when the `ggml-backend` feature is active.

#[cfg(feature = "ggml-backend")]
pub use inner::LlamaCppEngine;

#[cfg(feature = "ggml-backend")]
mod inner {
    use std::ffi::{CString, NulError};
    use std::path::Path;

    use crate::error::LTEmbedError;
    use crate::ffi::embedding as ffi;
    use crate::traits::engine::EmbeddingEngine;

    /// Embedding engine backed by llama.cpp GGUF models.
    ///
    /// Implements the same [`EmbeddingEngine`] trait as [`crate::engine::ZeroVecEngine`],
    /// allowing callers to swap backends without changing downstream code.
    pub struct LlamaCppEngine {
        ptr: *mut ffi::LtEmbedderOpaque,
        dim: usize,
    }

    impl std::fmt::Debug for LlamaCppEngine {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("LlamaCppEngine")
                .field("dim", &self.dim)
                .finish_non_exhaustive()
        }
    }

    // Safety: llama.cpp context is thread-safe for read-only inference.
    unsafe impl Send for LlamaCppEngine {}
    unsafe impl Sync for LlamaCppEngine {}

    impl LlamaCppEngine {
        /// Load a GGUF embedding model from `model_path`.
        ///
        /// `n_threads` controls the number of CPU threads used for inference.
        pub fn new(model_path: &Path, n_threads: u32) -> Result<Self, LTEmbedError> {
            let path_str = model_path
                .to_str()
                .ok_or_else(|| LTEmbedError::ModelLoad("model path is not valid UTF-8".into()))?;
            let c_path = CString::new(path_str)
                .map_err(|_| LTEmbedError::ModelLoad("model path contains null byte".into()))?;

            let ptr = unsafe { ffi::lt_emb_load(c_path.as_ptr(), n_threads as i32) };
            if ptr.is_null() {
                return Err(LTEmbedError::ModelLoad(format!(
                    "llama.cpp failed to load model: {}",
                    path_str
                )));
            }

            let dim = unsafe { ffi::lt_emb_dim(ptr) } as usize;

            Ok(Self { ptr, dim })
        }
    }

    impl EmbeddingEngine for LlamaCppEngine {
        fn embed(&self, text: &str) -> Result<Vec<f32>, LTEmbedError> {
            let c_text = CString::new(text).map_err(|_: NulError| {
                LTEmbedError::Tokenization("input contains null byte".into())
            })?;

            let mut out = vec![0.0f32; self.dim];
            let ret = unsafe {
                ffi::lt_emb_compute(self.ptr, c_text.as_ptr(), out.as_mut_ptr(), self.dim as i32)
            };

            if ret != 0 {
                return Err(LTEmbedError::Inference(
                    "llama.cpp embedding inference failed".into(),
                ));
            }

            Ok(out)
        }

        fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, LTEmbedError> {
            texts.iter().map(|t| self.embed(t)).collect()
        }

        fn embedding_dim(&self) -> usize {
            self.dim
        }
    }

    impl Drop for LlamaCppEngine {
        fn drop(&mut self) {
            unsafe { ffi::lt_emb_free(self.ptr) }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use static_assertions::assert_impl_all;

        assert_impl_all!(LlamaCppEngine: Send, Sync);
        assert_impl_all!(LlamaCppEngine: EmbeddingEngine);

        #[test]
        fn test_missing_gguf_returns_model_load_error() {
            let result = LlamaCppEngine::new(Path::new("/nonexistent/model.gguf"), 1);
            assert!(result.is_err());
            assert!(matches!(result.unwrap_err(), LTEmbedError::ModelLoad(_)));
        }
    }
}
