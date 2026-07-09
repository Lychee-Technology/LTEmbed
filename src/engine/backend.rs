use crate::error::LTEmbedError;
use crate::traits::tokenizer::TokenizerOutput;

/// Timing breakdown for one backend inference pass (populated only when profiling is
/// requested). Field names mirror the backend-relevant stages of [`super::EmbedBatchProfile`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct BackendRunProfile {
    /// Building backend inputs (batches/tensors), in ms.
    pub tensorize_ms: f64,
    /// Running the model (e.g. `llama_decode`), in ms.
    pub run_ms: f64,
    /// Extracting the pooled embeddings, in ms.
    pub extract_ms: f64,
}

/// An embedding inference backend.
///
/// Implementations own a loaded model and return, for each already-tokenized input, the
/// **raw, un-normalized, last-token-pooled** embedding of length `raw_embedding_dimension()`.
/// The shared engine layer ([`super::EmbeddingEngine`]) owns prefixing, tokenization, and
/// the Matryoshka truncation + optional L2 normalization applied on top.
///
/// This trait is the sole seam for adding future backends; today `LlamaBackend` is the only
/// implementation.
pub(crate) trait EmbeddingBackend: Send + Sync {
    /// Raw embedding width the backend emits before truncation (e.g. 768).
    fn raw_embedding_dimension(&self) -> usize;

    /// One raw, un-normalized pooled embedding (length = `raw_embedding_dimension()`) per
    /// tokenized input, in input order.
    fn embed(
        &self,
        tokenized: &[TokenizerOutput],
        collect_profile: bool,
    ) -> Result<(Vec<Vec<f32>>, Option<BackendRunProfile>), LTEmbedError>;
}
