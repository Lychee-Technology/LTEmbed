use crate::error::LTEmbedError;

/// Timing breakdown for one backend inference pass (populated only when profiling is
/// requested). Field names mirror the backend-relevant stages of [`super::EmbedBatchProfile`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct BackendRunProfile {
    /// Building backend inputs (batches/tensors), in ms.
    pub tensorize_ms: f64,
    /// Running the model (e.g. `llama_encode`), in ms.
    pub run_ms: f64,
    /// Extracting the pooled embeddings, in ms.
    pub extract_ms: f64,
}

/// An embedding inference backend.
///
/// Implementations own a loaded model and return, for each already-tokenized input, the
/// **raw, un-normalized, mean-pooled** embedding of length `raw_embedding_dimension()`.
/// The shared engine layer ([`super::EmbeddingEngine`]) owns prefixing, the input length
/// limit, and the Matryoshka truncation + optional L2 normalization applied on top. The
/// model's own vocabulary tokenizes, so the backend provides tokenization
/// ([`EmbeddingBackend::tokenize`]) and the engine calls it, and the vocabulary also sets the
/// size beyond which the engine rejects a text untokenized
/// ([`EmbeddingBackend::max_input_bytes`]).
///
/// This trait is the sole seam for adding future backends; today `LlamaBackend` is the only
/// implementation.
pub(crate) trait EmbeddingBackend: Send + Sync {
    /// Raw embedding width the backend emits before truncation (768 for EmbeddingGemma 2:
    /// its projected output, not its 512-wide hidden state).
    fn raw_embedding_dimension(&self) -> usize;

    /// The token ids of the whole of `text` in the model's own vocabulary, with the special
    /// tokens the model adds (EmbeddingGemma 2: BOS and EOS). Never truncated: the engine
    /// enforces the length limit.
    fn tokenize(&self, text: &str) -> Result<Vec<u32>, LTEmbedError>;

    /// A size in bytes that no text of at most `max_tokens` tokens exceeds, lowered to the
    /// longest text [`EmbeddingBackend::tokenize`] accepts. The engine rejects a longer text
    /// as too long without tokenizing it, so this must never be below the size of a text that
    /// fits.
    fn max_input_bytes(&self, max_tokens: usize) -> usize;

    /// One raw, un-normalized pooled embedding (length = `raw_embedding_dimension()`) per
    /// tokenized input, in input order.
    fn embed(
        &self,
        tokenized: &[Vec<u32>],
        collect_profile: bool,
    ) -> Result<(Vec<Vec<f32>>, Option<BackendRunProfile>), LTEmbedError>;
}
