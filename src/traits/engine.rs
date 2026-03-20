// src/traits/engine.rs
use crate::error::LTEmbedError;

/// Shared interface for all embedding backends.
///
/// Both `ZeroVecEngine` (SafeTensors/pure-Rust) and `LlamaCppEngine` (GGUF/llama.cpp)
/// implement this trait, allowing callers to hold either as `Box<dyn EmbeddingEngine>`
/// and swap backends without changing downstream code.
pub trait EmbeddingEngine: Send + Sync {
    /// Embed a single text. Returns an L2-normalized vector.
    fn embed(&self, text: &str) -> Result<Vec<f32>, LTEmbedError>;

    /// Embed a batch of texts. Returns one L2-normalized vector per input, in order.
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, LTEmbedError>;

    /// Dimension of the output embedding vectors.
    fn embedding_dim(&self) -> usize;
}
