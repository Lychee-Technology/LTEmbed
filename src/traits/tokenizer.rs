// src/traits/tokenizer.rs
//
// `EmbeddingEngine` does not go through this trait: it tokenizes with the GGUF's own
// vocabulary (src/engine/llama), so the library has no tokenizer.json and no `tokenizers`
// dependency. `benchmarking::scenario_token_lengths` takes any implementation.

use crate::error::LTEmbedError;

/// Output of a tokenization call. All three vecs have the same length.
#[derive(Debug, Clone)]
pub struct TokenizerOutput {
    pub input_ids: Vec<u32>, // token IDs including model-specific special tokens
    pub attention_mask: Vec<u32>, // 1 = real token, 0 = padding
    pub token_type_ids: Vec<u32>, // present for models that declare segment IDs
}

/// Converts raw text into model input tensors.
pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str, max_length: usize) -> Result<TokenizerOutput, LTEmbedError>;
}
