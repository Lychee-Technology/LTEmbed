// src/traits/tokenizer.rs

use crate::error::LTEmbedError;

/// Output of a tokenization call. All three vecs have the same length.
#[derive(Debug, Clone)]
pub struct TokenizerOutput {
    pub input_ids: Vec<u32>,      // token IDs including [CLS] and [SEP]
    pub attention_mask: Vec<u32>, // 1 = real token, 0 = padding
    pub token_type_ids: Vec<u32>, // all 0 for single-sequence tasks
}

/// Converts raw text into BERT input tensors.
pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str, max_length: usize) -> Result<TokenizerOutput, LTEmbedError>;
}

/// Backed by the HuggingFace `tokenizers` crate (pure Rust).
#[derive(Debug)]
pub struct HFTokenizer {
    inner: tokenizers::Tokenizer,
}

impl HFTokenizer {
    /// Load from a `tokenizer.json` file path.
    pub fn from_file(path: &str) -> Result<Self, LTEmbedError> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| LTEmbedError::ModelLoad(format!("Failed to load tokenizer: {e}")))?;
        Ok(Self { inner })
    }
}

impl Tokenizer for HFTokenizer {
    fn encode(&self, text: &str, max_length: usize) -> Result<TokenizerOutput, LTEmbedError> {
        let encoding = self
            .inner
            .encode(text, true) // true = add special tokens ([CLS], [SEP])
            .map_err(|e| LTEmbedError::Tokenization(e.to_string()))?;

        let input_ids: Vec<u32> = encoding.get_ids().to_vec();
        let attention_mask: Vec<u32> = encoding.get_attention_mask().to_vec();
        let token_type_ids: Vec<u32> = encoding.get_type_ids().to_vec();

        // LTEmbed is a library — do NOT silently truncate the caller's input.
        // Return an explicit error; the caller decides how to handle overlong text.
        if input_ids.len() > max_length {
            return Err(LTEmbedError::InputTooLong {
                tokens: input_ids.len(),
                max: max_length,
            });
        }

        Ok(TokenizerOutput {
            input_ids,
            attention_mask,
            token_type_ids,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    const TOKENIZER_PATH: &str = "assets/tokenizer.json";

    fn tokenizer_available() -> bool {
        Path::new(TOKENIZER_PATH).exists()
    }

    #[test]
    fn test_missing_tokenizer_file_returns_model_load_error() {
        let result = HFTokenizer::from_file("/nonexistent/path/tokenizer.json");
        assert!(result.is_err());
        assert!(
            matches!(result.unwrap_err(), LTEmbedError::ModelLoad(_)),
            "Expected ModelLoad error"
        );
    }

    #[test]
    fn test_encode_produces_aligned_vecs() {
        if !tokenizer_available() {
            eprintln!("Skipping: {} not found", TOKENIZER_PATH);
            return;
        }
        let tok = HFTokenizer::from_file(TOKENIZER_PATH).unwrap();
        let out = tok.encode("query: Hello, world!", 512).unwrap();
        assert!(out.input_ids.len() >= 3);
        assert_eq!(out.input_ids.len(), out.attention_mask.len());
        assert_eq!(out.input_ids.len(), out.token_type_ids.len());
        assert!(out.attention_mask.iter().all(|&m| m == 1));
        assert!(out.token_type_ids.iter().all(|&t| t == 0));
    }

    #[test]
    fn test_encode_overlong_returns_input_too_long_error() {
        if !tokenizer_available() {
            eprintln!("Skipping: {} not found", TOKENIZER_PATH);
            return;
        }
        let tok = HFTokenizer::from_file(TOKENIZER_PATH).unwrap();
        let long_text = "hello world ".repeat(5000); // encodes to >> 512 tokens
        let result = tok.encode(&long_text, 512);
        assert!(result.is_err());
        assert!(
            matches!(result.unwrap_err(), LTEmbedError::InputTooLong { .. }),
            "Expected InputTooLong error"
        );
    }
}
