// src/traits/tokenizer.rs

use crate::error::{LTEmbedError, ModelLoadError};

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

/// Backed by the HuggingFace `tokenizers` crate (pure Rust).
#[derive(Debug)]
pub struct HFTokenizer {
    inner: tokenizers::Tokenizer,
}

fn tokenizer_output_from_encoding(
    encoding: tokenizers::Encoding,
    max_length: usize,
) -> Result<TokenizerOutput, LTEmbedError> {
    let input_ids: Vec<u32> = encoding.get_ids().to_vec();
    let attention_mask: Vec<u32> = encoding.get_attention_mask().to_vec();
    let token_type_ids: Vec<u32> = encoding.get_type_ids().to_vec();

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

fn tokenizer_outputs_from_encodings(
    encodings: Vec<tokenizers::Encoding>,
    max_length: usize,
) -> Result<Vec<TokenizerOutput>, LTEmbedError> {
    encodings
        .into_iter()
        .map(|encoding| tokenizer_output_from_encoding(encoding, max_length))
        .collect()
}

impl HFTokenizer {
    /// Load from a `tokenizer.json` file path.
    pub fn from_file(path: &str) -> Result<Self, LTEmbedError> {
        let inner = tokenizers::Tokenizer::from_file(path).map_err(|e| {
            LTEmbedError::ModelLoad(ModelLoadError::Runtime(format!(
                "Failed to load tokenizer: {e}"
            )))
        })?;
        Ok(Self { inner })
    }

    pub fn encode_batch(
        &self,
        texts: &[String],
        max_length: usize,
    ) -> Result<Vec<TokenizerOutput>, LTEmbedError> {
        let encodings = self
            .inner
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| LTEmbedError::Tokenization(e.to_string()))?;

        tokenizer_outputs_from_encodings(encodings, max_length)
    }
}

impl Tokenizer for HFTokenizer {
    fn encode(&self, text: &str, max_length: usize) -> Result<TokenizerOutput, LTEmbedError> {
        let encoding = self
            .inner
            .encode(text, true) // true = add model-specific special tokens
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
    use tokenizers::Encoding;

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
        let out = tok
            .encode("Query: Hello, world!", crate::engine::MAX_LENGTH)
            .unwrap();
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
        let long_text = "hello world ".repeat(12000);
        let result = tok.encode(&long_text, crate::engine::MAX_LENGTH);
        assert!(result.is_err());
        assert!(
            matches!(result.unwrap_err(), LTEmbedError::InputTooLong { .. }),
            "Expected InputTooLong error"
        );
    }

    #[test]
    fn test_encode_batch_matches_individual_encodes_for_mixed_inputs() {
        if !tokenizer_available() {
            eprintln!("Skipping: {} not found", TOKENIZER_PATH);
            return;
        }

        let tok = HFTokenizer::from_file(TOKENIZER_PATH).unwrap();
        let texts = vec![
            "Query: Hello, world!".to_string(),
            "Document: The quick brown fox jumps over the lazy dog.".to_string(),
            "Query: What is the impact of large language models on software engineering productivity?"
                .to_string(),
        ];

        let batch = tok.encode_batch(&texts, crate::engine::MAX_LENGTH).unwrap();
        let individual = texts
            .iter()
            .map(|text| tok.encode(text, crate::engine::MAX_LENGTH).unwrap())
            .collect::<Vec<_>>();

        assert_eq!(batch.len(), individual.len());
        for (lhs, rhs) in batch.iter().zip(individual.iter()) {
            assert_eq!(lhs.input_ids, rhs.input_ids);
            assert_eq!(lhs.attention_mask, rhs.attention_mask);
            assert_eq!(lhs.token_type_ids, rhs.token_type_ids);
        }
    }

    #[test]
    fn test_encode_batch_overlong_returns_input_too_long_error() {
        if !tokenizer_available() {
            eprintln!("Skipping: {} not found", TOKENIZER_PATH);
            return;
        }

        let tok = HFTokenizer::from_file(TOKENIZER_PATH).unwrap();
        let texts = vec![
            "Query: Hello, world!".to_string(),
            format!("Document: {}", "hello world ".repeat(12000)),
        ];

        let result = tok.encode_batch(&texts, crate::engine::MAX_LENGTH);
        assert!(matches!(result, Err(LTEmbedError::InputTooLong { .. })));
    }

    #[test]
    fn test_tokenizer_outputs_from_encodings_rejects_overlong_items() {
        let encodings = vec![
            Encoding::new(
                vec![1, 2, 3],
                vec![0, 0, 0],
                vec!["a".to_string(), "b".to_string(), "c".to_string()],
                vec![None, None, None],
                vec![(0, 1), (1, 2), (2, 3)],
                vec![0, 0, 0],
                vec![1, 1, 1],
                vec![],
                Default::default(),
            ),
            Encoding::new(
                vec![1, 2, 3, 4],
                vec![0, 0, 0, 0],
                vec![
                    "a".to_string(),
                    "b".to_string(),
                    "c".to_string(),
                    "d".to_string(),
                ],
                vec![None, None, None, None],
                vec![(0, 1), (1, 2), (2, 3), (3, 4)],
                vec![0, 0, 0, 0],
                vec![1, 1, 1, 1],
                vec![],
                Default::default(),
            ),
        ];

        let err = tokenizer_outputs_from_encodings(encodings, 3).unwrap_err();
        assert!(matches!(
            err,
            LTEmbedError::InputTooLong { tokens: 4, max: 3 }
        ));
    }
}
