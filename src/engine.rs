// src/engine.rs

use crate::error::LTEmbedError;
use crate::models::bert::Bert;
use crate::traits::pooling::Pooling;
use crate::traits::tokenizer::{HFTokenizer, Tokenizer};
use crate::utils::l2_normalize;

const MAX_LENGTH: usize = 512;

pub struct ZeroVecEngine {
    bert: Bert,
    tokenizer: HFTokenizer,
    pooling: Box<dyn Pooling>,
}

impl std::fmt::Debug for ZeroVecEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ZeroVecEngine").finish_non_exhaustive()
    }
}

impl ZeroVecEngine {
    /// Initialize the engine from local file paths. Call this once at Lambda cold start.
    pub fn new(
        safetensors_path: &str,
        config_json: &str,
        tokenizer_path: &str,
        pooling: Box<dyn Pooling>,
    ) -> Result<Self, LTEmbedError> {
        let bert = Bert::from_files(safetensors_path, config_json)?;
        let tokenizer = HFTokenizer::from_file(tokenizer_path)?;
        Ok(Self { bert, tokenizer, pooling })
    }

    /// Full inference pipeline: text → L2-normalized 384-dim embedding.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, LTEmbedError> {
        let encoded = self.tokenizer.encode(text, MAX_LENGTH)?;
        let last_hidden_state = self.bert.forward(
            &encoded.input_ids,
            &encoded.token_type_ids,
            &encoded.attention_mask,
        )?;
        let pooled = self.pooling.pool(&last_hidden_state, &encoded.attention_mask)?;
        Ok(l2_normalize(&pooled))
    }

    /// Embed a batch of texts. Returns one vector per input.
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, LTEmbedError> {
        texts.iter().map(|t| self.embed(t)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::pooling::MeanPooling;
    use approx::assert_relative_eq;
    use std::path::Path;
    use static_assertions::assert_impl_all;

    // Compile-time guard: ZeroVecEngine must be Send + Sync to be stored in a
    // `static OnceLock<Result<ZeroVecEngine, String>>`.
    assert_impl_all!(ZeroVecEngine: Send, Sync);

    const SAFETENSORS: &str = "assets/model.safetensors";
    const CONFIG: &str = "assets/config.json";
    const TOKENIZER: &str = "assets/tokenizer.json";

    const DUMMY_CONFIG: &str = r#"{
        "hidden_size": 384, "num_hidden_layers": 12, "num_attention_heads": 12,
        "intermediate_size": 1536, "max_position_embeddings": 512,
        "vocab_size": 30522, "type_vocab_size": 2, "hidden_act": "gelu",
        "layer_norm_eps": 1e-12,
        "hidden_dropout_prob": 0.1,
        "initializer_range": 0.02,
        "pad_token_id": 0,
        "classifier_dropout": null
    }"#;

    fn assets_available() -> bool {
        Path::new(SAFETENSORS).exists()
            && Path::new(CONFIG).exists()
            && Path::new(TOKENIZER).exists()
    }

    fn make_engine() -> ZeroVecEngine {
        let config_str = std::fs::read_to_string(CONFIG).unwrap();
        ZeroVecEngine::new(SAFETENSORS, &config_str, TOKENIZER, Box::new(MeanPooling)).unwrap()
    }

    #[test]
    fn test_missing_model_file_returns_error() {
        let result = ZeroVecEngine::new(
            "/nonexistent/model.safetensors",
            DUMMY_CONFIG,
            "/nonexistent/tokenizer.json",
            Box::new(MeanPooling),
        );
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), LTEmbedError::ModelLoad(_)));
    }

    #[test]
    fn test_embed_returns_unit_vector() {
        if !assets_available() {
            eprintln!("Skipping: model assets not found");
            return;
        }
        let engine = make_engine();
        let v = engine.embed("query: Hello, world!").unwrap();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert_relative_eq!(norm, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_embed_dimension() {
        if !assets_available() {
            eprintln!("Skipping: model assets not found");
            return;
        }
        let engine = make_engine();
        let v = engine.embed("query: test").unwrap();
        assert_eq!(v.len(), 384);
    }

    #[test]
    fn test_embed_batch_matches_individual() {
        if !assets_available() {
            eprintln!("Skipping: model assets not found");
            return;
        }
        let engine = make_engine();
        let texts = vec!["query: foo", "query: bar"];
        let batch = engine.embed_batch(&texts).unwrap();
        let individual = engine.embed(texts[0]).unwrap();
        assert_eq!(batch[0], individual);
    }
}
