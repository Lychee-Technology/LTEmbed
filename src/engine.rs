// src/engine.rs

use crate::error::LTEmbedError;
use crate::models::bert::Bert;
use crate::traits::pooling::Pooling;
use crate::traits::tokenizer::{HFTokenizer, Tokenizer};
use crate::utils::l2_normalize_inplace;

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
        Ok(Self {
            bert,
            tokenizer,
            pooling,
        })
    }

    /// Full inference pipeline: text → L2-normalized 384-dim embedding.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, LTEmbedError> {
        let encoded = self.tokenizer.encode(text, MAX_LENGTH)?;
        let seq_len = encoded.input_ids.len();
        let last_hidden_state = self.bert.forward(
            &encoded.input_ids,
            &encoded.token_type_ids,
            &encoded.attention_mask,
        )?;
        let mut pooled = self.pooling.pool(
            &last_hidden_state,
            seq_len,
            self.bert.hidden_size(),
            &encoded.attention_mask,
        )?;
        l2_normalize_inplace(&mut pooled);
        Ok(pooled)
    }

    /// Embed a batch of texts. Returns one vector per input.
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, LTEmbedError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        if texts.len() == 1 {
            return self.embed(texts[0]).map(|embedding| vec![embedding]);
        }

        let encoded: Vec<_> = texts
            .iter()
            .map(|text| self.tokenizer.encode(text, MAX_LENGTH))
            .collect::<Result<_, _>>()?;
        let batch_size = encoded.len();
        let seq_len = encoded
            .iter()
            .map(|item| item.input_ids.len())
            .max()
            .unwrap_or(0);
        let total_tokens = batch_size * seq_len;
        let pad_token_id = self.bert.pad_token_id();

        let mut input_ids = vec![pad_token_id; total_tokens];
        let mut token_type_ids = vec![0u32; total_tokens];
        let mut attention_mask = vec![0u32; total_tokens];

        for (batch_idx, item) in encoded.iter().enumerate() {
            let row_start = batch_idx * seq_len;
            let row_end = row_start + item.input_ids.len();
            input_ids[row_start..row_end].copy_from_slice(&item.input_ids);
            token_type_ids[row_start..row_end].copy_from_slice(&item.token_type_ids);
            attention_mask[row_start..row_end].copy_from_slice(&item.attention_mask);
        }

        let last_hidden_state = self.bert.forward_batch(
            &input_ids,
            &token_type_ids,
            &attention_mask,
            batch_size,
            seq_len,
        )?;
        let hidden_size = self.bert.hidden_size();

        let mut embeddings = Vec::with_capacity(batch_size);
        for batch_idx in 0..batch_size {
            let state_start = batch_idx * seq_len * hidden_size;
            let state_end = state_start + seq_len * hidden_size;
            let mask_start = batch_idx * seq_len;
            let mask_end = mask_start + seq_len;
            let mut pooled = self.pooling.pool(
                &last_hidden_state[state_start..state_end],
                seq_len,
                hidden_size,
                &attention_mask[mask_start..mask_end],
            )?;
            l2_normalize_inplace(&mut pooled);
            embeddings.push(pooled);
        }

        Ok(embeddings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::pooling::MeanPooling;
    use approx::assert_relative_eq;
    use static_assertions::assert_impl_all;
    use std::path::Path;
    use std::sync::Arc;
    use std::thread;

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

    #[test]
    fn test_embed_batch_mixed_lengths_matches_individual() {
        if !assets_available() {
            eprintln!("Skipping: model assets not found");
            return;
        }
        let engine = make_engine();
        let texts = vec![
            "query: short",
            "query: this is a somewhat longer sentence used to exercise padding behavior",
        ];
        let batch = engine.embed_batch(&texts).unwrap();
        let first = engine.embed(texts[0]).unwrap();
        let second = engine.embed(texts[1]).unwrap();
        assert_eq!(batch[0], first);
        assert_eq!(batch[1], second);
    }

    #[test]
    fn test_embed_is_thread_safe_across_threads() {
        if !assets_available() {
            eprintln!("Skipping: model assets not found");
            return;
        }
        let engine = Arc::new(make_engine());
        let texts = ["query: alpha", "query: beta"];
        let handles: Vec<_> = texts
            .into_iter()
            .map(|text| {
                let engine = Arc::clone(&engine);
                thread::spawn(move || engine.embed(text).unwrap())
            })
            .collect();

        let outputs: Vec<_> = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect();
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].len(), 384);
        assert_eq!(outputs[1].len(), 384);
    }
}
