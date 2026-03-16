// src/models/bert.rs

use crate::error::LTEmbedError;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};

#[allow(dead_code)]
pub struct Bert {
    model: BertModel,
    device: Device,
}

impl Bert {
    /// Load from `.safetensors` weights and a `config.json` string.
    /// Uses OS-level mmap — no heap copy of the 130MB weight file.
    pub fn from_files(safetensors_path: &str, config_json: &str) -> Result<Self, LTEmbedError> {
        let device = Device::Cpu;
        let config: BertConfig = serde_json::from_str(config_json)
            .map_err(|e| LTEmbedError::ModelLoad(format!("Bad config JSON: {e}")))?;

        // VarBuilder::from_mmaped_safetensors is the zero-copy mmap path.
        // The `unsafe` block is required by candle's API; the safety invariant is
        // that the file at `safetensors_path` is not modified while the model is live.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[safetensors_path], DType::F32, &device)
                .map_err(|e| LTEmbedError::ModelLoad(format!("Failed to mmap model: {e}")))?
        };

        let model = BertModel::load(vb, &config)
            .map_err(|e| LTEmbedError::ModelLoad(format!("Failed to build BERT: {e}")))?;

        Ok(Self { model, device })
    }

    /// Forward pass. Returns last_hidden_state as [seq_len][hidden_size].
    pub fn forward(
        &self,
        input_ids: &[u32],
        token_type_ids: &[u32],
        attention_mask: &[u32],
    ) -> Result<Vec<Vec<f32>>, LTEmbedError> {
        let seq_len = input_ids.len();

        let to_tensor = |ids: &[u32]| -> Result<Tensor, LTEmbedError> {
            Tensor::from_vec(ids.to_vec(), (1, seq_len), &self.device)
                .map_err(|e| LTEmbedError::Inference(e.to_string()))
        };

        let input_ids_t = to_tensor(input_ids)?;
        let token_type_ids_t = to_tensor(token_type_ids)?;
        let attention_mask_t = to_tensor(attention_mask)?;

        // output shape: [1, seq_len, hidden_size]
        let output = self
            .model
            .forward(&input_ids_t, &token_type_ids_t, Some(&attention_mask_t))
            .map_err(|e| LTEmbedError::Inference(e.to_string()))?;

        // Squeeze the batch dimension → [seq_len, hidden_size]
        // then convert directly to Vec<Vec<f32>> via to_vec2
        let last_hidden: Vec<Vec<f32>> = output
            .squeeze(0)
            .map_err(|e| LTEmbedError::Inference(e.to_string()))?
            .to_vec2::<f32>()
            .map_err(|e| LTEmbedError::Inference(e.to_string()))?;

        Ok(last_hidden)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    const SAFETENSORS_PATH: &str = "assets/model.safetensors";
    const CONFIG_PATH: &str = "assets/config.json";

    fn assets_available() -> bool {
        Path::new(SAFETENSORS_PATH).exists() && Path::new(CONFIG_PATH).exists()
    }

    // Minimal e5-small-v2 config for error-path testing (no real weights needed).
    // Includes all required fields for BertConfig deserialization.
    const DUMMY_CONFIG: &str = r#"{
        "hidden_size": 384,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 1536,
        "max_position_embeddings": 512,
        "vocab_size": 30522,
        "type_vocab_size": 2,
        "hidden_act": "gelu",
        "layer_norm_eps": 1e-12,
        "hidden_dropout_prob": 0.1,
        "initializer_range": 0.02,
        "pad_token_id": 0,
        "classifier_dropout": null
    }"#;

    #[test]
    fn test_missing_safetensors_returns_model_load_error() {
        let result = Bert::from_files("/nonexistent/model.safetensors", DUMMY_CONFIG);
        assert!(result.is_err());
        assert!(
            matches!(result.err().unwrap(), LTEmbedError::ModelLoad(_)),
            "Expected ModelLoad error"
        );
    }

    #[test]
    fn test_forward_output_shape() {
        if !assets_available() {
            eprintln!("Skipping: model assets not found in assets/");
            return;
        }
        let config_str = std::fs::read_to_string(CONFIG_PATH).unwrap();
        let bert = Bert::from_files(SAFETENSORS_PATH, &config_str).unwrap();

        // Short sequence: [CLS]=101, "hello"=7592, [SEP]=102
        let input_ids = vec![101u32, 7592, 102];
        let token_type_ids = vec![0u32; 3];
        let attention_mask = vec![1u32; 3];

        let output = bert
            .forward(&input_ids, &token_type_ids, &attention_mask)
            .unwrap();
        assert_eq!(output.len(), 3); // seq_len = 3
        assert_eq!(output[0].len(), 384); // e5-small-v2 hidden_size = 384
    }
}
