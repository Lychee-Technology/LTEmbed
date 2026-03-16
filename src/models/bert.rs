// src/models/bert.rs
//
// From-scratch BERT inference engine using safetensors + matrixmultiply.
// No candle dependency.

use crate::error::LTEmbedError;
use safetensors::SafeTensors;
use std::fs;

// ── Config ────────────────────────────────────────────────────────────────────

#[derive(serde::Deserialize)]
struct BertConfig {
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
}

// ── Weight structs ────────────────────────────────────────────────────────────

struct LayerWeights {
    q_weight: Vec<f32>,
    q_bias: Vec<f32>,
    k_weight: Vec<f32>,
    k_bias: Vec<f32>,
    v_weight: Vec<f32>,
    v_bias: Vec<f32>,
    attn_out_weight: Vec<f32>,
    attn_out_bias: Vec<f32>,
    attn_ln_weight: Vec<f32>,
    attn_ln_bias: Vec<f32>,
    inter_weight: Vec<f32>,
    inter_bias: Vec<f32>,
    out_weight: Vec<f32>,
    out_bias: Vec<f32>,
    out_ln_weight: Vec<f32>,
    out_ln_bias: Vec<f32>,
}

pub struct Bert {
    config: BertConfig,
    word_emb: Vec<f32>,
    pos_emb: Vec<f32>,
    type_emb: Vec<f32>,
    emb_ln_weight: Vec<f32>,
    emb_ln_bias: Vec<f32>,
    layers: Vec<LayerWeights>,
}

// ── Helper: load tensor from SafeTensors into owned Vec<f32> ─────────────────

fn load_tensor(st: &SafeTensors, name: &str) -> Result<Vec<f32>, LTEmbedError> {
    let view = st
        .tensor(name)
        .map_err(|e| LTEmbedError::ModelLoad(format!("Missing tensor '{name}': {e}")))?;
    let data_u8 = view.data();
    let data_f32: &[f32] = bytemuck::cast_slice(data_u8);
    Ok(data_f32.to_vec())
}

// ── Math helpers ──────────────────────────────────────────────────────────────

/// Dense layer: out[i] = sum_j(x[j] * weight[i*input_size + j]) + bias[i]
/// weight is row-major [output_size, input_size]
#[allow(dead_code)]
fn linear(x: &[f32], weight: &[f32], bias: &[f32], output: &mut [f32]) {
    let input_size = x.len();
    let output_size = bias.len();
    debug_assert_eq!(weight.len(), output_size * input_size);
    debug_assert_eq!(output.len(), output_size);

    // Use matrixmultiply::sgemm: C = alpha*A*B + beta*C
    // A = weight [output_size × input_size], B = x as column [input_size × 1]
    // C = output [output_size × 1]
    unsafe {
        matrixmultiply::sgemm(
            output_size,         // m
            input_size,          // k
            1,                   // n
            1.0,                 // alpha
            weight.as_ptr(),     // A
            input_size as isize, // rsa (row stride of A)
            1,                   // csa (col stride of A)
            x.as_ptr(),          // B
            1,                   // rsb
            1,                   // csb
            0.0,                 // beta
            output.as_mut_ptr(), // C
            1,                   // rsc
            1,                   // csc
        );
    }
    // Add bias
    for (o, b) in output.iter_mut().zip(bias.iter()) {
        *o += b;
    }
}

/// Batched linear: each row of x_rows is processed independently.
/// x_rows: [batch, input_size] row-major
/// weight: [output_size, input_size] row-major
/// out: [batch, output_size] row-major
fn linear_batch(
    x_rows: &[f32],
    batch: usize,
    input_size: usize,
    weight: &[f32],
    bias: &[f32],
    out: &mut [f32],
) {
    let output_size = bias.len();
    debug_assert_eq!(x_rows.len(), batch * input_size);
    debug_assert_eq!(out.len(), batch * output_size);
    debug_assert_eq!(weight.len(), output_size * input_size);

    // sgemm: C[batch×output_size] = A[batch×input_size] * B[input_size×output_size]
    // weight is [output_size×input_size], so we need its transpose.
    // Equivalently compute: C = x_rows * weight^T
    // sgemm(m,k,n, alpha, A[m×k], rsa,csa, B[k×n], rsb,csb, beta, C[m×n], rsc,csc)
    // A = x_rows [batch × input_size], B = weight^T [input_size × output_size]
    // weight^T has row stride=1, col stride=input_size
    unsafe {
        matrixmultiply::sgemm(
            batch,
            input_size,
            output_size,
            1.0,
            x_rows.as_ptr(),
            input_size as isize, // rsa
            1,                   // csa
            weight.as_ptr(),
            1,                   // rsb (weight^T: iterate rows of weight^T = columns of weight)
            input_size as isize, // csb
            0.0,
            out.as_mut_ptr(),
            output_size as isize, // rsc
            1,                    // csc
        );
    }
    // Add bias to each row
    for row in 0..batch {
        let offset = row * output_size;
        for (j, b) in bias.iter().enumerate() {
            out[offset + j] += b;
        }
    }
}

/// Layer norm in-place: (x - mean) / sqrt(var + eps) * weight + bias
fn layer_norm(x: &mut [f32], weight: &[f32], bias: &[f32], eps: f32) {
    let n = x.len() as f32;
    let mean: f32 = x.iter().sum::<f32>() / n;
    let var: f32 = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv_std = 1.0 / (var + eps).sqrt();
    for (i, v) in x.iter_mut().enumerate() {
        *v = (*v - mean) * inv_std * weight[i] + bias[i];
    }
}

/// Layer norm applied to each row of a 2D slice [rows × hidden].
fn layer_norm_rows(x: &mut [f32], rows: usize, hidden: usize, weight: &[f32], bias: &[f32]) {
    for row in 0..rows {
        let start = row * hidden;
        layer_norm(&mut x[start..start + hidden], weight, bias, 1e-12);
    }
}

/// GELU activation (approximate tanh variant) in-place.
fn gelu(x: &mut [f32]) {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6; // sqrt(2/pi)
    for v in x.iter_mut() {
        let x3 = *v * *v * *v;
        let inner = SQRT_2_OVER_PI * (*v + 0.044715 * x3);
        *v = *v * 0.5 * (1.0 + inner.tanh());
    }
}

/// Softmax in-place over a slice.
fn softmax(x: &mut [f32]) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

/// Embedding lookup: copy embedding row for each token id into output.
/// output: [ids.len() × hidden_size]
fn embed(ids: &[u32], weight: &[f32], hidden_size: usize, output: &mut [f32]) {
    for (i, &id) in ids.iter().enumerate() {
        let src_start = id as usize * hidden_size;
        let dst_start = i * hidden_size;
        output[dst_start..dst_start + hidden_size]
            .copy_from_slice(&weight[src_start..src_start + hidden_size]);
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

impl Bert {
    /// Load from `.safetensors` weights and a `config.json` string.
    pub fn from_files(safetensors_path: &str, config_json: &str) -> Result<Self, LTEmbedError> {
        let config: BertConfig = serde_json::from_str(config_json)
            .map_err(|e| LTEmbedError::ModelLoad(format!("Bad config JSON: {e}")))?;

        let file_bytes = fs::read(safetensors_path)
            .map_err(|e| LTEmbedError::ModelLoad(format!("Failed to read model file: {e}")))?;

        let st = SafeTensors::deserialize(&file_bytes)
            .map_err(|e| LTEmbedError::ModelLoad(format!("Failed to parse safetensors: {e}")))?;

        // Embeddings
        let word_emb = load_tensor(&st, "embeddings.word_embeddings.weight")?;
        let pos_emb = load_tensor(&st, "embeddings.position_embeddings.weight")?;
        let type_emb = load_tensor(&st, "embeddings.token_type_embeddings.weight")?;
        let emb_ln_weight = load_tensor(&st, "embeddings.LayerNorm.weight")?;
        let emb_ln_bias = load_tensor(&st, "embeddings.LayerNorm.bias")?;

        // Encoder layers
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let p = format!("encoder.layer.{i}");
            let layer = LayerWeights {
                q_weight: load_tensor(&st, &format!("{p}.attention.self.query.weight"))?,
                q_bias: load_tensor(&st, &format!("{p}.attention.self.query.bias"))?,
                k_weight: load_tensor(&st, &format!("{p}.attention.self.key.weight"))?,
                k_bias: load_tensor(&st, &format!("{p}.attention.self.key.bias"))?,
                v_weight: load_tensor(&st, &format!("{p}.attention.self.value.weight"))?,
                v_bias: load_tensor(&st, &format!("{p}.attention.self.value.bias"))?,
                attn_out_weight: load_tensor(&st, &format!("{p}.attention.output.dense.weight"))?,
                attn_out_bias: load_tensor(&st, &format!("{p}.attention.output.dense.bias"))?,
                attn_ln_weight: load_tensor(
                    &st,
                    &format!("{p}.attention.output.LayerNorm.weight"),
                )?,
                attn_ln_bias: load_tensor(&st, &format!("{p}.attention.output.LayerNorm.bias"))?,
                inter_weight: load_tensor(&st, &format!("{p}.intermediate.dense.weight"))?,
                inter_bias: load_tensor(&st, &format!("{p}.intermediate.dense.bias"))?,
                out_weight: load_tensor(&st, &format!("{p}.output.dense.weight"))?,
                out_bias: load_tensor(&st, &format!("{p}.output.dense.bias"))?,
                out_ln_weight: load_tensor(&st, &format!("{p}.output.LayerNorm.weight"))?,
                out_ln_bias: load_tensor(&st, &format!("{p}.output.LayerNorm.bias"))?,
            };
            layers.push(layer);
        }

        Ok(Self {
            config,
            word_emb,
            pos_emb,
            type_emb,
            emb_ln_weight,
            emb_ln_bias,
            layers,
        })
    }

    /// Forward pass. Returns last_hidden_state as [seq_len][hidden_size].
    pub fn forward(
        &self,
        input_ids: &[u32],
        token_type_ids: &[u32],
        attention_mask: &[u32],
    ) -> Result<Vec<Vec<f32>>, LTEmbedError> {
        let seq_len = input_ids.len();
        let hidden = self.config.hidden_size;
        let num_heads = self.config.num_attention_heads;
        let head_dim = hidden / num_heads;
        let intermediate = self.config.intermediate_size;

        // ── 1. Embeddings ──────────────────────────────────────────────────────
        // x[seq × hidden] = word_emb + pos_emb + type_emb
        let mut x = vec![0.0f32; seq_len * hidden];

        // word embeddings
        embed(input_ids, &self.word_emb, hidden, &mut x);

        // add positional embeddings
        for (i, pos_row) in x.chunks_mut(hidden).enumerate() {
            let pos_start = i * hidden;
            for (j, v) in pos_row.iter_mut().enumerate() {
                *v += self.pos_emb[pos_start + j];
            }
        }

        // add token type embeddings
        for (i, type_row) in x.chunks_mut(hidden).enumerate() {
            let type_id = token_type_ids[i] as usize;
            let type_start = type_id * hidden;
            for (j, v) in type_row.iter_mut().enumerate() {
                *v += self.type_emb[type_start + j];
            }
        }

        // embedding layer norm
        layer_norm_rows(
            &mut x,
            seq_len,
            hidden,
            &self.emb_ln_weight,
            &self.emb_ln_bias,
        );

        // ── 2. Encoder layers ──────────────────────────────────────────────────
        for layer in &self.layers {
            // ── a. Self-attention ──────────────────────────────────────────────

            // Q, K, V projections: [seq × hidden]
            let mut q = vec![0.0f32; seq_len * hidden];
            let mut k = vec![0.0f32; seq_len * hidden];
            let mut v = vec![0.0f32; seq_len * hidden];

            linear_batch(&x, seq_len, hidden, &layer.q_weight, &layer.q_bias, &mut q);
            linear_batch(&x, seq_len, hidden, &layer.k_weight, &layer.k_bias, &mut k);
            linear_batch(&x, seq_len, hidden, &layer.v_weight, &layer.v_bias, &mut v);

            // Multi-head attention
            // scores[h][i][j] = sum_d(q[i,h,d] * k[j,h,d]) / sqrt(head_dim)
            // q layout: [seq, num_heads, head_dim] — q[i*hidden + h*head_dim + d]
            let scale = 1.0 / (head_dim as f32).sqrt();
            let mut attn_out = vec![0.0f32; seq_len * hidden];

            for h in 0..num_heads {
                let head_offset = h * head_dim;

                // Compute scores[seq_i][seq_j] for this head
                let mut scores = vec![0.0f32; seq_len * seq_len];
                for i in 0..seq_len {
                    let q_off = i * hidden + head_offset;
                    for j in 0..seq_len {
                        let k_off = j * hidden + head_offset;
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += q[q_off + d] * k[k_off + d];
                        }
                        scores[i * seq_len + j] = dot * scale;
                    }
                }

                // Apply attention mask
                for j in 0..seq_len {
                    let mask_val = attention_mask[j] as f32;
                    if mask_val == 0.0 {
                        for i in 0..seq_len {
                            scores[i * seq_len + j] += -10000.0;
                        }
                    }
                }

                // Softmax over j for each i
                for i in 0..seq_len {
                    softmax(&mut scores[i * seq_len..(i + 1) * seq_len]);
                }

                // Weighted sum of V: attn_out[i, h, d] = sum_j(scores[i,j] * v[j,h,d])
                for i in 0..seq_len {
                    let out_off = i * hidden + head_offset;
                    for d in 0..head_dim {
                        let mut acc = 0.0f32;
                        for j in 0..seq_len {
                            acc += scores[i * seq_len + j] * v[j * hidden + head_offset + d];
                        }
                        attn_out[out_off + d] = acc;
                    }
                }
            }

            // Output projection: [seq × hidden] → [seq × hidden]
            let mut attn_proj = vec![0.0f32; seq_len * hidden];
            linear_batch(
                &attn_out,
                seq_len,
                hidden,
                &layer.attn_out_weight,
                &layer.attn_out_bias,
                &mut attn_proj,
            );

            // Residual + LayerNorm
            for i in 0..seq_len * hidden {
                x[i] += attn_proj[i];
            }
            layer_norm_rows(
                &mut x,
                seq_len,
                hidden,
                &layer.attn_ln_weight,
                &layer.attn_ln_bias,
            );

            // ── b. FFN ─────────────────────────────────────────────────────────

            // Intermediate projection: [seq × hidden] → [seq × intermediate]
            let mut inter = vec![0.0f32; seq_len * intermediate];
            linear_batch(
                &x,
                seq_len,
                hidden,
                &layer.inter_weight,
                &layer.inter_bias,
                &mut inter,
            );

            // GELU activation
            gelu(&mut inter);

            // Output projection: [seq × intermediate] → [seq × hidden]
            let mut ffn_out = vec![0.0f32; seq_len * hidden];
            linear_batch(
                &inter,
                seq_len,
                intermediate,
                &layer.out_weight,
                &layer.out_bias,
                &mut ffn_out,
            );

            // Residual + LayerNorm
            for i in 0..seq_len * hidden {
                x[i] += ffn_out[i];
            }
            layer_norm_rows(
                &mut x,
                seq_len,
                hidden,
                &layer.out_ln_weight,
                &layer.out_ln_bias,
            );
        }

        // ── 3. Convert flat buffer to Vec<Vec<f32>> ────────────────────────────
        let result = x.chunks(hidden).map(|row| row.to_vec()).collect();

        Ok(result)
    }
}

// ── Unit tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    const SAFETENSORS_PATH: &str = "assets/model.safetensors";
    const CONFIG_PATH: &str = "assets/config.json";

    fn assets_available() -> bool {
        Path::new(SAFETENSORS_PATH).exists() && Path::new(CONFIG_PATH).exists()
    }

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

        let input_ids = vec![101u32, 7592, 102];
        let token_type_ids = vec![0u32; 3];
        let attention_mask = vec![1u32; 3];

        let output = bert
            .forward(&input_ids, &token_type_ids, &attention_mask)
            .unwrap();
        assert_eq!(output.len(), 3);
        assert_eq!(output[0].len(), 384);
    }

    #[test]
    fn test_layer_norm() {
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0];
        let weight = vec![1.0f32; 4];
        let bias = vec![0.0f32; 4];
        layer_norm(&mut x, &weight, &bias, 1e-12);
        let mean: f32 = x.iter().sum::<f32>() / 4.0;
        assert!(
            mean.abs() < 1e-5,
            "layer norm mean should be ~0, got {mean}"
        );
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let mut x = vec![1.0f32, 2.0, 3.0];
        softmax(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "softmax sum={sum}");
    }

    #[test]
    fn test_linear_simple() {
        // 2x2 identity weight, zero bias
        let x = vec![3.0f32, 4.0];
        let weight = vec![1.0f32, 0.0, 0.0, 1.0]; // row-major [2,2]
        let bias = vec![0.0f32; 2];
        let mut out = vec![0.0f32; 2];
        linear(&x, &weight, &bias, &mut out);
        assert_eq!(out, vec![3.0, 4.0]);
    }
}
