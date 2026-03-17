// src/models/bert.rs
//
// From-scratch BERT inference engine using safetensors + matrixmultiply.
// No candle dependency.

use crate::error::LTEmbedError;
use memmap2::Mmap;
use safetensors::SafeTensors;
use std::fs::File;
use std::sync::{Arc, Mutex};

// ── Config ────────────────────────────────────────────────────────────────────

#[derive(serde::Deserialize)]
struct BertConfig {
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    max_position_embeddings: usize,
    #[serde(default)]
    pad_token_id: u32,
}

// ── Scratch buffers ───────────────────────────────────────────────────────────

struct Scratch {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn_out: Vec<f32>,
    scores: Vec<f32>,
    attn_proj: Vec<f32>,
    inter: Vec<f32>,
    ffn_out: Vec<f32>,
}

// ── Weight structs ────────────────────────────────────────────────────────────

struct LayerWeights {
    q_weight: TensorData,
    q_bias: TensorData,
    k_weight: TensorData,
    k_bias: TensorData,
    v_weight: TensorData,
    v_bias: TensorData,
    attn_out_weight: TensorData,
    attn_out_bias: TensorData,
    attn_ln_weight: TensorData,
    attn_ln_bias: TensorData,
    inter_weight: TensorData,
    inter_bias: TensorData,
    out_weight: TensorData,
    out_bias: TensorData,
    out_ln_weight: TensorData,
    out_ln_bias: TensorData,
}

pub struct Bert {
    config: BertConfig,
    word_emb: TensorData,
    pos_emb: TensorData,
    type_emb: TensorData,
    emb_ln_weight: TensorData,
    emb_ln_bias: TensorData,
    layers: Vec<LayerWeights>,
    scratch: Mutex<Scratch>,
}

#[derive(Clone)]
struct TensorData {
    mmap: Arc<Mmap>,
    offset: usize,
    len_bytes: usize,
}

impl TensorData {
    fn as_f32(&self) -> &[f32] {
        bytemuck::cast_slice(&self.mmap[self.offset..self.offset + self.len_bytes])
    }
}

// ── Helper: load tensor from SafeTensors into mmap-backed view ───────────────

fn load_tensor_data(
    st: &SafeTensors,
    mmap: &Arc<Mmap>,
    name: &str,
) -> Result<TensorData, LTEmbedError> {
    let view = st
        .tensor(name)
        .map_err(|e| LTEmbedError::ModelLoad(format!("Missing tensor '{name}': {e}")))?;
    let data_u8 = view.data();
    let base_ptr = mmap.as_ptr() as usize;
    let data_ptr = data_u8.as_ptr() as usize;
    let offset = data_ptr
        .checked_sub(base_ptr)
        .ok_or_else(|| LTEmbedError::ModelLoad(format!("Tensor '{name}' is not in mmap")))?;
    Ok(TensorData {
        mmap: Arc::clone(mmap),
        offset,
        len_bytes: data_u8.len(),
    })
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
    pub fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    pub fn pad_token_id(&self) -> u32 {
        self.config.pad_token_id
    }

    /// Load from `.safetensors` weights and a `config.json` string.
    pub fn from_files(safetensors_path: &str, config_json: &str) -> Result<Self, LTEmbedError> {
        let config: BertConfig = serde_json::from_str(config_json)
            .map_err(|e| LTEmbedError::ModelLoad(format!("Bad config JSON: {e}")))?;

        let file = File::open(safetensors_path)
            .map_err(|e| LTEmbedError::ModelLoad(format!("Failed to open model file: {e}")))?;
        let mmap = Arc::new(unsafe {
            Mmap::map(&file)
                .map_err(|e| LTEmbedError::ModelLoad(format!("Failed to mmap model file: {e}")))?
        });

        let st = SafeTensors::deserialize(&mmap)
            .map_err(|e| LTEmbedError::ModelLoad(format!("Failed to parse safetensors: {e}")))?;

        // Embeddings
        let word_emb = load_tensor_data(&st, &mmap, "embeddings.word_embeddings.weight")?;
        let pos_emb = load_tensor_data(&st, &mmap, "embeddings.position_embeddings.weight")?;
        let type_emb = load_tensor_data(&st, &mmap, "embeddings.token_type_embeddings.weight")?;
        let emb_ln_weight = load_tensor_data(&st, &mmap, "embeddings.LayerNorm.weight")?;
        let emb_ln_bias = load_tensor_data(&st, &mmap, "embeddings.LayerNorm.bias")?;

        // Encoder layers
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let p = format!("encoder.layer.{i}");
            let layer = LayerWeights {
                q_weight: load_tensor_data(
                    &st,
                    &mmap,
                    &format!("{p}.attention.self.query.weight"),
                )?,
                q_bias: load_tensor_data(&st, &mmap, &format!("{p}.attention.self.query.bias"))?,
                k_weight: load_tensor_data(&st, &mmap, &format!("{p}.attention.self.key.weight"))?,
                k_bias: load_tensor_data(&st, &mmap, &format!("{p}.attention.self.key.bias"))?,
                v_weight: load_tensor_data(
                    &st,
                    &mmap,
                    &format!("{p}.attention.self.value.weight"),
                )?,
                v_bias: load_tensor_data(&st, &mmap, &format!("{p}.attention.self.value.bias"))?,
                attn_out_weight: load_tensor_data(
                    &st,
                    &mmap,
                    &format!("{p}.attention.output.dense.weight"),
                )?,
                attn_out_bias: load_tensor_data(
                    &st,
                    &mmap,
                    &format!("{p}.attention.output.dense.bias"),
                )?,
                attn_ln_weight: load_tensor_data(
                    &st,
                    &mmap,
                    &format!("{p}.attention.output.LayerNorm.weight"),
                )?,
                attn_ln_bias: load_tensor_data(
                    &st,
                    &mmap,
                    &format!("{p}.attention.output.LayerNorm.bias"),
                )?,
                inter_weight: load_tensor_data(
                    &st,
                    &mmap,
                    &format!("{p}.intermediate.dense.weight"),
                )?,
                inter_bias: load_tensor_data(&st, &mmap, &format!("{p}.intermediate.dense.bias"))?,
                out_weight: load_tensor_data(&st, &mmap, &format!("{p}.output.dense.weight"))?,
                out_bias: load_tensor_data(&st, &mmap, &format!("{p}.output.dense.bias"))?,
                out_ln_weight: load_tensor_data(
                    &st,
                    &mmap,
                    &format!("{p}.output.LayerNorm.weight"),
                )?,
                out_ln_bias: load_tensor_data(&st, &mmap, &format!("{p}.output.LayerNorm.bias"))?,
            };
            layers.push(layer);
        }

        // Pre-allocate scratch buffers sized for max_position_embeddings
        let max_seq = config.max_position_embeddings;
        let scratch = Scratch {
            q: vec![0.0f32; max_seq * config.hidden_size],
            k: vec![0.0f32; max_seq * config.hidden_size],
            v: vec![0.0f32; max_seq * config.hidden_size],
            attn_out: vec![0.0f32; max_seq * config.hidden_size],
            scores: vec![0.0f32; max_seq * max_seq],
            attn_proj: vec![0.0f32; max_seq * config.hidden_size],
            inter: vec![0.0f32; max_seq * config.intermediate_size],
            ffn_out: vec![0.0f32; max_seq * config.hidden_size],
        };

        Ok(Self {
            config,
            word_emb,
            pos_emb,
            type_emb,
            emb_ln_weight,
            emb_ln_bias,
            layers,
            scratch: Mutex::new(scratch),
        })
    }

    /// Forward pass. Returns last_hidden_state as a flat [seq_len * hidden_size] row-major buffer.
    pub fn forward(
        &self,
        input_ids: &[u32],
        token_type_ids: &[u32],
        attention_mask: &[u32],
    ) -> Result<Vec<f32>, LTEmbedError> {
        let seq_len = input_ids.len();
        let hidden = self.config.hidden_size;
        let num_heads = self.config.num_attention_heads;
        let head_dim = hidden / num_heads;
        let intermediate = self.config.intermediate_size;

        // ── 1. Embeddings ──────────────────────────────────────────────────────
        // x[seq × hidden] = word_emb + pos_emb + type_emb
        let mut x = vec![0.0f32; seq_len * hidden];

        // word embeddings
        embed(input_ids, self.word_emb.as_f32(), hidden, &mut x);

        // add positional embeddings
        for (i, pos_row) in x.chunks_mut(hidden).enumerate() {
            let pos_start = i * hidden;
            for (j, v) in pos_row.iter_mut().enumerate() {
                *v += self.pos_emb.as_f32()[pos_start + j];
            }
        }

        // add token type embeddings
        for (i, type_row) in x.chunks_mut(hidden).enumerate() {
            let type_id = token_type_ids[i] as usize;
            let type_start = type_id * hidden;
            for (j, v) in type_row.iter_mut().enumerate() {
                *v += self.type_emb.as_f32()[type_start + j];
            }
        }

        // embedding layer norm
        layer_norm_rows(
            &mut x,
            seq_len,
            hidden,
            self.emb_ln_weight.as_f32(),
            self.emb_ln_bias.as_f32(),
        );

        // ── 2. Encoder layers ──────────────────────────────────────────────────
        let mut sc = self.scratch.lock().unwrap();
        let seq_hidden = seq_len * hidden;
        let seq_inter = seq_len * intermediate;
        let seq_sq = seq_len * seq_len;

        for layer in &self.layers {
            // ── a. Self-attention ──────────────────────────────────────────────

            // Q, K, V projections: [seq × hidden]
            sc.q[..seq_hidden].fill(0.0);
            sc.k[..seq_hidden].fill(0.0);
            sc.v[..seq_hidden].fill(0.0);

            linear_batch(
                &x,
                seq_len,
                hidden,
                layer.q_weight.as_f32(),
                layer.q_bias.as_f32(),
                &mut sc.q[..seq_hidden],
            );
            linear_batch(
                &x,
                seq_len,
                hidden,
                layer.k_weight.as_f32(),
                layer.k_bias.as_f32(),
                &mut sc.k[..seq_hidden],
            );
            linear_batch(
                &x,
                seq_len,
                hidden,
                layer.v_weight.as_f32(),
                layer.v_bias.as_f32(),
                &mut sc.v[..seq_hidden],
            );

            // Multi-head attention using GEMM
            // scores[h][i][j] = sum_d(q[i,h,d] * k[j,h,d]) / sqrt(head_dim)
            // q layout: [seq, num_heads, head_dim] — q[i*hidden + h*head_dim + d]
            let scale = 1.0 / (head_dim as f32).sqrt();
            sc.attn_out[..seq_hidden].fill(0.0);

            // Allocate scores once outside the head loop
            sc.scores[..seq_sq].fill(0.0);

            for h in 0..num_heads {
                // Reset scores for this head
                sc.scores[..seq_sq].fill(0.0);

                // scores_h = Q_h @ K_h^T  scaled by 1/sqrt(head_dim)
                // A = Q_h: logical [seq_len, head_dim], pointer = q[h*head_dim..], rsa=hidden, csa=1
                // B = K_h^T: logical [head_dim, seq_len], same K data but transposed via strides:
                //   rsb=1 (moving "down a row of K_h^T" = moving +1 in K_h's column dimension)
                //   csb=hidden (moving "right in K_h^T" = moving to next row of K_h = +hidden in K)
                unsafe {
                    matrixmultiply::sgemm(
                        seq_len,                         // m
                        head_dim,                        // k
                        seq_len,                         // n
                        scale,                           // alpha
                        sc.q.as_ptr().add(h * head_dim), // A ptr
                        hidden as isize,                 // rsa
                        1,                               // csa
                        sc.k.as_ptr().add(h * head_dim), // B ptr (K transposed via strides)
                        1,                               // rsb
                        hidden as isize,                 // csb
                        0.0f32,                          // beta
                        sc.scores.as_mut_ptr(),          // C ptr
                        seq_len as isize,                // rsc
                        1,                               // csc
                    );
                }

                // Apply attention mask
                for (j, &m) in attention_mask.iter().enumerate() {
                    if m == 0 {
                        for i in 0..seq_len {
                            sc.scores[i * seq_len + j] += -10000.0;
                        }
                    }
                }

                // Softmax over j for each i
                for i in 0..seq_len {
                    softmax(&mut sc.scores[i * seq_len..(i + 1) * seq_len]);
                }

                // attn_out_h = scores_h @ V_h
                // A = scores_h: [seq_len, seq_len], contiguous
                // B = V_h: logical [seq_len, head_dim], pointer = v[h*head_dim..], rsb=hidden, csb=1
                // C = attn_out_h: written into attn_out[h*head_dim..], rsc=hidden, csc=1
                unsafe {
                    matrixmultiply::sgemm(
                        seq_len,                                    // m
                        seq_len,                                    // k
                        head_dim,                                   // n
                        1.0f32,                                     // alpha
                        sc.scores.as_ptr(),                         // A
                        seq_len as isize,                           // rsa
                        1,                                          // csa
                        sc.v.as_ptr().add(h * head_dim),            // B = V_h
                        hidden as isize,                            // rsb
                        1,                                          // csb
                        0.0f32,                                     // beta
                        sc.attn_out.as_mut_ptr().add(h * head_dim), // C output for this head
                        hidden as isize,                            // rsc (interleaved)
                        1,                                          // csc
                    );
                }
            }

            // Output projection: [seq × hidden] → [seq × hidden]
            sc.attn_proj[..seq_hidden].fill(0.0);
            {
                // Use raw pointers to allow simultaneous reads from attn_out and
                // writes to attn_proj, which are distinct fields of sc.
                let src = sc.attn_out.as_ptr();
                let dst = sc.attn_proj.as_mut_ptr();
                linear_batch(
                    unsafe { std::slice::from_raw_parts(src, seq_hidden) },
                    seq_len,
                    hidden,
                    layer.attn_out_weight.as_f32(),
                    layer.attn_out_bias.as_f32(),
                    unsafe { std::slice::from_raw_parts_mut(dst, seq_hidden) },
                );
            }

            // Residual + LayerNorm
            for (xi, ai) in x.iter_mut().zip(sc.attn_proj[..seq_hidden].iter()) {
                *xi += ai;
            }
            layer_norm_rows(
                &mut x,
                seq_len,
                hidden,
                layer.attn_ln_weight.as_f32(),
                layer.attn_ln_bias.as_f32(),
            );

            // ── b. FFN ─────────────────────────────────────────────────────────

            // Intermediate projection: [seq × hidden] → [seq × intermediate]
            sc.inter[..seq_inter].fill(0.0);
            linear_batch(
                &x,
                seq_len,
                hidden,
                layer.inter_weight.as_f32(),
                layer.inter_bias.as_f32(),
                &mut sc.inter[..seq_inter],
            );

            // GELU activation
            gelu(&mut sc.inter[..seq_inter]);

            // Output projection: [seq × intermediate] → [seq × hidden]
            sc.ffn_out[..seq_hidden].fill(0.0);
            {
                // Use raw pointers to allow simultaneous reads from inter and
                // writes to ffn_out, which are distinct fields of sc.
                let src = sc.inter.as_ptr();
                let dst = sc.ffn_out.as_mut_ptr();
                linear_batch(
                    unsafe { std::slice::from_raw_parts(src, seq_inter) },
                    seq_len,
                    intermediate,
                    layer.out_weight.as_f32(),
                    layer.out_bias.as_f32(),
                    unsafe { std::slice::from_raw_parts_mut(dst, seq_hidden) },
                );
            }

            // Residual + LayerNorm
            for (xi, fi) in x.iter_mut().zip(sc.ffn_out[..seq_hidden].iter()) {
                *xi += fi;
            }
            layer_norm_rows(
                &mut x,
                seq_len,
                hidden,
                layer.out_ln_weight.as_f32(),
                layer.out_ln_bias.as_f32(),
            );
        }

        Ok(x)
    }

    /// Batched forward pass over a padded [batch_size * seq_len] token layout.
    pub fn forward_batch(
        &self,
        input_ids: &[u32],
        token_type_ids: &[u32],
        attention_mask: &[u32],
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Vec<f32>, LTEmbedError> {
        let total_tokens = batch_size
            .checked_mul(seq_len)
            .ok_or_else(|| LTEmbedError::Inference("Batch shape overflow".to_string()))?;
        if input_ids.len() != total_tokens
            || token_type_ids.len() != total_tokens
            || attention_mask.len() != total_tokens
        {
            return Err(LTEmbedError::Inference(
                "Batched inputs do not match batch_size * seq_len".to_string(),
            ));
        }
        if batch_size == 0 || seq_len == 0 {
            return Ok(Vec::new());
        }

        let hidden = self.config.hidden_size;
        let num_heads = self.config.num_attention_heads;
        let head_dim = hidden / num_heads;
        let intermediate = self.config.intermediate_size;
        let seq_hidden = seq_len * hidden;
        let total_hidden = total_tokens * hidden;
        let total_inter = total_tokens * intermediate;
        let seq_sq = seq_len * seq_len;

        let mut x = vec![0.0f32; total_hidden];
        embed(input_ids, self.word_emb.as_f32(), hidden, &mut x);

        let pos_emb = self.pos_emb.as_f32();
        let type_emb = self.type_emb.as_f32();
        for batch_idx in 0..batch_size {
            let batch_start = batch_idx * seq_hidden;
            let batch_end = batch_start + seq_hidden;
            for (pos, token_row) in x[batch_start..batch_end].chunks_mut(hidden).enumerate() {
                let pos_start = pos * hidden;
                let type_id = token_type_ids[batch_idx * seq_len + pos] as usize;
                let type_start = type_id * hidden;
                for (j, v) in token_row.iter_mut().enumerate() {
                    *v += pos_emb[pos_start + j];
                    *v += type_emb[type_start + j];
                }
            }
        }

        layer_norm_rows(
            &mut x,
            total_tokens,
            hidden,
            self.emb_ln_weight.as_f32(),
            self.emb_ln_bias.as_f32(),
        );

        let mut q = vec![0.0f32; total_hidden];
        let mut k = vec![0.0f32; total_hidden];
        let mut v = vec![0.0f32; total_hidden];
        let mut attn_out = vec![0.0f32; total_hidden];
        let mut attn_proj = vec![0.0f32; total_hidden];
        let mut scores = vec![0.0f32; seq_sq];
        let mut inter = vec![0.0f32; total_inter];
        let mut ffn_out = vec![0.0f32; total_hidden];

        for layer in &self.layers {
            q.fill(0.0);
            k.fill(0.0);
            v.fill(0.0);

            linear_batch(
                &x,
                total_tokens,
                hidden,
                layer.q_weight.as_f32(),
                layer.q_bias.as_f32(),
                &mut q,
            );
            linear_batch(
                &x,
                total_tokens,
                hidden,
                layer.k_weight.as_f32(),
                layer.k_bias.as_f32(),
                &mut k,
            );
            linear_batch(
                &x,
                total_tokens,
                hidden,
                layer.v_weight.as_f32(),
                layer.v_bias.as_f32(),
                &mut v,
            );

            let scale = 1.0 / (head_dim as f32).sqrt();
            attn_out.fill(0.0);

            for batch_idx in 0..batch_size {
                let token_offset = batch_idx * seq_len;
                let hidden_offset = batch_idx * seq_hidden;
                let batch_mask = &attention_mask[token_offset..token_offset + seq_len];

                for h in 0..num_heads {
                    scores.fill(0.0);

                    unsafe {
                        matrixmultiply::sgemm(
                            seq_len,
                            head_dim,
                            seq_len,
                            scale,
                            q.as_ptr().add(hidden_offset + h * head_dim),
                            hidden as isize,
                            1,
                            k.as_ptr().add(hidden_offset + h * head_dim),
                            1,
                            hidden as isize,
                            0.0f32,
                            scores.as_mut_ptr(),
                            seq_len as isize,
                            1,
                        );
                    }

                    for (j, &m) in batch_mask.iter().enumerate() {
                        if m == 0 {
                            for i in 0..seq_len {
                                scores[i * seq_len + j] += -10000.0;
                            }
                        }
                    }

                    for i in 0..seq_len {
                        softmax(&mut scores[i * seq_len..(i + 1) * seq_len]);
                    }

                    unsafe {
                        matrixmultiply::sgemm(
                            seq_len,
                            seq_len,
                            head_dim,
                            1.0f32,
                            scores.as_ptr(),
                            seq_len as isize,
                            1,
                            v.as_ptr().add(hidden_offset + h * head_dim),
                            hidden as isize,
                            1,
                            0.0f32,
                            attn_out.as_mut_ptr().add(hidden_offset + h * head_dim),
                            hidden as isize,
                            1,
                        );
                    }
                }
            }

            attn_proj.fill(0.0);
            linear_batch(
                &attn_out,
                total_tokens,
                hidden,
                layer.attn_out_weight.as_f32(),
                layer.attn_out_bias.as_f32(),
                &mut attn_proj,
            );

            for (xi, ai) in x.iter_mut().zip(attn_proj.iter()) {
                *xi += ai;
            }
            layer_norm_rows(
                &mut x,
                total_tokens,
                hidden,
                layer.attn_ln_weight.as_f32(),
                layer.attn_ln_bias.as_f32(),
            );

            inter.fill(0.0);
            linear_batch(
                &x,
                total_tokens,
                hidden,
                layer.inter_weight.as_f32(),
                layer.inter_bias.as_f32(),
                &mut inter,
            );
            gelu(&mut inter);

            ffn_out.fill(0.0);
            linear_batch(
                &inter,
                total_tokens,
                intermediate,
                layer.out_weight.as_f32(),
                layer.out_bias.as_f32(),
                &mut ffn_out,
            );

            for (xi, fi) in x.iter_mut().zip(ffn_out.iter()) {
                *xi += fi;
            }
            layer_norm_rows(
                &mut x,
                total_tokens,
                hidden,
                layer.out_ln_weight.as_f32(),
                layer.out_ln_bias.as_f32(),
            );
        }

        Ok(x)
    }
}

// ── Unit tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::{
        serialize_to_file,
        tensor::{Dtype, TensorView},
    };
    use std::collections::HashMap;
    use std::fs::File;
    use std::path::Path;
    use std::sync::Arc;
    use std::time::{SystemTime, UNIX_EPOCH};

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
        assert_eq!(output.len(), 3 * 384);
    }

    #[test]
    fn test_forward_batch_output_shape() {
        if !assets_available() {
            eprintln!("Skipping: model assets not found in assets/");
            return;
        }
        let config_str = std::fs::read_to_string(CONFIG_PATH).unwrap();
        let bert = Bert::from_files(SAFETENSORS_PATH, &config_str).unwrap();

        let input_ids = vec![101u32, 7592, 102, 101, 2088, 102];
        let token_type_ids = vec![0u32; 6];
        let attention_mask = vec![1u32; 6];

        let output = bert
            .forward_batch(&input_ids, &token_type_ids, &attention_mask, 2, 3)
            .unwrap();
        assert_eq!(output.len(), 2 * 3 * 384);
    }

    #[test]
    fn test_load_tensor_data_reads_from_mmap() {
        let values = [1.0f32, 2.0, 3.0, 4.0];
        let tensor =
            TensorView::new(Dtype::F32, vec![2, 2], bytemuck::cast_slice(&values)).unwrap();
        let tensors = HashMap::from([("foo".to_string(), tensor)]);
        let filename = format!(
            "ltembed-test-{}.safetensors",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let path = std::env::temp_dir().join(filename);

        serialize_to_file(&tensors, None, &path).unwrap();

        let file = File::open(&path).unwrap();
        let mmap = Arc::new(unsafe { memmap2::Mmap::map(&file).unwrap() });
        let st = SafeTensors::deserialize(&mmap).unwrap();

        let data = load_tensor_data(&st, &mmap, "foo").unwrap();
        assert_eq!(data.as_f32(), &values);

        std::fs::remove_file(path).unwrap();
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
