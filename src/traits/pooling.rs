// src/traits/pooling.rs

use crate::error::LTEmbedError;

/// Collapses a [seq_len][hidden_size] last-hidden-state into a [hidden_size] vector.
/// `attention_mask`: 1 = real token, 0 = padding.
pub trait Pooling: Send + Sync {
    fn pool(
        &self,
        last_hidden_state: &[Vec<f32>],
        attention_mask: &[u32],
    ) -> Result<Vec<f32>, LTEmbedError>;
}

pub struct MeanPooling;
pub struct CLSPooling;

impl Pooling for MeanPooling {
    fn pool(
        &self,
        last_hidden_state: &[Vec<f32>],
        attention_mask: &[u32],
    ) -> Result<Vec<f32>, LTEmbedError> {
        if last_hidden_state.is_empty() {
            return Err(LTEmbedError::Inference("Empty hidden state".to_string()));
        }
        let hidden_size = last_hidden_state[0].len();
        let mut sum = vec![0.0_f32; hidden_size];
        let mut count = 0u32;

        for (token_vec, &mask_val) in last_hidden_state.iter().zip(attention_mask.iter()) {
            if mask_val == 1 {
                for (s, v) in sum.iter_mut().zip(token_vec.iter()) {
                    *s += v;
                }
                count += 1;
            }
        }

        if count == 0 {
            return Err(LTEmbedError::Inference("All tokens are padding".to_string()));
        }

        Ok(sum.iter().map(|x| x / count as f32).collect())
    }
}

impl Pooling for CLSPooling {
    fn pool(
        &self,
        last_hidden_state: &[Vec<f32>],
        _attention_mask: &[u32],
    ) -> Result<Vec<f32>, LTEmbedError> {
        last_hidden_state
            .first()
            .cloned()
            .ok_or_else(|| LTEmbedError::Inference("Empty hidden state for CLS pooling".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 3 tokens, hidden_size=2. Token 3 is padding.
    fn sample_hs() -> Vec<Vec<f32>> {
        vec![
            vec![1.0, 2.0], // [CLS]
            vec![3.0, 4.0], // real token
            vec![0.0, 0.0], // [PAD]
        ]
    }

    fn sample_mask() -> Vec<u32> {
        vec![1, 1, 0]
    }

    #[test]
    fn test_mean_pooling_ignores_padding() {
        let p = MeanPooling;
        let result = p.pool(&sample_hs(), &sample_mask()).unwrap();
        // Mean of tokens 0 and 1: ([1+3]/2, [2+4]/2) = (2.0, 3.0)
        assert_eq!(result, vec![2.0_f32, 3.0_f32]);
    }

    #[test]
    fn test_mean_pooling_all_tokens_active() {
        let p = MeanPooling;
        let hs = vec![vec![2.0_f32, 4.0], vec![4.0, 6.0]];
        let mask = vec![1u32, 1];
        let result = p.pool(&hs, &mask).unwrap();
        assert_eq!(result, vec![3.0_f32, 5.0_f32]);
    }

    #[test]
    fn test_mean_pooling_all_padding_returns_error() {
        let p = MeanPooling;
        let hs = vec![vec![1.0_f32, 2.0], vec![3.0, 4.0]];
        let mask = vec![0u32, 0]; // all padding
        let result = p.pool(&hs, &mask);
        assert!(result.is_err(), "Expected error when all tokens are padding");
    }

    #[test]
    fn test_mean_pooling_empty_hidden_state_returns_error() {
        let p = MeanPooling;
        let result = p.pool(&[], &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_cls_pooling_returns_first_token() {
        let p = CLSPooling;
        let result = p.pool(&sample_hs(), &sample_mask()).unwrap();
        assert_eq!(result, vec![1.0_f32, 2.0_f32]);
    }

    #[test]
    fn test_cls_pooling_empty_returns_error() {
        let p = CLSPooling;
        let result = p.pool(&[], &[]);
        assert!(result.is_err());
    }
}
