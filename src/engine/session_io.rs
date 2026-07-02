use ort::session::Session;

use crate::error::LTEmbedError;

#[derive(Debug)]
pub(crate) struct SessionIo {
    input_ids: String,
    attention_mask: String,
    last_hidden_state: String,
    sequence_length: Option<usize>,
}

impl SessionIo {
    /// Name of the model's `input_ids` input, as discovered from the session.
    pub(crate) fn input_ids_name(&self) -> &str {
        &self.input_ids
    }

    /// Name of the model's `attention_mask` input, as discovered from the session.
    pub(crate) fn attention_mask_name(&self) -> &str {
        &self.attention_mask
    }

    /// Name of the model's `last_hidden_state` output, as discovered from the session.
    pub(crate) fn output_name(&self) -> &str {
        &self.last_hidden_state
    }

    /// Fixed model sequence length, or `None` when the model accepts a dynamic length.
    pub(crate) fn sequence_length(&self) -> Option<usize> {
        self.sequence_length
    }

    pub(crate) fn from_session(
        session: &Session,
        raw_embedding_dimension: usize,
    ) -> Result<Self, LTEmbedError> {
        let input_ids = session
            .inputs()
            .iter()
            .find(|input| input.name() == "input_ids")
            .ok_or_else(|| {
                LTEmbedError::ModelLoad("ORT model is missing required input `input_ids`".into())
            })?;
        let attention_mask = session
            .inputs()
            .iter()
            .find(|input| input.name() == "attention_mask")
            .ok_or_else(|| {
                LTEmbedError::ModelLoad(
                    "ORT model is missing required input `attention_mask`".into(),
                )
            })?;
        let last_hidden_state = session
            .outputs()
            .iter()
            .find(|output| output.name() == "last_hidden_state")
            .ok_or_else(|| {
                LTEmbedError::ModelLoad(
                    "ORT model is missing required output `last_hidden_state`".into(),
                )
            })?;
        let raw_dim = last_hidden_state
            .dtype()
            .tensor_shape()
            .and_then(|shape| shape.last().copied());
        if raw_dim != Some(raw_embedding_dimension as i64) {
            return Err(LTEmbedError::ModelLoad(format!(
                "ORT model output `last_hidden_state` must expose raw embedding dimension {raw_embedding_dimension}, got {raw_dim:?}"
            )));
        }
        let input_ids_shape = input_ids.dtype().tensor_shape().ok_or_else(|| {
            LTEmbedError::ModelLoad("ORT model input `input_ids` must be a tensor".into())
        })?;
        let attention_mask_shape = attention_mask.dtype().tensor_shape().ok_or_else(|| {
            LTEmbedError::ModelLoad("ORT model input `attention_mask` must be a tensor".into())
        })?;
        let sequence_length =
            resolved_model_sequence_length(input_ids_shape, attention_mask_shape)?;

        Ok(Self {
            input_ids: input_ids.name().to_string(),
            attention_mask: attention_mask.name().to_string(),
            last_hidden_state: last_hidden_state.name().to_string(),
            sequence_length,
        })
    }
}

fn model_sequence_length(shape: &[i64]) -> Result<Option<usize>, LTEmbedError> {
    if shape.len() != 2 {
        return Err(LTEmbedError::ModelLoad(format!(
            "ORT model input must be rank-2, got shape {shape:?}"
        )));
    }

    match shape[1] {
        dim if dim < 0 => Ok(None),
        dim => usize::try_from(dim)
            .map(Some)
            .map_err(|_| LTEmbedError::ModelLoad(format!("Invalid ORT input shape {shape:?}"))),
    }
}

fn resolved_model_sequence_length(
    input_ids_shape: &[i64],
    attention_mask_shape: &[i64],
) -> Result<Option<usize>, LTEmbedError> {
    let input_sequence_length = model_sequence_length(input_ids_shape)?;
    let attention_mask_sequence_length = model_sequence_length(attention_mask_shape)?;

    match (input_sequence_length, attention_mask_sequence_length) {
        (Some(input_len), Some(mask_len)) if input_len != mask_len => Err(LTEmbedError::ModelLoad(
            format!(
                "ORT model inputs `input_ids` and `attention_mask` must expose compatible sequence lengths, got {input_ids_shape:?} and {attention_mask_shape:?}"
            ),
        )),
        (Some(input_len), Some(_)) => Ok(Some(input_len)),
        (Some(input_len), None) => Ok(Some(input_len)),
        (None, Some(mask_len)) => Ok(Some(mask_len)),
        (None, None) => Ok(None),
    }
}

pub(crate) fn effective_sequence_length(
    model_sequence_length: Option<usize>,
    batch_max_sequence_length: usize,
) -> Result<usize, LTEmbedError> {
    match model_sequence_length {
        Some(model_len) if batch_max_sequence_length > model_len => Err(LTEmbedError::Inference(
            format!(
                "encoded input length {batch_max_sequence_length} exceeds ORT model sequence length {model_len}"
            ),
        )),
        Some(model_len) => Ok(model_len),
        None => Ok(batch_max_sequence_length),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_sequence_length_is_dynamic_when_second_dim_is_negative() {
        assert_eq!(model_sequence_length(&[-1, -1]).unwrap(), None);
    }

    #[test]
    fn test_model_sequence_length_uses_fixed_second_dim() {
        assert_eq!(model_sequence_length(&[-1, 8192]).unwrap(), Some(8192));
    }

    #[test]
    fn test_effective_sequence_length_uses_batch_max_for_dynamic_models() {
        assert_eq!(effective_sequence_length(None, 7).unwrap(), 7);
    }

    #[test]
    fn test_effective_sequence_length_uses_fixed_model_length() {
        assert_eq!(effective_sequence_length(Some(8192), 7).unwrap(), 8192);
    }

    #[test]
    fn test_resolved_model_sequence_length_uses_fixed_input_ids_shape() {
        assert_eq!(
            resolved_model_sequence_length(&[-1, 8192], &[-1, -1]).unwrap(),
            Some(8192)
        );
    }

    #[test]
    fn test_resolved_model_sequence_length_uses_fixed_attention_mask_shape() {
        assert_eq!(
            resolved_model_sequence_length(&[-1, -1], &[-1, 8192]).unwrap(),
            Some(8192)
        );
    }

    #[test]
    fn test_resolved_model_sequence_length_rejects_mismatched_fixed_shapes() {
        let err = resolved_model_sequence_length(&[-1, 8192], &[-1, 4096]).unwrap_err();
        assert!(matches!(err, LTEmbedError::ModelLoad(_)));
    }
}
