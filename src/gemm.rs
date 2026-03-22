pub(crate) fn dense_backend_name() -> &'static str {
    "matrixmultiply"
}

/// Compute out[batch × output_size] = x_rows[batch × input_size] × weight^T + bias,
/// where weight[output_size × input_size] is row-major (natural PyTorch/safetensors layout).
pub(crate) fn linear_batch_with_bias(
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

    matrixmultiply_linear_batch(x_rows, batch, input_size, weight, out);

    for row in 0..batch {
        let offset = row * output_size;
        for (col, &b) in bias.iter().enumerate() {
            out[offset + col] += b;
        }
    }
}

pub(crate) fn matrixmultiply_linear_batch(
    x_rows: &[f32],
    batch: usize,
    input_size: usize,
    weight: &[f32],
    out: &mut [f32],
) {
    let output_size = out.len() / batch;
    debug_assert_eq!(weight.len(), input_size * output_size);

    // C[batch×output] = A[batch×input] × weight^T[input×output]
    // weight[output×input] accessed as column-major B: rsb=1, csb=input_size
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
            1,                   // rsb (stride across rows of weight^T = down columns of weight)
            input_size as isize, // csb (stride across cols of weight^T = across rows of weight)
            0.0,
            out.as_mut_ptr(),
            output_size as isize, // rsc
            1,                    // csc
        );
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a deterministic f32 pattern for test inputs.
    fn patterned(len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| (i as f32 * 0.01 - (len as f32 * 0.005)).sin())
            .collect()
    }

    fn run_linear_batch(batch: usize, input: usize, output: usize) {
        let x = patterned(batch * input);
        let w = patterned(output * input); // [output × input]
        let bias = patterned(output);

        let mut out = vec![0.0f32; batch * output];
        linear_batch_with_bias(&x, batch, input, &w, &bias, &mut out);

        // Verify each output element against scalar reference
        for b in 0..batch {
            for o in 0..output {
                let expected: f32 = (0..input)
                    .map(|i| x[b * input + i] * w[o * input + i])
                    .sum::<f32>()
                    + bias[o];
                let actual = out[b * output + o];
                let tol = 1e-4 * expected.abs().max(1.0);
                assert!(
                    (actual - expected).abs() < tol,
                    "mismatch at b={b} o={o}: {actual} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn test_linear_batch_projection_shapes() {
        run_linear_batch(1, 384, 384); // single-row QKV projection
        run_linear_batch(7, 384, 384); // single/short seq_len
        run_linear_batch(4, 384, 1024); // FFN expansion (e5-small-v2)
        run_linear_batch(4, 1024, 384); // FFN contraction
        run_linear_batch(32, 384, 384); // single/medium seq_len
    }
}
