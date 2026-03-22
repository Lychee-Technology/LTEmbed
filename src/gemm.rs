use wide::f32x4;

pub(crate) fn dense_backend_name() -> &'static str {
    "matrixmultiply+saxpy"
}

/// Threshold (inclusive) for the SAXPY path. For batch <= this value,
/// `small_batch_gemm_saxpy` is used instead of `matrixmultiply::sgemm`,
/// eliminating packing overhead that dominates for small m (e.g. seq_len ~8–16).
const GEMV_THRESHOLD: usize = 16;

pub(crate) fn linear_batch_transposed_with_bias(
    x_rows: &[f32],
    batch: usize,
    input_size: usize,
    weight_t: &[f32],
    bias: &[f32],
    out: &mut [f32],
) {
    let output_size = bias.len();
    debug_assert_eq!(x_rows.len(), batch * input_size);
    debug_assert_eq!(out.len(), batch * output_size);
    debug_assert_eq!(weight_t.len(), input_size * output_size);

    if batch <= GEMV_THRESHOLD {
        small_batch_gemm_saxpy(x_rows, batch, input_size, weight_t, out);
    } else {
        matrixmultiply_linear_batch_transposed(x_rows, batch, input_size, weight_t, out);
    }

    for row in 0..batch {
        let offset = row * output_size;
        for (col, bias_value) in bias.iter().enumerate() {
            out[offset + col] += bias_value;
        }
    }
}

pub(crate) fn matrixmultiply_linear_batch_transposed(
    x_rows: &[f32],
    batch: usize,
    input_size: usize,
    weight_t: &[f32],
    out: &mut [f32],
) {
    let output_size = out.len() / batch;
    debug_assert_eq!(weight_t.len(), input_size * output_size);

    unsafe {
        matrixmultiply::sgemm(
            batch,
            input_size,
            output_size,
            1.0,
            x_rows.as_ptr(),
            input_size as isize,
            1,
            weight_t.as_ptr(),
            output_size as isize,
            1,
            0.0,
            out.as_mut_ptr(),
            output_size as isize,
            1,
        );
    }
}

/// SAXPY-based GEMM for small batch (m <= GEMV_THRESHOLD).
///
/// For each batch row, iterates over input dimensions and accumulates
/// `x[i] * weight_t_row_i` into the output using f32x4 SIMD. The
/// weight_t rows are contiguous in memory, making this access pattern
/// cache-friendly.
///
/// `out` is zeroed before accumulation; caller applies bias afterwards.
fn small_batch_gemm_saxpy(
    x_rows: &[f32],
    batch: usize,
    input_size: usize,
    weight_t: &[f32],
    out: &mut [f32],
) {
    let output_size = out.len() / batch;
    for b in 0..batch {
        let x = &x_rows[b * input_size..(b + 1) * input_size];
        let o = &mut out[b * output_size..(b + 1) * output_size];
        o.fill(0.0);
        for (i, &xi) in x.iter().enumerate() {
            saxpy_row(xi, &weight_t[i * output_size..(i + 1) * output_size], o);
        }
    }
}

/// `out += xi * weight_row` using f32x4 SIMD.
#[inline]
fn saxpy_row(xi: f32, weight_row: &[f32], out: &mut [f32]) {
    let output_size = out.len();
    debug_assert_eq!(weight_row.len(), output_size);

    let xi_v = f32x4::splat(xi);
    let chunks = output_size / 4;

    for c in 0..chunks {
        let base = c * 4;
        let w: [f32; 4] = weight_row[base..base + 4].try_into().unwrap();
        let o: [f32; 4] = out[base..base + 4].try_into().unwrap();
        let result: [f32; 4] = (f32x4::from(o) + xi_v * f32x4::from(w)).into();
        out[base..base + 4].copy_from_slice(&result);
    }

    for j in chunks * 4..output_size {
        out[j] += xi * weight_row[j];
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

    fn run_both(batch: usize, input: usize, output: usize) {
        let x = patterned(batch * input);
        let wt = patterned(input * output);
        let bias = patterned(output);

        let mut out_saxpy = vec![0.0f32; batch * output];
        let mut out_sgemm = vec![0.0f32; batch * output];

        // SAXPY path
        small_batch_gemm_saxpy(&x, batch, input, &wt, &mut out_saxpy);
        for row in 0..batch {
            let off = row * output;
            for (j, &b) in bias.iter().enumerate() {
                out_saxpy[off + j] += b;
            }
        }

        // matrixmultiply path
        matrixmultiply_linear_batch_transposed(&x, batch, input, &wt, &mut out_sgemm);
        for row in 0..batch {
            let off = row * output;
            for (j, &b) in bias.iter().enumerate() {
                out_sgemm[off + j] += b;
            }
        }

        for (a, b) in out_saxpy.iter().zip(out_sgemm.iter()) {
            assert!(
                (a - b).abs() < 1e-3,
                "saxpy vs sgemm mismatch at batch={batch} input={input} output={output}: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_saxpy_matches_matrixmultiply_projection_shapes() {
        run_both(1, 384, 384); // single-row QKV projection
        run_both(4, 384, 384); // batch=4 QKV
        run_both(4, 384, 1536); // FFN expansion
        run_both(4, 1536, 384); // FFN contraction
        run_both(16, 384, 384); // at threshold boundary
    }
}
