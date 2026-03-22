pub(crate) fn dense_backend_name() -> &'static str {
    #[cfg(target_arch = "aarch64")]
    return "matrixmultiply+neon-dot";
    #[cfg(not(target_arch = "aarch64"))]
    return "matrixmultiply+dot";
}

/// Threshold (inclusive) for the GEMV path. For batch <= this value,
/// `small_batch_gemm_dot` is used instead of `matrixmultiply::sgemm`,
/// eliminating packing overhead that dominates for small m (e.g. seq_len ~8–16).
const GEMV_THRESHOLD: usize = 16;

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

    if batch <= GEMV_THRESHOLD {
        small_batch_gemm_dot(x_rows, batch, input_size, weight, output_size, out);
    } else {
        matrixmultiply_linear_batch(x_rows, batch, input_size, weight, out);
    }

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

/// DOT-PRODUCT GEMV for small batch (m <= GEMV_THRESHOLD).
///
/// For each batch row computes out[o] = dot(x, weight[o*input_size..]).
/// Uses NEON on aarch64 (4-row unrolling, 2 accumulators/row to hide FMA latency);
/// falls back to scalar on other targets.
fn small_batch_gemm_dot(
    x_rows: &[f32],
    batch: usize,
    input_size: usize,
    weight: &[f32],
    output_size: usize,
    out: &mut [f32],
) {
    for b in 0..batch {
        let x = &x_rows[b * input_size..(b + 1) * input_size];
        let o = &mut out[b * output_size..(b + 1) * output_size];
        gemv(x, weight, output_size, o);
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn gemv(x: &[f32], weight: &[f32], output_size: usize, out: &mut [f32]) {
    unsafe { gemv_neon(x, weight, output_size, out) }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn gemv(x: &[f32], weight: &[f32], output_size: usize, out: &mut [f32]) {
    gemv_scalar(x, weight, output_size, 0, out);
}

/// NEON DOT-PRODUCT: processes 4 output rows at a time, 2 accumulators per row
/// to hide the 4-cycle `fmla` latency on Cortex-A57/A72/A55/A78.
#[cfg(target_arch = "aarch64")]
unsafe fn gemv_neon(x: &[f32], weight: &[f32], output_size: usize, out: &mut [f32]) {
    use std::arch::aarch64::*;

    let input_size = x.len();
    let mut o = 0usize;

    while o + 4 <= output_size {
        let a0 = weight.as_ptr().add(o * input_size);
        let a1 = weight.as_ptr().add((o + 1) * input_size);
        let a2 = weight.as_ptr().add((o + 2) * input_size);
        let a3 = weight.as_ptr().add((o + 3) * input_size);

        let mut acc0a = vdupq_n_f32(0.0);
        let mut acc0b = vdupq_n_f32(0.0);
        let mut acc1a = vdupq_n_f32(0.0);
        let mut acc1b = vdupq_n_f32(0.0);
        let mut acc2a = vdupq_n_f32(0.0);
        let mut acc2b = vdupq_n_f32(0.0);
        let mut acc3a = vdupq_n_f32(0.0);
        let mut acc3b = vdupq_n_f32(0.0);

        let mut j = 0usize;
        // Main 8-element inner loop: load x once, multiply into 4 rows × 2 accumulators
        while j + 8 <= input_size {
            let xp = x.as_ptr().add(j);
            let x0 = vld1q_f32(xp);
            let x1 = vld1q_f32(xp.add(4));

            acc0a = vmlaq_f32(acc0a, vld1q_f32(a0.add(j)), x0);
            acc0b = vmlaq_f32(acc0b, vld1q_f32(a0.add(j + 4)), x1);
            acc1a = vmlaq_f32(acc1a, vld1q_f32(a1.add(j)), x0);
            acc1b = vmlaq_f32(acc1b, vld1q_f32(a1.add(j + 4)), x1);
            acc2a = vmlaq_f32(acc2a, vld1q_f32(a2.add(j)), x0);
            acc2b = vmlaq_f32(acc2b, vld1q_f32(a2.add(j + 4)), x1);
            acc3a = vmlaq_f32(acc3a, vld1q_f32(a3.add(j)), x0);
            acc3b = vmlaq_f32(acc3b, vld1q_f32(a3.add(j + 4)), x1);
            j += 8;
        }

        // Handle remaining 4-element chunk
        if j + 4 <= input_size {
            let x0 = vld1q_f32(x.as_ptr().add(j));
            acc0a = vmlaq_f32(acc0a, vld1q_f32(a0.add(j)), x0);
            acc1a = vmlaq_f32(acc1a, vld1q_f32(a1.add(j)), x0);
            acc2a = vmlaq_f32(acc2a, vld1q_f32(a2.add(j)), x0);
            acc3a = vmlaq_f32(acc3a, vld1q_f32(a3.add(j)), x0);
            j += 4;
        }

        // Reduce accumulators
        let mut sum0 = vaddvq_f32(vaddq_f32(acc0a, acc0b));
        let mut sum1 = vaddvq_f32(vaddq_f32(acc1a, acc1b));
        let mut sum2 = vaddvq_f32(vaddq_f32(acc2a, acc2b));
        let mut sum3 = vaddvq_f32(vaddq_f32(acc3a, acc3b));

        // Scalar tail for remaining input elements
        while j < input_size {
            let xj = *x.get_unchecked(j);
            sum0 += *a0.add(j) * xj;
            sum1 += *a1.add(j) * xj;
            sum2 += *a2.add(j) * xj;
            sum3 += *a3.add(j) * xj;
            j += 1;
        }

        *out.get_unchecked_mut(o) = sum0;
        *out.get_unchecked_mut(o + 1) = sum1;
        *out.get_unchecked_mut(o + 2) = sum2;
        *out.get_unchecked_mut(o + 3) = sum3;
        o += 4;
    }

    // Scalar tail for remaining output rows
    gemv_scalar(x, weight, output_size, o, out);
}

fn gemv_scalar(x: &[f32], weight: &[f32], output_size: usize, start: usize, out: &mut [f32]) {
    let input_size = x.len();
    for o in start..output_size {
        let row = &weight[o * input_size..(o + 1) * input_size];
        out[o] = x.iter().zip(row.iter()).map(|(a, b)| a * b).sum();
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
        let w = patterned(output * input); // [output × input]
        let bias = patterned(output);

        let mut out_dot = vec![0.0f32; batch * output];
        let mut out_sgemm = vec![0.0f32; batch * output];

        // DOT-PRODUCT path
        small_batch_gemm_dot(&x, batch, input, &w, output, &mut out_dot);
        for row in 0..batch {
            let off = row * output;
            for (j, &b) in bias.iter().enumerate() {
                out_dot[off + j] += b;
            }
        }

        // matrixmultiply path
        matrixmultiply_linear_batch(&x, batch, input, &w, &mut out_sgemm);
        for row in 0..batch {
            let off = row * output;
            for (j, &b) in bias.iter().enumerate() {
                out_sgemm[off + j] += b;
            }
        }

        for (a, b) in out_dot.iter().zip(out_sgemm.iter()) {
            // Allow relative error of 1e-4: for large accumulations (e.g. input=1536,
            // sum ~700) the dot-product and sgemm paths accumulate in different orders.
            let tol = 1e-4 * b.abs().max(1.0);
            assert!(
                (a - b).abs() < tol,
                "dot vs sgemm mismatch at batch={batch} input={input} output={output}: {a} vs {b} (tol={tol})"
            );
        }
    }

    #[test]
    fn test_dot_matches_matrixmultiply_projection_shapes() {
        run_both(1, 384, 384); // single-row QKV projection
        run_both(4, 384, 384); // batch=4 QKV
        run_both(4, 384, 1536); // FFN expansion
        run_both(4, 1536, 384); // FFN contraction
        run_both(16, 384, 384); // at threshold boundary
    }
}
