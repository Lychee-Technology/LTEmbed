#[cfg(all(
    feature = "vendored-blas",
    target_arch = "aarch64",
    target_os = "linux"
))]
use openblas_src as _;

#[cfg(all(
    feature = "vendored-blas",
    target_arch = "aarch64",
    target_os = "linux"
))]
const ACTIVE_BACKEND_NAME: &str = "openblas-cblas";
#[cfg(not(all(
    feature = "vendored-blas",
    target_arch = "aarch64",
    target_os = "linux"
)))]
const ACTIVE_BACKEND_NAME: &str = "matrixmultiply";

pub(crate) fn dense_backend_name() -> &'static str {
    ACTIVE_BACKEND_NAME
}

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

    backend_linear_batch_transposed(x_rows, batch, input_size, weight_t, out);

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

#[cfg(all(
    feature = "vendored-blas",
    target_arch = "aarch64",
    target_os = "linux"
))]
fn backend_linear_batch_transposed(
    x_rows: &[f32],
    batch: usize,
    input_size: usize,
    weight_t: &[f32],
    out: &mut [f32],
) {
    let output_size = out.len() / batch;
    unsafe {
        cblas::sgemm(
            cblas::Layout::RowMajor,
            cblas::Transpose::None,
            cblas::Transpose::None,
            batch as i32,
            output_size as i32,
            input_size as i32,
            1.0,
            x_rows,
            input_size as i32,
            weight_t,
            output_size as i32,
            0.0,
            out,
            output_size as i32,
        );
    }
}

#[cfg(not(all(
    feature = "vendored-blas",
    target_arch = "aarch64",
    target_os = "linux"
)))]
fn backend_linear_batch_transposed(
    x_rows: &[f32],
    batch: usize,
    input_size: usize,
    weight_t: &[f32],
    out: &mut [f32],
) {
    matrixmultiply_linear_batch_transposed(x_rows, batch, input_size, weight_t, out);
}
