// src/utils.rs

/// Normalize `v` to unit length in-place (L2 norm = 1).
///
/// Uses a floor of `1e-12` on the norm to avoid division by zero; a zero
/// vector remains a zero vector after normalization.
pub fn l2_normalize_inplace(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let inv = 1.0 / norm.max(1e-12);
    for x in v.iter_mut() {
        *x *= inv;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_unit_vector_unchanged() {
        let mut v = vec![1.0_f32, 0.0, 0.0];
        l2_normalize_inplace(&mut v);
        assert_relative_eq!(v[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(v[1], 0.0, epsilon = 1e-6);
        assert_relative_eq!(v[2], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_output_has_unit_norm() {
        let mut v = vec![3.0_f32, 4.0]; // hypotenuse = 5
        l2_normalize_inplace(&mut v);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert_relative_eq!(norm, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_arbitrary_vector_values() {
        let mut v = vec![1.0_f32, 2.0, 3.0];
        l2_normalize_inplace(&mut v);
        let sqrt14 = 14.0_f32.sqrt();
        assert_relative_eq!(v[0], 1.0 / sqrt14, epsilon = 1e-6);
        assert_relative_eq!(v[1], 2.0 / sqrt14, epsilon = 1e-6);
        assert_relative_eq!(v[2], 3.0 / sqrt14, epsilon = 1e-6);
    }

    #[test]
    fn test_negative_values_produce_unit_norm() {
        let mut v = vec![-3.0_f32, 4.0];
        l2_normalize_inplace(&mut v);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert_relative_eq!(norm, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_zero_vector_remains_zero() {
        // Edge case: zero norm → must not divide by zero or produce NaN
        let mut v = vec![0.0_f32, 0.0, 0.0];
        l2_normalize_inplace(&mut v);
        assert_eq!(v, vec![0.0_f32, 0.0, 0.0]);
    }
}
