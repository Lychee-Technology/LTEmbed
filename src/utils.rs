// src/utils.rs

/// Normalize vector `v` to unit length (L2 norm = 1).
/// Returns `v` unchanged if the norm is zero (all-zero input).
pub fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < f32::EPSILON {
        return v.to_vec();
    }
    v.iter().map(|x| x / norm).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_unit_vector_unchanged() {
        let v = vec![1.0_f32, 0.0, 0.0];
        let result = l2_normalize(&v);
        assert_relative_eq!(result[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(result[1], 0.0, epsilon = 1e-6);
        assert_relative_eq!(result[2], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_output_has_unit_norm() {
        let v = vec![3.0_f32, 4.0]; // hypotenuse = 5
        let result = l2_normalize(&v);
        let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert_relative_eq!(norm, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_arbitrary_vector_values() {
        let v = vec![1.0_f32, 2.0, 3.0];
        let result = l2_normalize(&v);
        let sqrt14 = 14.0_f32.sqrt();
        assert_relative_eq!(result[0], 1.0 / sqrt14, epsilon = 1e-6);
        assert_relative_eq!(result[1], 2.0 / sqrt14, epsilon = 1e-6);
        assert_relative_eq!(result[2], 3.0 / sqrt14, epsilon = 1e-6);
    }

    #[test]
    fn test_negative_values_produce_unit_norm() {
        let v = vec![-3.0_f32, 4.0];
        let result = l2_normalize(&v);
        let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert_relative_eq!(norm, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_zero_vector_returns_unchanged() {
        // Edge case: zero norm → must not divide by zero or produce NaN
        let v = vec![0.0_f32, 0.0, 0.0];
        let result = l2_normalize(&v);
        assert_eq!(result, vec![0.0_f32, 0.0, 0.0]);
    }
}
