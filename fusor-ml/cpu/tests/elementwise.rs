//! Tests for elementwise (unary) tensor operations: Neg, Abs, Sqrt

use fusor_cpu::{Abs, ConcreteTensor, Neg, ResolveTensor, Sqrt};

// ========== Neg Tests ==========

#[test]
fn test_neg_tensor_f32() {
    let data: Vec<f32> = vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0];
    let tensor: ConcreteTensor<f32, 2> = ConcreteTensor::from_slice([2, 3], &data);

    let neg = Neg::new(tensor);
    let result = neg.to_concrete();

    assert_eq!(result.get([0, 0]), -1.0);
    assert_eq!(result.get([0, 1]), 2.0);
    assert_eq!(result.get([0, 2]), -3.0);
    assert_eq!(result.get([1, 0]), 4.0);
    assert_eq!(result.get([1, 1]), -5.0);
    assert_eq!(result.get([1, 2]), 6.0);
}

#[test]
fn test_neg_large_tensor() {
    let size = 1024;
    let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let tensor: ConcreteTensor<f32, 1> = ConcreteTensor::from_slice([size], &data);

    let neg = Neg::new(tensor);
    let result = neg.to_concrete();

    for i in 0..size {
        assert_eq!(result.get([i]), -(i as f32));
    }
}

// ========== Abs Tests ==========

#[test]
fn test_abs_tensor_f32() {
    let data: Vec<f32> = vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0];
    let tensor: ConcreteTensor<f32, 2> = ConcreteTensor::from_slice([2, 3], &data);

    let abs = Abs::new(tensor);
    let result = abs.to_concrete();

    assert_eq!(result.get([0, 0]), 1.0);
    assert_eq!(result.get([0, 1]), 2.0);
    assert_eq!(result.get([0, 2]), 3.0);
    assert_eq!(result.get([1, 0]), 4.0);
    assert_eq!(result.get([1, 1]), 5.0);
    assert_eq!(result.get([1, 2]), 6.0);
}

#[test]
fn test_abs_i32() {
    let data: Vec<i32> = vec![1, -2, 3, -4];
    let tensor: ConcreteTensor<i32, 1> = ConcreteTensor::from_slice([4], &data);

    let abs = Abs::new(tensor);
    let result = abs.to_concrete();

    assert_eq!(result.get([0]), 1);
    assert_eq!(result.get([1]), 2);
    assert_eq!(result.get([2]), 3);
    assert_eq!(result.get([3]), 4);
}

// ========== Sqrt Tests ==========

#[test]
fn test_sqrt_tensor_f32() {
    let data: Vec<f32> = vec![1.0, 4.0, 9.0, 16.0, 25.0, 36.0];
    let tensor: ConcreteTensor<f32, 2> = ConcreteTensor::from_slice([2, 3], &data);

    let sqrt = Sqrt::new(tensor);
    let result = sqrt.to_concrete();

    assert_eq!(result.get([0, 0]), 1.0);
    assert_eq!(result.get([0, 1]), 2.0);
    assert_eq!(result.get([0, 2]), 3.0);
    assert_eq!(result.get([1, 0]), 4.0);
    assert_eq!(result.get([1, 1]), 5.0);
    assert_eq!(result.get([1, 2]), 6.0);
}

#[test]
fn test_sqrt_large_tensor() {
    let size = 1024;
    let data: Vec<f32> = (0..size).map(|i| (i * i) as f32).collect();
    let tensor: ConcreteTensor<f32, 1> = ConcreteTensor::from_slice([size], &data);

    let sqrt = Sqrt::new(tensor);
    let result = sqrt.to_concrete();

    for i in 0..size {
        assert!((result.get([i]) - i as f32).abs() < 1e-5);
    }
}

#[test]
fn test_sqrt_f64() {
    let data: Vec<f64> = vec![1.0, 4.0, 9.0, 16.0];
    let tensor: ConcreteTensor<f64, 1> = ConcreteTensor::from_slice([4], &data);

    let sqrt = Sqrt::new(tensor);
    let result = sqrt.to_concrete();

    assert_eq!(result.get([0]), 1.0);
    assert_eq!(result.get([1]), 2.0);
    assert_eq!(result.get([2]), 3.0);
    assert_eq!(result.get([3]), 4.0);
}
