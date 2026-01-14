//! Tests for reduction operations: sum, max, min, prod (full tensor and axis-wise)

use fusor_cpu::{ConcreteTensor, ResolvedTensor};

// ========== Full Tensor Reduce Tests ==========

#[test]
fn test_sum_1d() {
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let tensor: ConcreteTensor<f32, 1> = ConcreteTensor::from_slice([5], &data);
    assert_eq!(tensor.sum(), 15.0);
}

#[test]
fn test_sum_2d() {
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor: ConcreteTensor<f32, 2> = ConcreteTensor::from_slice([2, 3], &data);
    assert_eq!(tensor.sum(), 21.0);
}

#[test]
fn test_sum_large() {
    let size = 10000;
    let data: Vec<f32> = (1..=size).map(|i| i as f32).collect();
    let tensor: ConcreteTensor<f32, 1> = ConcreteTensor::from_slice([size], &data);
    // Sum of 1..n = n*(n+1)/2
    let expected = (size * (size + 1) / 2) as f32;
    assert!((tensor.sum() - expected).abs() < 1.0);
}

#[test]
fn test_max_1d() {
    let data: Vec<f32> = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    let tensor: ConcreteTensor<f32, 1> = ConcreteTensor::from_slice([8], &data);
    assert_eq!(tensor.max(), 9.0);
}

#[test]
fn test_max_negative() {
    let data: Vec<f32> = vec![-3.0, -1.0, -4.0, -1.0, -5.0];
    let tensor: ConcreteTensor<f32, 1> = ConcreteTensor::from_slice([5], &data);
    assert_eq!(tensor.max(), -1.0);
}

#[test]
fn test_min_1d() {
    let data: Vec<f32> = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    let tensor: ConcreteTensor<f32, 1> = ConcreteTensor::from_slice([8], &data);
    assert_eq!(tensor.min(), 1.0);
}

#[test]
fn test_min_negative() {
    let data: Vec<f32> = vec![-3.0, -1.0, -4.0, -1.0, -5.0];
    let tensor: ConcreteTensor<f32, 1> = ConcreteTensor::from_slice([5], &data);
    assert_eq!(tensor.min(), -5.0);
}

#[test]
fn test_prod_1d() {
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let tensor: ConcreteTensor<f32, 1> = ConcreteTensor::from_slice([5], &data);
    assert_eq!(tensor.prod(), 120.0); // 5! = 120
}

#[test]
fn test_prod_2d() {
    let data: Vec<f32> = vec![2.0, 3.0, 4.0, 5.0];
    let tensor: ConcreteTensor<f32, 2> = ConcreteTensor::from_slice([2, 2], &data);
    assert_eq!(tensor.prod(), 120.0);
}

#[test]
fn test_reduce_i32() {
    let data: Vec<i32> = vec![1, 2, 3, 4, 5];
    let tensor: ConcreteTensor<i32, 1> = ConcreteTensor::from_slice([5], &data);
    assert_eq!(tensor.sum(), 15);
    assert_eq!(tensor.max(), 5);
    assert_eq!(tensor.min(), 1);
}

#[test]
fn test_reduce_f64() {
    let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
    let tensor: ConcreteTensor<f64, 1> = ConcreteTensor::from_slice([4], &data);
    assert_eq!(tensor.sum(), 10.0);
    assert_eq!(tensor.max(), 4.0);
    assert_eq!(tensor.min(), 1.0);
    assert_eq!(tensor.prod(), 24.0);
}

#[test]
fn test_reduce_single_element() {
    let tensor: ConcreteTensor<f32, 1> = ConcreteTensor::from_slice([1], &[42.0]);
    assert_eq!(tensor.sum(), 42.0);
    assert_eq!(tensor.max(), 42.0);
    assert_eq!(tensor.min(), 42.0);
    assert_eq!(tensor.prod(), 42.0);
}

// ========== Axis Reduce Tests ==========

#[test]
fn test_sum_axis_2d_axis0() {
    // [[1, 2, 3],
    //  [4, 5, 6]]
    // sum along axis 0 -> [5, 7, 9]
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor: ConcreteTensor<f32, 2> = ConcreteTensor::from_slice([2, 3], &data);

    let result: ConcreteTensor<f32, 1> = tensor.sum_axis::<1, 0>();

    assert_eq!(result.shape(), &[3]);
    assert_eq!(result.get([0]), 5.0);
    assert_eq!(result.get([1]), 7.0);
    assert_eq!(result.get([2]), 9.0);
}

#[test]
fn test_sum_axis_2d_axis1() {
    // [[1, 2, 3],
    //  [4, 5, 6]]
    // sum along axis 1 -> [6, 15]
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor: ConcreteTensor<f32, 2> = ConcreteTensor::from_slice([2, 3], &data);

    let result: ConcreteTensor<f32, 1> = tensor.sum_axis::<1, 1>();

    assert_eq!(result.shape(), &[2]);
    assert_eq!(result.get([0]), 6.0);
    assert_eq!(result.get([1]), 15.0);
}

#[test]
fn test_max_axis_2d() {
    // [[1, 5, 3],
    //  [4, 2, 6]]
    // max along axis 0 -> [4, 5, 6]
    let data: Vec<f32> = vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
    let tensor: ConcreteTensor<f32, 2> = ConcreteTensor::from_slice([2, 3], &data);

    let result: ConcreteTensor<f32, 1> = tensor.max_axis::<1, 0>();

    assert_eq!(result.shape(), &[3]);
    assert_eq!(result.get([0]), 4.0);
    assert_eq!(result.get([1]), 5.0);
    assert_eq!(result.get([2]), 6.0);
}

#[test]
fn test_min_axis_2d() {
    // [[1, 5, 3],
    //  [4, 2, 6]]
    // min along axis 1 -> [1, 2]
    let data: Vec<f32> = vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
    let tensor: ConcreteTensor<f32, 2> = ConcreteTensor::from_slice([2, 3], &data);

    let result: ConcreteTensor<f32, 1> = tensor.min_axis::<1, 1>();

    assert_eq!(result.shape(), &[2]);
    assert_eq!(result.get([0]), 1.0);
    assert_eq!(result.get([1]), 2.0);
}

#[test]
fn test_prod_axis_2d() {
    // [[1, 2],
    //  [3, 4]]
    // prod along axis 0 -> [3, 8]
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let tensor: ConcreteTensor<f32, 2> = ConcreteTensor::from_slice([2, 2], &data);

    let result: ConcreteTensor<f32, 1> = tensor.prod_axis::<1, 0>();

    assert_eq!(result.shape(), &[2]);
    assert_eq!(result.get([0]), 3.0);
    assert_eq!(result.get([1]), 8.0);
}

#[test]
fn test_sum_axis_3d() {
    // 2x2x2 tensor
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let tensor: ConcreteTensor<f32, 3> = ConcreteTensor::from_slice([2, 2, 2], &data);

    // Sum along axis 0 -> 2x2 tensor
    let result: ConcreteTensor<f32, 2> = tensor.sum_axis::<2, 0>();

    assert_eq!(result.shape(), &[2, 2]);
    // [1,2;3,4] + [5,6;7,8] = [6,8;10,12]
    assert_eq!(result.get([0, 0]), 6.0);
    assert_eq!(result.get([0, 1]), 8.0);
    assert_eq!(result.get([1, 0]), 10.0);
    assert_eq!(result.get([1, 1]), 12.0);
}
