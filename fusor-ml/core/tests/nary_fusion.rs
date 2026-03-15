use fusor_core::Device;
use fusor_core::Tensor;

#[tokio::test]
async fn test_nary_triple_add() {
    let device = Device::new().await.unwrap();
    let a = Tensor::new(&device, &[[1., 2.], [3., 4.]]);
    let b = Tensor::new(&device, &[[5., 6.], [7., 8.]]);
    let c = Tensor::new(&device, &[[9., 10.], [11., 12.]]);

    // x + y + z should be ONE kernel
    let result = &(&a + &b) + &c;
    assert_eq!(result.count_kernels_to_resolve(), 1);
    let output = result.as_slice().await.unwrap();

    assert_eq!(output[[0, 0]], 1. + 5. + 9.);
    assert_eq!(output[[1, 1]], 4. + 8. + 12.);
}

#[tokio::test]
async fn test_nary_mixed_ops() {
    let device = Device::new().await.unwrap();
    let a = Tensor::new(&device, &[[1., 2.], [3., 4.]]);
    let b = Tensor::new(&device, &[[5., 6.], [7., 8.]]);
    let c = Tensor::new(&device, &[[9., 10.], [11., 12.]]);

    // (a + b) * c should be ONE kernel
    let result = &(&a + &b) * &c;
    assert_eq!(result.count_kernels_to_resolve(), 1);
    let output = result.as_slice().await.unwrap();

    assert_eq!(output[[0, 0]], (1. + 5.) * 9.);
    assert_eq!(output[[1, 1]], (4. + 8.) * 12.);
}

#[tokio::test]
async fn test_nary_nested_pairwise() {
    let device = Device::new().await.unwrap();
    let a = Tensor::new(&device, &[[1., 2.], [3., 4.]]);
    let b = Tensor::new(&device, &[[5., 6.], [7., 8.]]);
    let c = Tensor::new(&device, &[[9., 10.], [11., 12.]]);
    let d = Tensor::new(&device, &[[1., 1.], [1., 1.]]);

    // (a + b) * (c - d) should be ONE kernel with 4 inputs
    let result = &(&a + &b) * &(&c - &d);
    assert_eq!(result.count_kernels_to_resolve(), 1);
    let output = result.as_slice().await.unwrap();

    assert_eq!(output[[0, 0]], (1. + 5.) * (9. - 1.));
    assert_eq!(output[[1, 1]], (4. + 8.) * (12. - 1.));
}

#[tokio::test]
async fn test_nary_unary_in_middle() {
    let device = Device::new().await.unwrap();
    let a = Tensor::new(&device, &[[1., 2.], [3., 4.]]);
    let b = Tensor::new(&device, &[[0.5, 0.5], [0.5, 0.5]]);

    // (-a + b.sin()).cos() + 1.0 should be ONE kernel
    let result = ((-a.clone()) + b.sin()).cos() + 1.0;
    assert_eq!(result.count_kernels_to_resolve(), 1);
    let output = result.as_slice().await.unwrap();

    let expected_00 = ((-1.0_f32) + 0.5_f32.sin()).cos() + 1.0;
    assert!((output[[0, 0]] - expected_00).abs() < 0.001);
}

#[tokio::test]
async fn test_nary_element_wise_chain_then_pairwise() {
    let device = Device::new().await.unwrap();
    let a = Tensor::new(&device, &[[1., 2.], [3., 4.]]);
    let b = Tensor::new(&device, &[[5., 6.], [7., 8.]]);

    // (a + 1.0).exp() + b.sin() should be ONE kernel
    let result = &(a + 1.0).exp() + &b.sin();
    assert_eq!(result.count_kernels_to_resolve(), 1);
    let output = result.as_slice().await.unwrap();

    let expected_00 = (1.0_f32 + 1.0).exp() + 5.0_f32.sin();
    assert!((output[[0, 0]] - expected_00).abs() < 0.001);
}

#[tokio::test]
async fn test_nary_same_input_multiple_times() {
    let device = Device::new().await.unwrap();
    let a = Tensor::new(&device, &[[1., 2.], [3., 4.]]);

    // a + a + a should deduplicate inputs
    let result = &(&a + &a) + &a;
    assert_eq!(result.count_kernels_to_resolve(), 1);
    let output = result.as_slice().await.unwrap();

    assert_eq!(output[[0, 0]], 3.0);
    assert_eq!(output[[1, 1]], 12.0);
}

#[tokio::test]
async fn test_nary_where_cond_basic() {
    let device = Device::new().await.unwrap();
    let condition = Tensor::new(&device, &[[0u32, 1], [1, 0]]);
    let on_true = Tensor::new(&device, &[[10., 20.], [30., 40.]]);
    let on_false = Tensor::new(&device, &[[1., 2.], [3., 4.]]);

    // Basic where_cond
    let result = condition.where_cond(&on_true, &on_false);
    assert_eq!(result.count_kernels_to_resolve(), 1);
    let output = result.as_slice().await.unwrap();

    assert_eq!(output[[0, 0]], 1.); // condition=0 -> on_false
    assert_eq!(output[[0, 1]], 20.); // condition=1 -> on_true
    assert_eq!(output[[1, 0]], 30.); // condition=1 -> on_true
    assert_eq!(output[[1, 1]], 4.); // condition=0 -> on_false
}

#[tokio::test]
async fn test_nary_fusion_respects_binding_limit() {
    let device = Device::new().await.unwrap();

    // Get the actual GPU storage buffer limit
    let max_storage_buffers = device.limits().max_storage_buffers_per_shader_stage as usize;

    // Create enough tensors to exceed the limit
    // We need max_storage_buffers + 1 unique inputs to exceed the limit
    // (since we also need 1 binding for output)
    let num_tensors = max_storage_buffers + 1;

    let tensors: Vec<_> = (0..num_tensors)
        .map(|i| {
            Tensor::new(
                &device,
                &[[i as f32, (i + 1) as f32], [(i + 2) as f32, (i + 3) as f32]],
            )
        })
        .collect();

    // Add all tensors together in a chain
    // This requires num_tensors input bindings + 1 output = num_tensors + 1 bindings
    // which exceeds the max_storage_buffers limit
    let result: Tensor<2, _> = tensors.iter().sum();

    // The number of kernels should be more than 1 due to the binding limit
    let kernel_count = result.count_kernels_to_resolve();
    assert!(
        kernel_count > 1,
        "Expected more than 1 kernel due to storage binding limit (max_storage_buffers={}), got {}",
        max_storage_buffers,
        kernel_count
    );

    // Verify the result is still correct
    let output = result.as_slice().await.unwrap();
    let expected_00: f32 = (0..num_tensors).map(|i| i as f32).sum();
    let expected_11: f32 = (0..num_tensors).map(|i| (i + 3) as f32).sum();
    assert_eq!(output[[0, 0]], expected_00);
    assert_eq!(output[[1, 1]], expected_11);
}

/// Regression test: when a single node feeds two consumers (sin + cos),
/// the GPU must produce correct results for both.
/// Variant 1: resolving sin and cos separately
#[tokio::test]
async fn test_dual_consumer_sin_cos() {
    let device = Device::new().await.unwrap();

    // Create a matmul to produce a non-trivial intermediate
    let a = Tensor::new(&device, &[[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]]);
    let b = Tensor::new(&device, &[[0.1f32, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]);
    let mm = a.mat_mul(&b);

    // scaled is consumed by BOTH sin and cos (dual consumer)
    let scaled = mm * (2.0 * std::f32::consts::PI);
    let sin_out = scaled.sin();
    let cos_out = scaled.cos();

    let sin_slice = sin_out.as_slice().await.unwrap();
    let cos_slice = cos_out.as_slice().await.unwrap();

    // Compute expected values on CPU
    let a_data = [[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let b_data = [[0.1f32, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]];
    for i in 0..3 {
        for j in 0..4 {
            let mm_val = a_data[i][0] * b_data[0][j] + a_data[i][1] * b_data[1][j];
            let scaled_val = mm_val * 2.0 * std::f32::consts::PI;
            let expected_sin = scaled_val.sin();
            let expected_cos = scaled_val.cos();
            assert!(
                (sin_slice[[i, j]] - expected_sin).abs() < 0.01,
                "sin mismatch at [{}, {}]: got {} expected {}",
                i,
                j,
                sin_slice[[i, j]],
                expected_sin,
            );
            assert!(
                (cos_slice[[i, j]] - expected_cos).abs() < 0.01,
                "cos mismatch at [{}, {}]: got {} expected {}",
                i,
                j,
                cos_slice[[i, j]],
                expected_cos,
            );
        }
    }
}

/// Regression test: when a single node feeds two consumers (sin + cos),
/// both resolved together via cat (single resolve call).
#[tokio::test]
async fn test_dual_consumer_sin_cos_cat() {
    let device = Device::new().await.unwrap();

    // Create a matmul to produce a non-trivial intermediate
    let a = Tensor::new(&device, &[[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]]);
    let b = Tensor::new(&device, &[[0.1f32, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]);
    let mm = a.mat_mul(&b);

    // scaled is consumed by BOTH sin and cos (dual consumer)
    let scaled = mm * (2.0 * std::f32::consts::PI);
    let sin_out = scaled.sin();
    let cos_out = scaled.cos();

    // Concatenate sin and cos - resolved together in one graph pass
    let result = Tensor::cat([sin_out, cos_out], 1);
    let result_slice = result.as_slice().await.unwrap();

    // Compute expected values on CPU
    let a_data = [[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let b_data = [[0.1f32, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]];
    for i in 0..3 {
        for j in 0..4 {
            let mm_val = a_data[i][0] * b_data[0][j] + a_data[i][1] * b_data[1][j];
            let scaled_val = mm_val * 2.0 * std::f32::consts::PI;
            let expected_sin = scaled_val.sin();
            let expected_cos = scaled_val.cos();
            // sin values are in columns 0..4, cos values in columns 4..8
            let sin_diff = (result_slice[[i, j]] - expected_sin).abs();
            let cos_diff = (result_slice[[i, j + 4]] - expected_cos).abs();
            assert!(
                sin_diff < 0.01,
                "sin mismatch at [{}, {}]: got {} expected {} diff {}",
                i,
                j,
                result_slice[[i, j]],
                expected_sin,
                sin_diff,
            );
            assert!(
                cos_diff < 0.01,
                "cos mismatch at [{}, {}]: got {} expected {} diff {}",
                i,
                j + 4,
                result_slice[[i, j + 4]],
                expected_cos,
                cos_diff,
            );
        }
    }
}
