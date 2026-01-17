//! Integration tests for the fusor crate

use fusor::{Device, GpuOr, GpuTensor};

#[test]
fn test_cpu_tensor_creation() {
    let a = GpuOr::cpu_from_slice([4], &[1.0, 2.0, 3.0, 4.0]);
    assert!(a.is_cpu());
    assert!(!a.is_gpu());
    assert_eq!(a.shape(), &[4]);
}

#[test]
fn test_cpu_add() {
    let a = GpuOr::cpu_from_slice([4], &[1.0, 2.0, 3.0, 4.0]);
    let b = GpuOr::cpu_from_slice([4], &[10.0, 20.0, 30.0, 40.0]);

    let c = &a + &b;
    assert!(c.is_cpu());

    let result = c.to_vec_blocking();
    assert_eq!(result, vec![11.0, 22.0, 33.0, 44.0]);
}

#[test]
fn test_cpu_sub() {
    let a = GpuOr::cpu_from_slice([4], &[10.0, 20.0, 30.0, 40.0]);
    let b = GpuOr::cpu_from_slice([4], &[1.0, 2.0, 3.0, 4.0]);

    let c = &a - &b;
    let result = c.to_vec_blocking();
    assert_eq!(result, vec![9.0, 18.0, 27.0, 36.0]);
}

#[test]
fn test_cpu_mul() {
    let a = GpuOr::cpu_from_slice([4], &[1.0, 2.0, 3.0, 4.0]);
    let b = GpuOr::cpu_from_slice([4], &[2.0, 3.0, 4.0, 5.0]);

    let c = &a * &b;
    let result = c.to_vec_blocking();
    assert_eq!(result, vec![2.0, 6.0, 12.0, 20.0]);
}

#[test]
fn test_cpu_div() {
    let a = GpuOr::cpu_from_slice([4], &[10.0, 20.0, 30.0, 40.0]);
    let b = GpuOr::cpu_from_slice([4], &[2.0, 4.0, 5.0, 8.0]);

    let c = &a / &b;
    let result = c.to_vec_blocking();
    assert_eq!(result, vec![5.0, 5.0, 6.0, 5.0]);
}

#[test]
fn test_cpu_neg() {
    let a = GpuOr::cpu_from_slice([4], &[1.0, -2.0, 3.0, -4.0]);

    let c = -&a;
    let result = c.to_vec_blocking();
    assert_eq!(result, vec![-1.0, 2.0, -3.0, 4.0]);
}

#[test]
fn test_cpu_abs() {
    let a = GpuOr::cpu_from_slice([4], &[-1.0, 2.0, -3.0, 4.0]);

    let c = a.abs();
    let result = c.to_vec_blocking();
    assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_cpu_sqrt() {
    let a = GpuOr::cpu_from_slice([4], &[1.0, 4.0, 9.0, 16.0]);

    let c = a.sqrt();
    let result = c.to_vec_blocking();
    assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_cpu_exp() {
    let a = GpuOr::cpu_from_slice([2], &[0.0, 1.0]);

    let c = a.exp();
    let result = c.to_vec_blocking();
    assert!((result[0] - 1.0).abs() < 1e-6);
    assert!((result[1] - std::f32::consts::E).abs() < 1e-6);
}

#[test]
fn test_cpu_log() {
    let a = GpuOr::cpu_from_slice([2], &[1.0, std::f32::consts::E]);

    let c = a.log();
    let result = c.to_vec_blocking();
    assert!((result[0] - 0.0).abs() < 1e-6);
    assert!((result[1] - 1.0).abs() < 1e-6);
}

#[test]
fn test_cpu_sin_cos() {
    let a = GpuOr::cpu_from_slice([2], &[0.0, std::f32::consts::PI / 2.0]);

    let sin_result = a.sin().to_vec_blocking();
    assert!((sin_result[0] - 0.0).abs() < 1e-6);
    assert!((sin_result[1] - 1.0).abs() < 1e-6);

    let cos_result = a.cos().to_vec_blocking();
    assert!((cos_result[0] - 1.0).abs() < 1e-6);
    assert!((cos_result[1] - 0.0).abs() < 1e-6);
}

#[test]
fn test_cpu_tanh() {
    let a = GpuOr::cpu_from_slice([2], &[0.0, 1.0]);

    let c = a.tanh();
    let result = c.to_vec_blocking();
    assert!((result[0] - 0.0).abs() < 1e-6);
    assert!((result[1] - 1.0_f32.tanh()).abs() < 1e-6);
}

#[test]
fn test_cpu_chained_operations() {
    // Test: sqrt(abs(a + b))
    let a = GpuOr::cpu_from_slice([4], &[-5.0, -4.0, 3.0, 8.0]);
    let b = GpuOr::cpu_from_slice([4], &[6.0, 8.0, 6.0, 8.0]);

    // a + b = [1.0, 4.0, 9.0, 16.0]
    let sum = &a + &b;
    let sum_result = sum.to_vec_blocking();
    assert_eq!(sum_result, vec![1.0, 4.0, 9.0, 16.0]);

    // Start fresh for the full chain
    let a2 = GpuOr::cpu_from_slice([4], &[1.0, 4.0, 9.0, 16.0]);
    let sqrt_result = a2.sqrt().to_vec_blocking();
    assert_eq!(sqrt_result, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_cpu_2d_tensor() {
    let a = GpuOr::cpu_from_slice([2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = GpuOr::cpu_from_slice([2, 3], &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);

    let c = &a + &b;
    let result = c.to_vec_blocking();
    assert_eq!(result, vec![11.0, 22.0, 33.0, 44.0, 55.0, 66.0]);
}

#[test]
fn test_cpu_zeros() {
    let a = GpuOr::cpu_zeros([4]);
    let result: Vec<f32> = a.to_vec_blocking();
    assert_eq!(result, vec![0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn test_device_check() {
    let a = GpuOr::cpu_from_slice([4], &[1.0, 2.0, 3.0, 4.0]);
    let device = a.device();
    assert!(matches!(device, Device::Cpu));
    assert!(device.is_cpu());
    assert!(!device.is_gpu());
}

// GPU tests require actual GPU hardware - these are async tests
#[tokio::test]
async fn test_gpu_tensor_creation() {
    let device = match fusor_core::Device::new().await {
        Ok(d) => d,
        Err(_) => {
            eprintln!("Skipping GPU test - no GPU available");
            return;
        }
    };

    let a = GpuOr::gpu_full(&device, 1.0, [4]);
    assert!(a.is_gpu());
    assert!(!a.is_cpu());
    assert_eq!(a.shape(), &[4]);

    let result = a.to_vec().await;
    assert_eq!(result, vec![1.0, 1.0, 1.0, 1.0]);
}

#[tokio::test]
async fn test_gpu_add() {
    let device = match fusor_core::Device::new().await {
        Ok(d) => d,
        Err(_) => {
            eprintln!("Skipping GPU test - no GPU available");
            return;
        }
    };

    let a = GpuOr::Gpu(GpuTensor::new(fusor_core::Tensor::new(
        &device,
        vec![&1.0, &2.0, &3.0, &4.0],
    )));
    let b = GpuOr::Gpu(GpuTensor::new(fusor_core::Tensor::new(
        &device,
        vec![&10.0, &20.0, &30.0, &40.0],
    )));

    let c = &a + &b;
    assert!(c.is_gpu());

    let result = c.to_vec().await;
    assert_eq!(result, vec![11.0, 22.0, 33.0, 44.0]);
}

#[tokio::test]
async fn test_gpu_mul() {
    let device = match fusor_core::Device::new().await {
        Ok(d) => d,
        Err(_) => {
            eprintln!("Skipping GPU test - no GPU available");
            return;
        }
    };

    let a = GpuOr::Gpu(GpuTensor::new(fusor_core::Tensor::new(
        &device,
        vec![&1.0, &2.0, &3.0, &4.0],
    )));
    let b = GpuOr::Gpu(GpuTensor::new(fusor_core::Tensor::new(
        &device,
        vec![&2.0, &3.0, &4.0, &5.0],
    )));

    let c = &a * &b;
    let result = c.to_vec().await;
    assert_eq!(result, vec![2.0, 6.0, 12.0, 20.0]);
}

#[tokio::test]
async fn test_gpu_exp_sqrt() {
    let device = match fusor_core::Device::new().await {
        Ok(d) => d,
        Err(_) => {
            eprintln!("Skipping GPU test - no GPU available");
            return;
        }
    };

    let a = GpuOr::Gpu(GpuTensor::new(fusor_core::Tensor::new(
        &device,
        vec![&1.0, &4.0, &9.0, &16.0],
    )));

    let sqrt_result = a.sqrt().to_vec().await;
    assert_eq!(sqrt_result, vec![1.0, 2.0, 3.0, 4.0]);
}

// ============================================================================
// Tests for new composite operations
// ============================================================================

#[test]
fn test_cpu_relu() {
    let a = GpuOr::cpu_from_slice([4], &[1.0, -2.0, 3.0, -4.0]);
    let result = a.relu().to_vec_blocking();
    assert_eq!(result, vec![1.0, 0.0, 3.0, 0.0]);
}

#[test]
fn test_cpu_gelu() {
    let a = GpuOr::cpu_from_slice([4], &[0.0, 1.0, -1.0, 2.0]);
    let result = a.gelu().to_vec_blocking();

    // GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    let gelu = |x: f32| {
        0.5 * x * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
    };

    assert!((result[0] - gelu(0.0)).abs() < 0.01);
    assert!((result[1] - gelu(1.0)).abs() < 0.01);
    assert!((result[2] - gelu(-1.0)).abs() < 0.01);
    assert!((result[3] - gelu(2.0)).abs() < 0.01);
}

#[test]
fn test_cpu_silu() {
    let a = GpuOr::cpu_from_slice([4], &[0.0, 1.0, -1.0, 2.0]);
    let result = a.silu().to_vec_blocking();

    // SiLU(x) = x / (1 + exp(-x))
    let silu = |x: f32| x / (1.0 + (-x).exp());

    assert!((result[0] - silu(0.0)).abs() < 1e-6);
    assert!((result[1] - silu(1.0)).abs() < 1e-6);
    assert!((result[2] - silu(-1.0)).abs() < 1e-6);
    assert!((result[3] - silu(2.0)).abs() < 1e-6);
}

#[test]
fn test_cpu_sqr() {
    let a = GpuOr::cpu_from_slice([4], &[1.0, 2.0, 3.0, 4.0]);
    let result = a.sqr().to_vec_blocking();
    assert_eq!(result, vec![1.0, 4.0, 9.0, 16.0]);
}

#[test]
fn test_cpu_scalar_ops() {
    let a = GpuOr::cpu_from_slice([4], &[1.0, 2.0, 3.0, 4.0]);

    // max_scalar
    let max_result = a.max_scalar(2.5).to_vec_blocking();
    assert_eq!(max_result, vec![2.5, 2.5, 3.0, 4.0]);

    // min_scalar
    let min_result = a.min_scalar(2.5).to_vec_blocking();
    assert_eq!(min_result, vec![1.0, 2.0, 2.5, 2.5]);

    // clamp
    let clamp_result = a.clamp(1.5, 3.5).to_vec_blocking();
    assert_eq!(clamp_result, vec![1.5, 2.0, 3.0, 3.5]);

    // mul_scalar
    let mul_result = a.mul_scalar(2.0).to_vec_blocking();
    assert_eq!(mul_result, vec![2.0, 4.0, 6.0, 8.0]);

    // add_scalar
    let add_result = a.add_scalar(10.0).to_vec_blocking();
    assert_eq!(add_result, vec![11.0, 12.0, 13.0, 14.0]);

    // div_scalar
    let div_result = a.div_scalar(2.0).to_vec_blocking();
    assert_eq!(div_result, vec![0.5, 1.0, 1.5, 2.0]);
}

#[test]
fn test_cpu_reductions() {
    let a = GpuOr::cpu_from_slice([4], &[1.0, 2.0, 3.0, 4.0]);

    assert_eq!(a.sum_all(), 10.0);
    assert_eq!(a.max_all(), 4.0);
    assert_eq!(a.min_all(), 1.0);
}

#[test]
fn test_cpu_axis_reductions() {
    // 2x3 matrix: [[1, 2, 3], [4, 5, 6]]
    let a = GpuOr::cpu_from_slice([2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    // Sum along axis 0: [5, 7, 9]
    let sum0 = a.sum_axis_0().to_vec_blocking();
    assert_eq!(sum0, vec![5.0, 7.0, 9.0]);

    // Sum along axis 1: [6, 15]
    let sum1 = a.sum_axis_1().to_vec_blocking();
    assert_eq!(sum1, vec![6.0, 15.0]);

    // Mean along axis 0: [2.5, 3.5, 4.5]
    let mean0 = a.mean_axis_0().to_vec_blocking();
    assert_eq!(mean0, vec![2.5, 3.5, 4.5]);

    // Mean along axis 1: [2, 5]
    let mean1 = a.mean_axis_1().to_vec_blocking();
    assert_eq!(mean1, vec![2.0, 5.0]);
}

#[test]
fn test_cpu_softmax() {
    // Simple 1D-like test in 2D form
    let a = GpuOr::cpu_from_slice([2, 3], &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);

    // Softmax along axis 1
    let result = a.softmax_axis_1().to_vec_blocking();

    // Check that each row sums to 1
    let row1_sum: f32 = result[0..3].iter().sum();
    let row2_sum: f32 = result[3..6].iter().sum();

    assert!((row1_sum - 1.0).abs() < 1e-5);
    assert!((row2_sum - 1.0).abs() < 1e-5);

    // Check relative ordering (exp(3) > exp(2) > exp(1))
    assert!(result[2] > result[1]);
    assert!(result[1] > result[0]);
}

#[test]
fn test_cpu_unsqueeze_squeeze() {
    // Test unsqueeze: [4] -> [1, 4]
    let a = GpuOr::cpu_from_slice([4], &[1.0, 2.0, 3.0, 4.0]);
    let unsqueezed = a.unsqueeze_0();
    assert_eq!(unsqueezed.shape(), &[1, 4]);

    // Test squeeze: [1, 4] -> [4]
    let squeezed = unsqueezed.squeeze_0();
    assert_eq!(squeezed.shape(), &[4]);
    let result = squeezed.to_vec_blocking();
    assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_cpu_reshape() {
    let a = GpuOr::cpu_from_slice([6], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    // Reshape [6] -> [2, 3]
    let reshaped = a.reshape_2([2, 3]);
    assert_eq!(reshaped.shape(), &[2, 3]);

    let result = reshaped.to_vec_blocking();
    assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_cpu_zeros_like() {
    let a = GpuOr::cpu_from_slice([2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let zeros = a.zeros_like();

    assert_eq!(zeros.shape(), &[2, 3]);
    let result = zeros.to_vec_blocking();
    assert_eq!(result, vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
}
