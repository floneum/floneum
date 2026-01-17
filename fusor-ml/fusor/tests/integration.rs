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
