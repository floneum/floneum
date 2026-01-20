//! Construction operations that work on both CPU and GPU backends.

use crate::{Device, Tensor, SimdElement};
use fusor_core::DataType;
use fusor_cpu::Expr;

impl<const R: usize, D> Tensor<R, D>
where
    D: SimdElement + DataType + Default,
{
    /// Create a tensor filled with zeros.
    pub fn zeros(device: &Device, shape: [usize; R]) -> Self {
        match device {
            Device::Cpu => Tensor::Cpu(fusor_cpu::Tensor::zeros(shape)),
            Device::Gpu(gpu_device) => Tensor::Gpu(fusor_core::Tensor::zeros(gpu_device, shape)),
        }
    }

    /// Create a tensor filled with zeros that has the same shape as this tensor.
    pub fn zeros_like(&self) -> Self {
        match self {
            Tensor::Cpu(t) => {
                let shape: [usize; R] = t
                    .inner()
                    .shape()
                    .try_into()
                    .expect("Shape length mismatch");
                Tensor::Cpu(fusor_cpu::Tensor::zeros(shape))
            }
            Tensor::Gpu(t) => Tensor::Gpu(t.zeros_like()),
        }
    }

    /// Create a tensor filled with a specific value.
    pub fn splat(device: &Device, value: D, shape: [usize; R]) -> Self {
        match device {
            Device::Cpu => {
                let data = vec![value; shape.iter().product()];
                Tensor::Cpu(fusor_cpu::Tensor::from_slice(shape, &data))
            }
            Device::Gpu(gpu_device) => Tensor::Gpu(fusor_core::Tensor::splat(gpu_device, value, shape)),
        }
    }

    /// Create a tensor filled with a specific value (alias for splat).
    pub fn full(device: &Device, shape: [usize; R], value: D) -> Self {
        Self::splat(device, value, shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_zeros_cpu() {
        let device = Device::Cpu;
        let t: Tensor<2, f32> = Tensor::zeros(&device, [2, 3]);
        let slice = t.as_slice().await.unwrap();

        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(slice[[i, j]], 0.0);
            }
        }
    }

    #[tokio::test]
    async fn test_zeros_like_cpu() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t: Tensor<2, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice([2, 3], &data));
        let zeros = t.zeros_like();
        let slice = zeros.as_slice().await.unwrap();

        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(slice[[i, j]], 0.0);
            }
        }
    }

    #[tokio::test]
    async fn test_splat_cpu() {
        let device = Device::Cpu;
        let t: Tensor<2, f32> = Tensor::splat(&device, 5.0, [2, 3]);
        let slice = t.as_slice().await.unwrap();

        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(slice[[i, j]], 5.0);
            }
        }
    }
}
