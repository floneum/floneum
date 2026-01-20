//! Layer normalization implementation.

use crate::{ConcreteTensor, GpuOr, SimdElement};
use fusor_core::{DataType, FloatDataType};
use fusor_cpu::FloatOps;

/// Layer Normalization.
///
/// Normalizes the input over the last dimension.
/// Formula: output = (input - mean) / sqrt(variance + eps) * weight + bias
pub struct LayerNorm<D: SimdElement> {
    weight: GpuOr<1, D, ConcreteTensor<D, 1>>,
    bias: Option<GpuOr<1, D, ConcreteTensor<D, 1>>>,
    eps: D,
}

impl<D> LayerNorm<D>
where
    D: SimdElement + DataType + FloatDataType + FloatOps + Default,
{
    /// Create a new LayerNorm layer.
    ///
    /// Weight and bias should have shape (normalized_dim,).
    pub fn new(
        weight: GpuOr<1, D, ConcreteTensor<D, 1>>,
        bias: Option<GpuOr<1, D, ConcreteTensor<D, 1>>>,
        eps: D,
    ) -> Self {
        Self { weight, bias, eps }
    }

    /// Get the weight tensor.
    pub fn weight(&self) -> &GpuOr<1, D, ConcreteTensor<D, 1>> {
        &self.weight
    }

    /// Get the bias tensor if present.
    pub fn bias(&self) -> Option<&GpuOr<1, D, ConcreteTensor<D, 1>>> {
        self.bias.as_ref()
    }

    /// Get the epsilon value.
    pub fn eps(&self) -> D {
        self.eps
    }

    /// Forward pass for 2D input (batch, features).
    ///
    /// Normalizes over the last dimension (features).
    pub fn forward_2d(
        &self,
        input: &GpuOr<2, D, ConcreteTensor<D, 2>>,
    ) -> GpuOr<2, D, ConcreteTensor<D, 2>>
    where
        D: std::ops::Add<Output = D>
            + std::ops::Sub<Output = D>
            + std::ops::Mul<Output = D>
            + std::ops::Div<Output = D>,
        crate::AddOp: fusor_cpu::SimdBinaryOp<D>,
        crate::SubOp: fusor_cpu::SimdBinaryOp<D>,
        crate::MulOp: fusor_cpu::SimdBinaryOp<D>,
        crate::DivOp: fusor_cpu::SimdBinaryOp<D>,
        fusor_cpu::SumOp: fusor_cpu::SimdReduceOp<D>,
        fusor_cpu::SqrtOp: fusor_cpu::SimdUnaryOp<D>,
    {
        // Broadcast weight to input shape
        let weight_broadcast: GpuOr<2, D, _> = self.weight.broadcast_as(input.shape());
        let bias_broadcast: Option<GpuOr<2, D, _>> =
            self.bias.as_ref().map(|b| b.broadcast_as(input.shape()));
        input.layer_norm(&weight_broadcast, bias_broadcast.as_ref(), self.eps, true)
    }

    /// Forward pass for 3D input (batch, seq_len, features).
    ///
    /// Normalizes over the last dimension (features).
    pub fn forward(
        &self,
        input: &GpuOr<3, D, ConcreteTensor<D, 3>>,
    ) -> GpuOr<3, D, ConcreteTensor<D, 3>>
    where
        D: std::ops::Add<Output = D>
            + std::ops::Sub<Output = D>
            + std::ops::Mul<Output = D>
            + std::ops::Div<Output = D>,
        crate::AddOp: fusor_cpu::SimdBinaryOp<D>,
        crate::SubOp: fusor_cpu::SimdBinaryOp<D>,
        crate::MulOp: fusor_cpu::SimdBinaryOp<D>,
        crate::DivOp: fusor_cpu::SimdBinaryOp<D>,
        fusor_cpu::SumOp: fusor_cpu::SimdReduceOp<D>,
        fusor_cpu::SqrtOp: fusor_cpu::SimdUnaryOp<D>,
    {
        // Broadcast weight to input shape
        let weight_broadcast: GpuOr<3, D, _> = self.weight.broadcast_as(input.shape());
        let bias_broadcast: Option<GpuOr<3, D, _>> =
            self.bias.as_ref().map(|b| b.broadcast_as(input.shape()));
        input.layer_norm(&weight_broadcast, bias_broadcast.as_ref(), self.eps, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_layer_norm_2d() {
        // Weight and bias: (3,)
        let weight_data = [1.0f32, 1.0, 1.0];
        let bias_data = [0.0f32, 0.0, 0.0];
        let weight: GpuOr<1, f32> =
            GpuOr::Cpu(fusor_cpu::Tensor::from_slice([3], &weight_data));
        let bias: GpuOr<1, f32> =
            GpuOr::Cpu(fusor_cpu::Tensor::from_slice([3], &bias_data));

        let layer_norm = LayerNorm::new(weight, Some(bias), 1e-5);

        // Input: (2, 3)
        let input_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input: GpuOr<2, f32> =
            GpuOr::Cpu(fusor_cpu::Tensor::from_slice([2, 3], &input_data));

        let output = layer_norm.forward_2d(&input);
        let result = output.as_slice().await.unwrap();

        assert_eq!(result.shape(), &[2, 3]);

        // Each row should have mean ~0 and std ~1 after normalization
        // For [1, 2, 3]: mean=2, std=sqrt(2/3)
        // Normalized: [-sqrt(3/2), 0, sqrt(3/2)] ≈ [-1.22, 0, 1.22]
        let expected_val = (3.0f32 / 2.0).sqrt();
        assert!((result[[0, 0]] - (-expected_val)).abs() < 1e-4);
        assert!(result[[0, 1]].abs() < 1e-4);
        assert!((result[[0, 2]] - expected_val).abs() < 1e-4);
    }

    #[tokio::test]
    async fn test_layer_norm_3d() {
        let weight_data = [1.0f32, 1.0];
        let weight: GpuOr<1, f32> =
            GpuOr::Cpu(fusor_cpu::Tensor::from_slice([2], &weight_data));

        let layer_norm = LayerNorm::new(weight, None, 1e-5);

        // Input: (1, 2, 2)
        let input_data = [1.0f32, 3.0, 2.0, 4.0];
        let input: GpuOr<3, f32> =
            GpuOr::Cpu(fusor_cpu::Tensor::from_slice([1, 2, 2], &input_data));

        let output = layer_norm.forward(&input);
        let result = output.as_slice().await.unwrap();

        assert_eq!(result.shape(), &[1, 2, 2]);

        // First position [1, 3]: mean=2, std=1, normalized=[-1, 1]
        assert!((result[[0, 0, 0]] - (-1.0)).abs() < 1e-4);
        assert!((result[[0, 0, 1]] - 1.0).abs() < 1e-4);
    }
}
