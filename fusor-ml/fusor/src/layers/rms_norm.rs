//! RMS normalization implementation.

use crate::{ConcreteTensor, Tensor, SimdElement};
use fusor_core::{DataType, FloatDataType};
use fusor_cpu::FloatOps;

/// Root Mean Square Normalization.
///
/// Normalizes the input over the last dimension without centering.
/// Formula: output = input / sqrt(mean(x^2) + eps) * weight
pub struct RmsNorm<D: SimdElement> {
    weight: Tensor<1, D, ConcreteTensor<D, 1>>,
    eps: D,
}

impl<D> RmsNorm<D>
where
    D: SimdElement + DataType + FloatDataType + FloatOps + Default,
{
    /// Create a new RmsNorm layer.
    ///
    /// Weight should have shape (normalized_dim,).
    pub fn new(weight: Tensor<1, D, ConcreteTensor<D, 1>>, eps: D) -> Self {
        Self { weight, eps }
    }

    /// Get the weight tensor.
    pub fn weight(&self) -> &Tensor<1, D, ConcreteTensor<D, 1>> {
        &self.weight
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
        input: &Tensor<2, D, ConcreteTensor<D, 2>>,
    ) -> Tensor<2, D, ConcreteTensor<D, 2>>
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
        let weight_broadcast: Tensor<2, D, _> = self.weight.broadcast_as(input.shape());
        input.rms_norm(&weight_broadcast, self.eps)
    }

    /// Forward pass for 3D input (batch, seq_len, features).
    ///
    /// Normalizes over the last dimension (features).
    pub fn forward(
        &self,
        input: &Tensor<3, D, ConcreteTensor<D, 3>>,
    ) -> Tensor<3, D, ConcreteTensor<D, 3>>
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
        let weight_broadcast: Tensor<3, D, _> = self.weight.broadcast_as(input.shape());
        input.rms_norm(&weight_broadcast, self.eps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_rms_norm_2d() {
        // Weight: (3,)
        let weight_data = [1.0f32, 1.0, 1.0];
        let weight: Tensor<1, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([3], &weight_data));

        let rms_norm = RmsNorm::new(weight, 1e-5);

        // Input: (2, 3)
        let input_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input: Tensor<2, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([2, 3], &input_data));

        let output = rms_norm.forward_2d(&input);
        let result = output.as_slice().await.unwrap();

        assert_eq!(result.shape(), &[2, 3]);

        // RMS for [1, 2, 3] = sqrt((1+4+9)/3) = sqrt(14/3) ≈ 2.16
        // Normalized: [1/2.16, 2/2.16, 3/2.16] ≈ [0.46, 0.93, 1.39]
        let rms = ((1.0 + 4.0 + 9.0) / 3.0f32).sqrt();
        assert!((result[[0, 0]] - 1.0 / rms).abs() < 1e-4);
        assert!((result[[0, 1]] - 2.0 / rms).abs() < 1e-4);
        assert!((result[[0, 2]] - 3.0 / rms).abs() < 1e-4);
    }

    #[tokio::test]
    async fn test_rms_norm_3d() {
        let weight_data = [2.0f32, 2.0];
        let weight: Tensor<1, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([2], &weight_data));

        let rms_norm = RmsNorm::new(weight, 1e-5);

        // Input: (1, 2, 2)
        let input_data = [3.0f32, 4.0, 6.0, 8.0];
        let input: Tensor<3, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 2, 2], &input_data));

        let output = rms_norm.forward(&input);
        let result = output.as_slice().await.unwrap();

        assert_eq!(result.shape(), &[1, 2, 2]);

        // RMS for [3, 4] = sqrt((9+16)/2) = sqrt(12.5) ≈ 3.54
        // Normalized * weight: [3/3.54*2, 4/3.54*2] ≈ [1.70, 2.26]
        let rms1 = ((9.0 + 16.0) / 2.0f32).sqrt();
        assert!((result[[0, 0, 0]] - 3.0 / rms1 * 2.0).abs() < 1e-4);
        assert!((result[[0, 0, 1]] - 4.0 / rms1 * 2.0).abs() < 1e-4);
    }
}
