//! Linear layer implementation.

use crate::{ConcreteTensor, Device, GpuOr, MatmulImpl, SimdElement};
use fusor_core::{DataType, FloatDataType};
use fusor_cpu::FloatOps;

/// A linear (fully connected) layer.
///
/// Computes `output = input @ weight.T + bias`
pub struct Linear<D: SimdElement> {
    weight: GpuOr<2, D, ConcreteTensor<D, 2>>,
    bias: Option<GpuOr<1, D, ConcreteTensor<D, 1>>>,
}

impl<D> Linear<D>
where
    D: SimdElement
        + DataType
        + FloatDataType
        + FloatOps
        + Default
        + MatmulImpl
        + std::ops::Mul<Output = D>
        + std::ops::Add<Output = D>,
{
    /// Create a new Linear layer with the given weight and optional bias.
    ///
    /// Weight shape: (out_features, in_features)
    /// Bias shape: (out_features,)
    pub fn new(
        weight: GpuOr<2, D, ConcreteTensor<D, 2>>,
        bias: Option<GpuOr<1, D, ConcreteTensor<D, 1>>>,
    ) -> Self {
        Self { weight, bias }
    }

    /// Create a new Linear layer with random weights for testing.
    pub fn zeros(device: &Device, in_features: usize, out_features: usize) -> Self {
        let weight = GpuOr::zeros(device, [out_features, in_features]);
        let bias = Some(GpuOr::zeros(device, [out_features]));
        Self { weight, bias }
    }

    /// Forward pass for 2D input (batch, in_features)
    ///
    /// Input shape: (batch, in_features)
    /// Output shape: (batch, out_features)
    pub fn forward_2d(
        &self,
        input: &GpuOr<2, D, ConcreteTensor<D, 2>>,
    ) -> GpuOr<2, D, ConcreteTensor<D, 2>>
    where
        crate::AddOp: fusor_cpu::SimdBinaryOp<D>,
    {
        // Transpose weight: (out_features, in_features) -> (in_features, out_features)
        let weight_t = self.weight.t();
        // Matmul: (batch, in_features) @ (in_features, out_features) = (batch, out_features)
        let output = input.matmul(&weight_t);

        if let Some(bias) = &self.bias {
            // Broadcast bias (out_features,) to (batch, out_features) and add
            let bias_broadcast: GpuOr<2, D, _> = bias.broadcast_as(output.shape());
            match (&output, &bias_broadcast) {
                (GpuOr::Cpu(a), GpuOr::Cpu(b)) => GpuOr::Cpu((a + b).eval()),
                (GpuOr::Gpu(a), GpuOr::Gpu(b)) => GpuOr::Gpu(a.add_(b)),
                _ => panic!("Cannot mix CPU and GPU tensors"),
            }
        } else {
            output
        }
    }

    /// Forward pass for 3D input (batch, seq_len, in_features)
    ///
    /// Input shape: (batch, seq_len, in_features)
    /// Output shape: (batch, seq_len, out_features)
    pub fn forward(
        &self,
        input: &GpuOr<3, D, ConcreteTensor<D, 3>>,
    ) -> GpuOr<3, D, ConcreteTensor<D, 3>>
    where
        crate::AddOp: fusor_cpu::SimdBinaryOp<D>,
    {
        let [batch, seq_len, in_features] = input.shape();
        let out_features = self.weight.shape()[0];

        // Reshape to 2D: (batch * seq_len, in_features)
        let input_2d: GpuOr<2, D, _> = input.reshape([batch * seq_len, in_features]);

        // Forward through 2D
        let output_2d = self.forward_2d(&input_2d);

        // Reshape back to 3D
        output_2d.reshape([batch, seq_len, out_features])
    }

    /// Get the weight tensor.
    pub fn weight(&self) -> &GpuOr<2, D, ConcreteTensor<D, 2>> {
        &self.weight
    }

    /// Get the bias tensor if present.
    pub fn bias(&self) -> Option<&GpuOr<1, D, ConcreteTensor<D, 1>>> {
        self.bias.as_ref()
    }

    /// Get the input features size.
    pub fn in_features(&self) -> usize {
        self.weight.shape()[1]
    }

    /// Get the output features size.
    pub fn out_features(&self) -> usize {
        self.weight.shape()[0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_linear_2d() {
        let _device = Device::cpu();

        // Weight: (3, 2) - 3 out features, 2 in features
        let weight_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let weight: GpuOr<2, f32> =
            GpuOr::Cpu(fusor_cpu::Tensor::from_slice([3, 2], &weight_data));

        // Bias: (3,)
        let bias_data = [0.1f32, 0.2, 0.3];
        let bias: GpuOr<1, f32> = GpuOr::Cpu(fusor_cpu::Tensor::from_slice([3], &bias_data));

        let linear = Linear::new(weight, Some(bias));

        // Input: (2, 2) - batch=2, in_features=2
        let input_data = [1.0f32, 2.0, 3.0, 4.0];
        let input: GpuOr<2, f32> =
            GpuOr::Cpu(fusor_cpu::Tensor::from_slice([2, 2], &input_data));

        let output = linear.forward_2d(&input);
        let result = output.as_slice().await.unwrap();

        // Manual calculation for batch 0: [1, 2] @ [[1, 3, 5], [2, 4, 6]] + [0.1, 0.2, 0.3]
        // = [1*1+2*2, 1*3+2*4, 1*5+2*6] + [0.1, 0.2, 0.3]
        // = [5, 11, 17] + [0.1, 0.2, 0.3] = [5.1, 11.2, 17.3]
        assert!((result[[0, 0]] - 5.1).abs() < 1e-5);
        assert!((result[[0, 1]] - 11.2).abs() < 1e-5);
        assert!((result[[0, 2]] - 17.3).abs() < 1e-5);
    }

    #[tokio::test]
    async fn test_linear_3d() {
        let _device = Device::cpu();

        // Weight: (3, 2) - 3 out features, 2 in features
        let weight_data = [1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0];
        let weight: GpuOr<2, f32> =
            GpuOr::Cpu(fusor_cpu::Tensor::from_slice([3, 2], &weight_data));

        let linear = Linear::new(weight, None);

        // Input: (1, 2, 2) - batch=1, seq_len=2, in_features=2
        let input_data = [1.0f32, 2.0, 3.0, 4.0];
        let input: GpuOr<3, f32> =
            GpuOr::Cpu(fusor_cpu::Tensor::from_slice([1, 2, 2], &input_data));

        let output = linear.forward(&input);
        let result = output.as_slice().await.unwrap();

        assert_eq!(result.shape(), &[1, 2, 3]);

        // [1, 2] @ [[1, 0, 1], [0, 1, 1]] = [1, 2, 3]
        assert!((result[[0, 0, 0]] - 1.0).abs() < 1e-5);
        assert!((result[[0, 0, 1]] - 2.0).abs() < 1e-5);
        assert!((result[[0, 0, 2]] - 3.0).abs() < 1e-5);
    }
}
