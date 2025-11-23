use crate::{
    CastTensor, DataType, Device, LastRank, MaxRank, NextRankInner, Result, Tensor, VarBuilder,
};

/// Layer Normalization
///
/// Normalizes the input over the last dimension.
/// Formula: output = (input - mean) / sqrt(variance + eps) * weight + bias
pub struct LayerNorm<const N: usize, T> {
    weight: Tensor<N, T>,
    bias: Option<Tensor<N, T>>,
    eps: f32,
}

impl<const N: usize, T: DataType> LayerNorm<N, T> {
    /// Create a new LayerNorm layer
    pub fn new(weight: Tensor<N, T>, bias: Option<Tensor<N, T>>, eps: f32) -> Self {
        Self { weight, bias, eps }
    }

    /// Load LayerNorm from VarBuilder
    ///
    /// Expects:
    /// - weight: Tensor with shape matching the normalized dimension
    /// - bias (optional): Tensor with same shape as weight
    pub fn load(device: &Device, vb: &mut VarBuilder, eps: f32) -> Result<Self> {
        let weight = vb.get("weight", device)?.dequantize();
        let bias = vb.get("bias", device).ok().map(|b| b.dequantize());
        Ok(Self::new(weight, bias, eps))
    }

    pub fn weight(&self) -> &Tensor<N, T> {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor<N, T>> {
        self.bias.as_ref()
    }

    pub fn eps(&self) -> f32 {
        self.eps
    }
}

impl<const R: usize, T> LayerNorm<R, T> {
    /// Forward pass
    ///
    /// Normalizes the input over the last dimension
    pub fn forward<const N: usize, const N2: usize>(&self, input: &Tensor<N, T>) -> Tensor<N, T>
    where
        T: DataType + CastTensor<f32>,
        f32: CastTensor<T>,
        (Tensor<N, f32>, Tensor<R, f32>): MaxRank<N, f32>,
        (Tensor<N, T>, Tensor<R, T>): MaxRank<N, T>,
        (Tensor<N, f32>, Tensor<N, f32>): MaxRank<N, f32>,
        Tensor<N, f32>: LastRank<N2, f32>,
        Tensor<N2, f32>: NextRankInner<NextRank = Tensor<N, f32>>,
    {
        // remove_mean = true for standard LayerNorm (as opposed to RMSNorm)
        input.layer_norm(&self.weight, self.bias.as_ref(), self.eps, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;

    #[tokio::test]
    async fn test_layer_norm_shape() {
        let device = Device::new().await.unwrap();

        // Create weight and bias
        let weight_data = [1.0, 1.0, 1.0];
        let bias_data = [0.0, 0.0, 0.0];
        let weight = Tensor::new(&device, &weight_data);
        let bias = Tensor::new(&device, &bias_data);

        let layer_norm = LayerNorm::new(weight, Some(bias), 1e-5);

        // Input: [1.0, 2.0, 3.0]
        let input_data = [1.0, 2.0, 3.0];
        let input = Tensor::new(&device, &input_data);

        let result = layer_norm.forward(&input);

        assert_eq!(result.shape(), &[3]);
    }

    #[tokio::test]
    async fn test_layer_norm_properties() {
        let device = Device::new().await.unwrap();

        let weight_data = [1.0, 1.0];
        let bias_data = [0.0, 0.0];
        let weight = Tensor::new(&device, &weight_data);
        let bias = Tensor::new(&device, &bias_data);

        let layer_norm = LayerNorm::new(weight.clone(), Some(bias.clone()), 1e-5);

        assert_eq!(layer_norm.eps(), 1e-5);
        assert_eq!(layer_norm.weight().shape(), &[2]);
        assert!(layer_norm.bias().is_some());
    }
}
