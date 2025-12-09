use crate::{
    CastTensor, DataType, Device, LastRank, MaxRank, NextRankInner, Result, Tensor, VarBuilder,
};

/// Root Mean Square Normalization
///
/// Normalizes the input over the last dimension.
/// Formula: output = input / sqrt(variance + eps) * weight + bias
pub struct RmsNorm<const N: usize, T> {
    weight: Tensor<N, T>,
    bias: Option<Tensor<N, T>>,
    eps: f32,
}

impl<const N: usize, T: DataType> RmsNorm<N, T> {
    /// Create a new RmsNorm layer
    pub fn new(weight: Tensor<N, T>, bias: Option<Tensor<N, T>>, eps: f32) -> Self {
        Self { weight, bias, eps }
    }

    /// Load RmsNorm from VarBuilder
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

impl<const R: usize, T> RmsNorm<R, T> {
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
        input.rms_norm_fused(&self.weight, self.bias.as_ref(), self.eps)
    }
}
