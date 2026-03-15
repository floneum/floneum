//! LayerNorm2d implementation for normalizing over the channel dimension of BCHW tensors.

use crate::{ConcreteTensor, Device, Tensor, VarBuilder};

/// Layer Normalization for 2D spatial data (channel-wise normalization).
///
/// Unlike standard LayerNorm which normalizes the last dimension,
/// LayerNorm2d normalizes over the channel dimension (dim=1) of BCHW tensors.
///
/// Formula: output = (input - mean) / sqrt(variance + eps) * weight + bias
/// where mean and variance are computed over the channel dimension.
pub struct LayerNorm2d {
    weight: Tensor<1, f32, ConcreteTensor<f32, 1>>,
    bias: Tensor<1, f32, ConcreteTensor<f32, 1>>,
    num_channels: usize,
    eps: f32,
}

impl LayerNorm2d {
    /// Create a new LayerNorm2d layer.
    pub fn new(
        weight: Tensor<1, f32, ConcreteTensor<f32, 1>>,
        bias: Tensor<1, f32, ConcreteTensor<f32, 1>>,
        num_channels: usize,
        eps: f32,
    ) -> Self {
        Self {
            weight,
            bias,
            num_channels,
            eps,
        }
    }

    /// Forward pass for 4D input (batch, channels, height, width).
    ///
    /// Normalizes over the channel dimension (dim=1).
    pub fn forward<B>(&self, xs: &Tensor<4, f32, B>) -> Tensor<4, f32>
    where
        B: fusor_cpu::TensorBacking<4, Elem = f32>,
    {
        let shape = xs.shape();

        let u: Tensor<4, f32> = xs.mean_keepdim(1);
        let u_broadcast: Tensor<4, f32> = u.broadcast_as(shape).to_concrete();
        let xs_centered: Tensor<4, f32> = (xs - &u_broadcast).to_concrete();
        let s: Tensor<4, f32> = (&xs_centered * &xs_centered).to_concrete().mean_keepdim(1);
        let s_eps = (s + self.eps).to_concrete();
        let denom: Tensor<4, f32> = s_eps.sqrt().broadcast_as(shape).to_concrete();
        let xs_norm: Tensor<4, f32> = (&xs_centered / denom).to_concrete();

        let w: Tensor<4, f32> = self
            .weight
            .reshape([1, self.num_channels, 1, 1])
            .broadcast_as(shape)
            .to_concrete();
        let b: Tensor<4, f32> = self
            .bias
            .reshape([1, self.num_channels, 1, 1])
            .broadcast_as(shape)
            .to_concrete();

        (xs_norm * w).to_concrete().add_(&b)
    }

    /// Load LayerNorm2d from VarBuilder.
    pub fn load(device: &Device, vb: &mut VarBuilder, eps: f32) -> crate::Result<Self> {
        let weight: Tensor<1, f32> = vb.get("weight", device)?.dequantize();
        let bias: Tensor<1, f32> = vb.get("bias", device)?.dequantize();
        let num_channels = weight.shape()[0];
        Ok(Self::new(
            weight.to_concrete(),
            bias.to_concrete(),
            num_channels,
            eps,
        ))
    }
}
