use std::fmt::Debug;

use fusor_core::{Device, LastRank, MaxRank, MappedBuffer, NextRankInner, Result, Tensor, TensorSlice};

pub fn layer_norm<const N: usize>(
    device: &Device,
    vb: &mut crate::VarBuilder,
    eps: f32,
) -> Result<LayerNorm<N>> {
    let weight = vb.get("weight", device)?.dequantize();
    let bias = vb.get("bias", device).ok().map(|b| b.dequantize());
    Ok(LayerNorm { weight, bias, eps })
}

pub struct LayerNorm<const N: usize> {
    weight: Tensor<N, f32>,
    bias: Option<Tensor<N, f32>>,
    eps: f32,
}

impl<const N: usize> LayerNorm<N> {
    pub fn forward<const N2: usize, const N3: usize>(
        &self,
        input: &Tensor<N2, f32>,
    ) -> Tensor<N2, f32>
    where
        Tensor<N2, f32>: LastRank<N3, f32>,
        (Tensor<N2, f32>, Tensor<N, f32>): MaxRank<N2, f32>,
        (Tensor<N2, f32>, Tensor<N2, f32>): MaxRank<N2, f32>,
        Tensor<N3, f32>: NextRankInner<NextRank = Tensor<N2, f32>>,
        TensorSlice<N2, f32, MappedBuffer>: Debug,
    {
        input.layer_norm(&self.weight, self.bias.as_ref(), self.eps, true)
    }
}
