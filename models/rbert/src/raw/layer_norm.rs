use fusor_core::{Device, Result, Tensor};

pub fn layer_norm(device: &Device, vb: &mut crate::VarBuilder, eps: f32) -> Result<LayerNorm> {
    let weight = vb.get("weight", device)?.dequantize();
    Ok(LayerNorm { weight, eps })
}

pub struct LayerNorm {
    weight: Tensor<1, f32>,
    eps: f32,
}

impl LayerNorm {
    pub fn forward(&self, input: &Tensor<2, f32>) -> Tensor<2, f32> {
        input.layer_norm(&self.weight, self.eps)
    }
}
