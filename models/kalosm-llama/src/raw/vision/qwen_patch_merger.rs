use candle_core::{Result, Tensor};
use candle_transformers::{
    quantized_nn::{Linear, RmsNorm},
    quantized_var_builder::VarBuilder,
};

use super::QWEN_EPS;

pub(crate) struct Qwen2VLPatchMerger {
    hidden_size: usize,
    ln_q: RmsNorm,
    mlp: [Linear; 2],
}

impl Qwen2VLPatchMerger {
    pub(crate) fn new(
        dim: usize,
        context_dim: usize,
        spatial_merge_size: usize,
        vb: &VarBuilder,
    ) -> Result<Self> {
        let hidden_size = context_dim * spatial_merge_size.pow(2);
        let ln_q = RmsNorm::new(context_dim, QWEN_EPS, vb.pp("ln_q"))?;
        let mlp_0_weight = vb.get((hidden_size, hidden_size), "mlp.0.weight")?;
        let mlp_0_bias = vb
            .get((hidden_size,), "mlp.0.bias")?
            .dequantize(vb.device())?;
        let mlp_2_weight = vb.get((dim, hidden_size), "mlp.2.weight")?;
        let mlp_2_bias = vb.get((dim,), "mlp.2.bias")?.dequantize(vb.device())?;
        let mlp = [
            Linear::from_arc(mlp_0_weight, Some(mlp_0_bias))?,
            Linear::from_arc(mlp_2_weight, Some(mlp_2_bias))?,
        ];

        Ok(Self {
            hidden_size,
            ln_q,
            mlp,
        })
    }

    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.unsqueeze(0)?
            .apply(&self.ln_q)?
            .reshape(((), self.hidden_size))?
            .apply(&self.mlp[0])?
            .gelu()?
            .apply(&self.mlp[1])?
            .squeeze(0)
    }
}
