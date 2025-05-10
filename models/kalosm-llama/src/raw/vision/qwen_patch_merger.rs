use std::sync::Arc;

use candle_core::{quantized::QTensor, DType, Device, Result, Tensor};
use candle_transformers::{
    quantized_nn::{Linear, RmsNorm},
    quantized_var_builder::VarBuilder,
};

use super::QWEN_EPS;

struct Qwen2VLPatchMerger {
    hidden_size: usize,
    ln_q: RmsNorm,
    mlp: [Linear; 2],
}

impl Qwen2VLPatchMerger {
    fn new(
        dim: usize,
        context_dim: usize,
        spatial_merge_size: usize,
        vb: &VarBuilder,
    ) -> Result<Self> {
        let hidden_size = context_dim * spatial_merge_size.pow(2);
        let ln_q = RmsNorm::new(context_dim, QWEN_EPS, vb.pp("ln_q"))?;
        let mlp_0 = vb.get((hidden_size, hidden_size), "mlp.0")?;
        let mlp_2 = vb.get((hidden_size, dim), "mlp.2")?;
        let mlp = [
            Linear::from_arc(mlp_0, None)?,
            Linear::from_arc(mlp_2, None)?,
        ];

        Ok(Self {
            hidden_size,
            ln_q,
            mlp,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.unsqueeze(0)?
            .apply(&self.ln_q)?
            .reshape(((), self.hidden_size))?
            .apply(&self.mlp[0])?
            .gelu()?
            .apply(&self.mlp[1])?
            .squeeze(0)
    }
}
