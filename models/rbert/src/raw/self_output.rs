use fusor_core::{Device, VarBuilder};
use fusor_core::{Result, Tensor};

use crate::raw::layer_norm::{layer_norm, LayerNorm};
use crate::raw::linear::Linear;

pub(crate) struct BertSelfOutput {
    dense: Linear,
    layer_norm: LayerNorm,
    span: tracing::Span,
}

impl BertSelfOutput {
    pub(crate) fn load(
        device: &Device,
        vb: &mut VarBuilder,
        config: &super::Config,
    ) -> Result<Self> {
        let dense = Linear::load(device, &mut vb.pp("dense"))?;
        let layer_norm = layer_norm(device, &mut vb.pp("LayerNorm"), config.layer_norm_eps as _)?;
        Ok(Self {
            dense,
            layer_norm,
            span: tracing::span!(tracing::Level::TRACE, "self-out"),
        })
    }

    pub(crate) fn forward(
        &self,
        hidden_states: &Tensor<2, f32>,
        input_tensor: &Tensor<2, f32>,
    ) -> Tensor<2, f32> {
        let _enter = self.span.enter();
        let hidden_states = self.dense.forward(hidden_states);
        self.layer_norm.forward(&(&hidden_states + input_tensor))
    }
}
