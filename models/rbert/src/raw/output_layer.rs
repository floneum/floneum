use fusor_core::{Device, VarBuilder};
use fusor_core::{Result, Tensor};

use crate::raw::layer_norm::{layer_norm, LayerNorm};
use crate::raw::linear::Linear;

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L456
pub(crate) struct BertOutput {
    dense: Linear,
    layer_norm: LayerNorm<1>,
    span: tracing::Span,
}

impl BertOutput {
    pub(crate) fn load(
        device: &Device,
        vb: &mut VarBuilder,
        config: &super::Config,
    ) -> Result<Self> {
        let dense = Linear::load(device, &mut vb.pp("ffn_down"))?;
        let layer_norm = layer_norm(
            device,
            &mut vb.pp("layer_output_norm"),
            config.layer_norm_eps as _,
        )?;
        Ok(Self {
            dense,
            layer_norm,
            span: tracing::span!(tracing::Level::TRACE, "out"),
        })
    }

    pub(crate) fn forward(
        &self,
        hidden_states: &Tensor<3, f32>,
        input_tensor: &Tensor<3, f32>,
    ) -> Tensor<3, f32> {
        let _enter = self.span.enter();
        let hidden_states = self.dense.forward(hidden_states);
        self.layer_norm.forward(&(&hidden_states + input_tensor))
    }
}
