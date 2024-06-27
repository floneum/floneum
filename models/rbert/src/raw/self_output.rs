use candle_core::{Result, Tensor};
use candle_nn::{Dropout, Module, ModuleT, VarBuilder};
use candle_transformers::models::with_tracing::{layer_norm, linear, LayerNorm, Linear};

pub(crate) struct BertSelfOutput {
    dense: Linear,
    layer_norm: LayerNorm,
    dropout: Dropout,
    span: tracing::Span,
}

impl BertSelfOutput {
    pub(crate) fn load(vb: VarBuilder, config: &super::Config) -> Result<Self> {
        let dense = linear(config.hidden_size, config.hidden_size, vb.pp("dense"))?;
        let layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;
        let dropout = Dropout::new(config.hidden_dropout_prob);
        Ok(Self {
            dense,
            layer_norm,
            dropout,
            span: tracing::span!(tracing::Level::TRACE, "self-out"),
        })
    }

    pub(crate) fn forward(
        &self,
        hidden_states: &Tensor,
        input_tensor: &Tensor,
        train: bool,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = self.dropout.forward_t(&hidden_states, train)?;
        self.layer_norm.forward(&(hidden_states + input_tensor)?)
    }
}
