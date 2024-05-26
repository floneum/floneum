use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{embedding, Dropout, Embedding, Module, ModuleT, VarBuilder};
use candle_transformers::models::with_tracing::{layer_norm, linear, LayerNorm, Linear};
use serde::Deserialize;

use super::{BertAttention, BertIntermediate, BertOutput};

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L470
pub(crate) struct BertLayer {
    attention: BertAttention,
    intermediate: BertIntermediate,
    output: BertOutput,
    span: tracing::Span,
}

impl BertLayer {
    pub(crate) fn load(vb: VarBuilder, config: &super::Config) -> Result<Self> {
        let attention = BertAttention::load(vb.pp("attention"), config)?;
        let intermediate = BertIntermediate::load(vb.pp("intermediate"), config)?;
        let output = BertOutput::load(vb.pp("output"), config)?;
        Ok(Self {
            attention,
            intermediate,
            output,
            span: tracing::span!(tracing::Level::TRACE, "layer"),
        })
    }

    pub(crate) fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        train: bool,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let attention_output = self
            .attention
            .forward(hidden_states, attention_mask, train)?;
        let intermediate_output = self.intermediate.forward(&attention_output)?;
        let layer_output = self
            .output
            .forward(&intermediate_output, &attention_output, train)?;
        Ok(layer_output)
    }
}
