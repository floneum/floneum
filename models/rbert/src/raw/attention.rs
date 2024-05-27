use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{embedding, Dropout, Embedding, Module, ModuleT, VarBuilder};
use candle_transformers::models::with_tracing::{layer_norm, linear, LayerNorm, Linear};
use serde::Deserialize;

use super::{BertSelfAttention, BertSelfOutput};

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L392
pub(crate) struct BertAttention {
    self_attention: BertSelfAttention,
    self_output: BertSelfOutput,
    span: tracing::Span,
}

impl BertAttention {
    pub(crate) fn load(vb: VarBuilder, config: &super::Config) -> Result<Self> {
        let self_attention = BertSelfAttention::load(vb.pp("self"), config)?;
        let self_output = BertSelfOutput::load(vb.pp("output"), config)?;
        Ok(Self {
            self_attention,
            self_output,
            span: tracing::span!(tracing::Level::TRACE, "attn"),
        })
    }

    pub(crate) fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        train: bool,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let self_outputs = self
            .self_attention
            .forward(hidden_states, attention_mask, train)?;
        let attention_output = self
            .self_output
            .forward(&self_outputs, hidden_states, train)?;
        Ok(attention_output)
    }
}
