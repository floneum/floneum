use candle_core::{DType, Result, Tensor};
use candle_nn::{Dropout, Module, ModuleT, VarBuilder};
use candle_transformers::models::with_tracing::{linear, Linear};

pub(crate) struct BertSelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    dropout: Dropout,
    num_attention_heads: usize,
    attention_head_size: usize,
    span: tracing::Span,
    span_softmax: tracing::Span,
}

impl BertSelfAttention {
    pub(crate) fn load(vb: VarBuilder, config: &super::Config) -> Result<Self> {
        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let all_head_size = config.num_attention_heads * attention_head_size;
        let dropout = Dropout::new(config.hidden_dropout_prob);
        let hidden_size = config.hidden_size;
        let query = linear(hidden_size, all_head_size, vb.pp("query"))?;
        let value = linear(hidden_size, all_head_size, vb.pp("value"))?;
        let key = linear(hidden_size, all_head_size, vb.pp("key"))?;
        Ok(Self {
            query,
            key,
            value,
            dropout,
            num_attention_heads: config.num_attention_heads,
            attention_head_size,
            span: tracing::span!(tracing::Level::TRACE, "self-attn"),
            span_softmax: tracing::span!(tracing::Level::TRACE, "softmax"),
        })
    }

    pub(crate) fn transpose_for_scores(&self, xs: &Tensor) -> Result<Tensor> {
        let mut new_x_shape = xs.dims().to_vec();
        new_x_shape.pop();
        new_x_shape.push(self.num_attention_heads);
        new_x_shape.push(self.attention_head_size);
        let xs = xs.reshape(new_x_shape.as_slice())?.transpose(1, 2)?;
        xs.contiguous()
    }

    pub(crate) fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        train: bool,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let query_layer = self.query.forward(hidden_states)?;
        let key_layer = self.key.forward(hidden_states)?;
        let value_layer = self.value.forward(hidden_states)?;

        let query_layer = self.transpose_for_scores(&query_layer)?;
        let key_layer = self.transpose_for_scores(&key_layer)?;
        let value_layer = self.transpose_for_scores(&value_layer)?;

        let attention_scores = query_layer.matmul(&key_layer.t()?)?;
        let mut attention_scores = (attention_scores / (self.attention_head_size as f64).sqrt())?;
        // If there is an attention mask, filter the attention scores by that mask
        if let Some(attention_mask) = attention_mask {
            // The attention mask is a tensor of shape (bsize, seq_len)
            // the attention scores are a tensor of shape (bsize, _, seq_len, seq_len)
            // We expand the attention mask to (bsize, 1, 1, seq_len)
            let mask = attention_mask.unsqueeze(1)?.unsqueeze(2)?;
            let shape = attention_scores.shape();
            let mask = mask.broadcast_as(shape)?.to_dtype(DType::U8)?;
            // We use a value slightly larger that the true f32 min value to avoid NaN
            const FALSE_MIN: f32 = -3.4028235e34f32;
            let on_false = Tensor::new(FALSE_MIN, mask.device())?.broadcast_as(shape)?;
            attention_scores = mask.where_cond(&attention_scores, &on_false)?;
        }

        let attention_probs = {
            let _enter_sm = self.span_softmax.enter();
            candle_nn::ops::softmax(&attention_scores, candle_core::D::Minus1)?
        };
        let attention_probs = self.dropout.forward_t(&attention_probs, train)?;
        let context_layer = attention_probs.matmul(&value_layer)?;
        let context_layer = context_layer.transpose(1, 2)?.contiguous()?;
        let context_layer = context_layer.flatten_from(candle_core::D::Minus2)?;
        Ok(context_layer)
    }
}
