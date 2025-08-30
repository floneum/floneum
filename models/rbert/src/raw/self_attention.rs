use fusor_core::{Device, VarBuilder};
use fusor_core::{Result, Tensor};

use crate::raw::linear::Linear;

pub(crate) struct BertSelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    num_attention_heads: usize,
    attention_head_size: usize,
    span: tracing::Span,
    span_softmax: tracing::Span,
}

impl BertSelfAttention {
    pub(crate) fn load(
        device: &Device,
        vb: &mut VarBuilder,
        config: &super::Config,
    ) -> Result<Self> {
        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let query = Linear::load(device, &mut vb.pp("query"))?;
        let value = Linear::load(device, &mut vb.pp("value"))?;
        let key = Linear::load(device, &mut vb.pp("key"))?;
        Ok(Self {
            query,
            key,
            value,
            num_attention_heads: config.num_attention_heads,
            attention_head_size,
            span: tracing::span!(tracing::Level::TRACE, "self-attn"),
            span_softmax: tracing::span!(tracing::Level::TRACE, "softmax"),
        })
    }

    pub(crate) fn transpose_for_scores(&self, xs: &Tensor<2, f32>) -> Tensor<3, f32> {
        let new_x_shape = [
            xs.shape()[0],
            self.num_attention_heads,
            self.attention_head_size,
        ];
        xs.reshape(new_x_shape).transpose(1, 2)
    }

    pub(crate) fn forward(
        &self,
        hidden_states: &Tensor<2, f32>,
        attention_mask: Option<&Tensor<2, u32>>,
    ) -> Tensor<2, f32> {
        let _enter = self.span.enter();
        let query_layer = self.query.forward(hidden_states);
        let key_layer = self.key.forward(hidden_states);
        let value_layer = self.value.forward(hidden_states);

        let query_layer = self.transpose_for_scores(&query_layer);
        let key_layer = self.transpose_for_scores(&key_layer);
        let value_layer = self.transpose_for_scores(&value_layer);

        let attention_scores = query_layer.mat_mul(&key_layer.t());
        let mut attention_scores = attention_scores / (self.attention_head_size as f32).sqrt();
        // If there is an attention mask, filter the attention scores by that mask
        if let Some(attention_mask) = attention_mask {
            // The attention mask is a tensor of shape (bsize, seq_len)
            // the attention scores are a tensor of shape (bsize, heads, seq_len) or similar
            // We expand the attention mask to match the shape
            let shape = *attention_scores.shape();
            let mask = if shape.len() == 3 {
                attention_mask.unsqueeze(1).broadcast_as(shape).cast()
            } else {
                // For other dimensions, we need to create the proper mask
                // For a 2D attention_mask (batch, seq_len), we need to expand it correctly
                let expanded_mask = attention_mask.unsqueeze(1);
                let mut mask_shape = vec![shape[0], 1];
                mask_shape.extend(&shape[2..]);
                expanded_mask.broadcast_as(shape).cast()
            };
            // We use a value slightly larger that the true f32 min value to avoid NaN
            const FALSE_MIN: f32 = -3.4028235e34f32;
            let on_false = Tensor::splat(mask.device(), FALSE_MIN, shape);
            attention_scores = mask.where_cond(&attention_scores, &on_false);
        }

        let attention_probs = {
            let _enter_sm = self.span_softmax.enter();
            attention_scores.softmax(attention_scores.rank() - 1)
        };
        let context_layer = attention_probs.mat_mul(&value_layer);
        let context_layer = context_layer.transpose(1, 2);
        context_layer.flatten_last_n::<1, _>()
    }
}
