use super::assert_all_f32_not_nan;
use super::rope::RopeCache;
use super::silu::fast_cpu_silu;
use candle_core::{quantized::QMatMul, Module, Tensor};
use candle_core::{Device, D};
use candle_transformers::quantized_nn::RmsNorm;
use kalosm_common::AttentionMask;
use kalosm_common::KvCache;

pub enum FeedForwardVariant {
    Llama(LlamaFeedForward),
    Phi(PhiFeedForward),
}

impl FeedForwardVariant {
    pub(crate) fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            FeedForwardVariant::Llama(ffn) => ffn.forward(x),
            FeedForwardVariant::Phi(ffn) => ffn.forward(x),
        }
    }
}

pub struct PhiFeedForward {
    pub up: QMatMul,
    pub down: QMatMul,
    pub feed_forward_length: usize,
}

impl PhiFeedForward {
    pub(crate) fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let up_states = x.apply(&self.up)?;
        let gate = up_states.narrow(D::Minus1, 0, self.feed_forward_length)?;
        let up_states = up_states.narrow(
            D::Minus1,
            self.feed_forward_length,
            self.feed_forward_length,
        )?;
        let gate = fast_cpu_silu(&gate)?;
        let up_states = (up_states * gate)?;
        up_states.apply(&self.down)
    }
}

pub struct LlamaFeedForward {
    pub feed_forward_w1: QMatMul,
    pub feed_forward_w2: QMatMul,
    pub feed_forward_w3: QMatMul,
}

impl LlamaFeedForward {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let device = x.device();
        if matches!(device, Device::Cpu) {
            std::thread::scope(|scope| {
                let w1 = scope.spawn(|| {
                    let w1 = self.feed_forward_w1.forward(x)?;
                    fast_cpu_silu(&w1)
                });

                let w3 = self.feed_forward_w3.forward(x)?;
                let w1 = w1
                    .join()
                    .map_err(|_| candle_core::Error::Msg("Failed to join thread".to_string()))??;

                self.feed_forward_w2.forward(&(&w1 * w3)?)
            })
        } else {
            let w1 = self.feed_forward_w1.forward(x)?;
            let w1 = fast_cpu_silu(&w1)?;

            let w3 = self.feed_forward_w3.forward(x)?;

            self.feed_forward_w2.forward(&(&w1 * w3)?)
        }
    }
}

pub enum AttentionVariant {
    Separate(SeparateAttention),
    Grouped(GroupedAttention),
}

pub struct AttentionBias {
    pub bias_q: Tensor,
    pub bias_k: Tensor,
    pub bias_v: Tensor,
}

pub struct SeparateAttention {
    pub attention_wq: QMatMul,
    pub attention_wk: QMatMul,
    pub attention_wv: QMatMul,
    pub bias: Option<AttentionBias>,
    pub interleaved_rope: bool,
}

impl SeparateAttention {
    fn forward(
        &self,
        num_heads: usize,
        head_dim: usize,
        num_key_value_heads: usize,
        hidden_states: &Tensor,
        rope_cache: &RopeCache,
        start_pos: usize,
    ) -> candle_core::Result<(Tensor, Tensor, Tensor)> {
        let b_sz = hidden_states.dims()[0];
        let seq_len = hidden_states.dims()[1];
        let device = hidden_states.device();

        if matches!(device, Device::Cpu) {
            std::thread::scope(|s| -> Result<_, candle_core::Error> {
                let query_states = s.spawn(|| {
                    let mut query_states = self.attention_wq.forward(hidden_states)?;

                    if let Some(bias) = &self.bias {
                        query_states = query_states.broadcast_add(&bias.bias_q)?;
                    }

                    query_states
                        .reshape((b_sz, seq_len, num_heads, head_dim))?
                        .transpose(1, 2)
                });
                let key_states = s.spawn(|| {
                    let mut key_states = self.attention_wk.forward(hidden_states)?;

                    if let Some(bias) = &self.bias {
                        key_states = key_states.broadcast_add(&bias.bias_k)?;
                    }

                    key_states
                        .reshape((b_sz, seq_len, num_key_value_heads, head_dim))?
                        .transpose(1, 2)
                });
                let value_states = s.spawn(|| {
                    let mut value_states = self.attention_wv.forward(hidden_states)?;

                    if let Some(bias) = &self.bias {
                        value_states = value_states.broadcast_add(&bias.bias_v)?;
                    }

                    value_states
                        .reshape((b_sz, seq_len, num_key_value_heads, head_dim))?
                        .transpose(1, 2)
                });

                let query_states = query_states.join().map_err(|_| {
                    candle_core::Error::Msg("failed to join query states".to_string())
                })??;
                let key_states = key_states.join().map_err(|_| {
                    candle_core::Error::Msg("failed to join key states".to_string())
                })??;

                let (query_states, key_states) = if self.interleaved_rope {
                    rope_cache.forward_i(&query_states, &key_states, start_pos)?
                } else {
                    rope_cache.forward(&query_states, &key_states, start_pos)?
                };

                let value_states = value_states.join().map_err(|_| {
                    candle_core::Error::Msg("failed to join value states".to_string())
                })??;

                Ok((query_states, key_states, value_states))
            })
        } else {
            let query_states = {
                let mut query_states = self.attention_wq.forward(hidden_states)?;

                if let Some(bias) = &self.bias {
                    query_states = query_states.broadcast_add(&bias.bias_q)?;
                }

                query_states
                    .reshape((b_sz, seq_len, num_heads, head_dim))?
                    .transpose(1, 2)?
            };
            let key_states = {
                let mut key_states = self.attention_wk.forward(hidden_states)?;

                if let Some(bias) = &self.bias {
                    key_states = key_states.broadcast_add(&bias.bias_k)?;
                }

                key_states
                    .reshape((b_sz, seq_len, num_key_value_heads, head_dim))?
                    .transpose(1, 2)?
            };
            let value_states = {
                let mut value_states = self.attention_wv.forward(hidden_states)?;

                if let Some(bias) = &self.bias {
                    value_states = value_states.broadcast_add(&bias.bias_v)?;
                }

                value_states
                    .reshape((b_sz, seq_len, num_key_value_heads, head_dim))?
                    .transpose(1, 2)?
            };

            let (query_states, key_states) = if self.interleaved_rope {
                rope_cache.forward_i(&query_states, &key_states, start_pos)?
            } else {
                rope_cache.forward(&query_states, &key_states, start_pos)?
            };

            Ok((query_states, key_states, value_states))
        }
    }
}

pub struct GroupedAttention {
    pub attention_qkv: QMatMul,
}

impl GroupedAttention {
    fn forward(
        &self,
        num_heads: usize,
        head_dim: usize,
        num_key_value_heads: usize,
        x: &Tensor,
        rope_cache: &RopeCache,
        start_pos: usize,
    ) -> candle_core::Result<(Tensor, Tensor, Tensor)> {
        let b_sz = x.dims()[0];
        let seq_len = x.dims()[1];
        let qkv = self.attention_qkv.forward(x)?;

        let query_pos = num_heads * head_dim;
        let query_states = qkv.narrow(D::Minus1, 0, query_pos)?;
        let key_states = qkv.narrow(D::Minus1, query_pos, num_key_value_heads * head_dim)?;
        let value_states = qkv.narrow(
            D::Minus1,
            query_pos + num_key_value_heads * head_dim,
            num_key_value_heads * head_dim,
        )?;

        let query_states = query_states
            .reshape((b_sz, seq_len, num_heads, head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((b_sz, seq_len, num_key_value_heads, head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((b_sz, seq_len, num_key_value_heads, head_dim))?
            .transpose(1, 2)?;

        let (query_states, key_states) =
            rope_cache.forward(&query_states, &key_states, start_pos)?;

        Ok((query_states, key_states, value_states))
    }
}

pub struct LlamaAttention {
    pub attention_variant: AttentionVariant,
    pub attention_wo: QMatMul,
    pub attention_norm: RmsNorm,
    pub feed_forward_variant: FeedForwardVariant,
    pub ffn_norm: RmsNorm,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    pub hidden_size: usize,
    pub rope_cache: RopeCache,
}

impl LlamaAttention {
    pub(crate) fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&AttentionMask>,
        start_pos: usize,
        cache: Option<&mut KvCache>,
    ) -> candle_core::Result<Tensor> {
        let bsz = hidden_states.dims()[0];
        let q_len = hidden_states.dims()[1];
        let hidden_size = self.hidden_size;
        let num_heads = self.n_head;
        let head_dim = self.head_dim;
        let num_key_value_heads = self.n_kv_head;
        let num_key_value_groups = num_heads / num_key_value_heads;

        let (query_states, key_states, value_states) = match self.attention_variant {
            AttentionVariant::Separate(ref attention) => attention.forward(
                num_heads,
                head_dim,
                num_key_value_heads,
                hidden_states,
                &self.rope_cache,
                start_pos,
            )?,
            AttentionVariant::Grouped(ref attention) => attention.forward(
                num_heads,
                head_dim,
                num_key_value_heads,
                hidden_states,
                &self.rope_cache,
                start_pos,
            )?,
        };
        assert_all_f32_not_nan(&query_states);
        assert_all_f32_not_nan(&key_states);
        assert_all_f32_not_nan(&value_states);

        let key_states = repeat_kv(key_states.clone(), num_key_value_groups)?;
        let value_states = repeat_kv(value_states, num_key_value_groups)?;

        let (key_states, value_states) = match cache {
            None => (key_states, value_states),
            Some(cache) => cache.append(&key_states, &value_states)?,
        };

        let scale = 1. / (head_dim as f64).sqrt();

        let mut attn_output = {
            let mut attn_weights = (query_states.matmul(&key_states.t()?)? * scale)?;

            if let Some(attention_mask) = attention_mask {
                attention_mask.forward(&mut attn_weights)?;
            }

            attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            assert_all_f32_not_nan(&attn_weights);

            attn_weights.matmul(&value_states)?
        };
        assert_all_f32_not_nan(&attn_output);

        if attn_output.dims() != [bsz, num_heads, q_len, head_dim] {
            return Err(candle_core::Error::Msg(format!(
                "`attn_output` should be of size {:?}, but is {:?}",
                [bsz, self.n_head, q_len, head_dim],
                attn_output.dims()
            )));
        }

        attn_output = attn_output.transpose(1, 2).unwrap();

        attn_output = attn_output.reshape(&[bsz, q_len, hidden_size]).unwrap();

        attn_output = self.attention_wo.forward(&attn_output).unwrap();

        Ok(attn_output)
    }
}

fn repeat_kv(x: Tensor, num_key_value_groups: usize) -> candle_core::Result<Tensor> {
    if num_key_value_groups == 1 {
        Ok(x)
    } else {
        let (b_sz, n_kv_head, seq_len, head_dim) = x.dims4()?;
        Tensor::cat(&vec![&x; num_key_value_groups], 2)?.reshape((
            b_sz,
            n_kv_head * num_key_value_groups,
            seq_len,
            head_dim,
        ))
    }
}
