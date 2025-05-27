use super::debug_assert_none_nan;
use super::rope::RopeImplementation;
use super::silu::fast_cpu_silu;
use candle_core::quantized::QTensor;
use candle_core::{quantized::QMatMul, Module, Tensor};
use candle_core::{Device, D};
use candle_transformers::quantized_nn::{Linear, RmsNorm};
use kalosm_common::AttentionMask;
use kalosm_common::KvCache;

pub enum FeedForwardVariant {
    // Used by the Llama, Qwen, and Gemma models
    Llama(LlamaFeedForward),
    // Used by the Phi models
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
        let up_states = x.apply(&self.up).unwrap();
        let gate = up_states
            .narrow(D::Minus1, 0, self.feed_forward_length)
            .unwrap();
        let up_states = up_states
            .narrow(
                D::Minus1,
                self.feed_forward_length,
                self.feed_forward_length,
            )
            .unwrap();
        let gate = fast_cpu_silu(&gate).unwrap();
        let up_states = (up_states * gate).unwrap();
        up_states.apply(&self.down)
    }
}

pub struct LlamaFeedForward {
    gate: QMatMul,
    gate_bias: Option<Tensor>,
    down: QMatMul,
    down_bias: Option<Tensor>,
    up: QMatMul,
    up_bias: Option<Tensor>,
}

impl LlamaFeedForward {
    pub(crate) fn new(gate: QMatMul, down: QMatMul, up: QMatMul) -> Self {
        Self {
            gate,
            down,
            up,
            gate_bias: None,
            down_bias: None,
            up_bias: None,
        }
    }

    pub(crate) fn new_with_bias(
        gate: QMatMul,
        gate_bias: Option<Tensor>,
        down: QMatMul,
        down_bias: Option<Tensor>,
        up: QMatMul,
        up_bias: Option<Tensor>,
    ) -> Self {
        Self {
            gate,
            gate_bias,
            down,
            down_bias,
            up,
            up_bias,
        }
    }

    pub(crate) fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let device = x.device();
        if matches!(device, Device::Cpu) {
            std::thread::scope(|scope| {
                let w1 = scope.spawn(|| {
                    let mut w1 = self.gate.forward(x).unwrap();
                    if let Some(ref bias) = self.gate_bias {
                        w1 = w1.broadcast_add(bias).unwrap();
                    }
                    fast_cpu_silu(&w1)
                });

                let mut w3 = self.up.forward(x).unwrap();
                if let Some(ref bias) = self.up_bias {
                    w3 = w3.broadcast_add(bias).unwrap();
                }
                let w1 = w1
                    .join()
                    .map_err(|_| candle_core::Error::Msg("Failed to join thread".to_string()))
                    .unwrap()
                    .unwrap();

                let mut up = self.down.forward(&(&w1 * w3).unwrap()).unwrap();

                if let Some(ref bias) = self.down_bias {
                    up = up.broadcast_add(bias).unwrap();
                }

                Ok(up)
            })
        } else {
            let mut w1 = self.gate.forward(x).unwrap();
            if let Some(ref bias) = self.gate_bias {
                w1 = w1.broadcast_add(bias).unwrap();
            }
            let w1 = fast_cpu_silu(&w1).unwrap();

            let mut w3 = self.up.forward(x).unwrap();
            if let Some(ref bias) = self.up_bias {
                w3 = w3.broadcast_add(bias).unwrap();
            }

            let mut up = self.down.forward(&(&w1 * w3).unwrap()).unwrap();
            if let Some(ref bias) = self.down_bias {
                up = up.broadcast_add(bias).unwrap();
            }
            Ok(up)
        }
    }
}

pub enum AttentionVariant {
    Separate(SeparateAttention),
    Grouped(GroupedAttention),
}

pub struct AttentionBias {
    bias_q: Tensor,
    bias_k: Tensor,
    bias_v: Tensor,
}

impl AttentionBias {
    pub fn new(q: Tensor, k: Tensor, v: Tensor) -> Self {
        Self {
            bias_q: q,
            bias_k: k,
            bias_v: v,
        }
    }

    pub fn from_qtensor(q: &QTensor, k: &QTensor, v: &QTensor) -> candle_core::Result<Self> {
        Ok(Self::new(
            q.dequantize(&q.device()).unwrap(),
            k.dequantize(&k.device()).unwrap(),
            v.dequantize(&v.device()).unwrap(),
        ))
    }
}

pub struct SeparateAttention {
    pub attention_wq: QMatMul,
    pub attention_q_norm: Option<RmsNorm>,
    pub attention_wk: QMatMul,
    pub attention_k_norm: Option<RmsNorm>,
    pub attention_wv: QMatMul,
    pub bias: Option<AttentionBias>,
    pub interleaved_rope: bool,
}

impl SeparateAttention {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        num_heads: usize,
        head_dim: usize,
        num_key_value_heads: usize,
        hidden_states: &Tensor,
        rope_cache: &RopeImplementation,
        start_pos: usize,
        pos_ids: Option<&Tensor>,
    ) -> candle_core::Result<(Tensor, Tensor, Tensor)> {
        let b_sz = hidden_states.dims()[0];
        let seq_len = hidden_states.dims()[1];
        let device = hidden_states.device();

        if matches!(device, Device::Cpu) {
            std::thread::scope(|s| -> Result<_, candle_core::Error> {
                let query_states: std::thread::ScopedJoinHandle<'_, candle_core::Result<Tensor>> =
                    s.spawn(|| {
                        let mut query_states = self.attention_wq.forward(hidden_states).unwrap();

                        if let Some(bias) = &self.bias {
                            query_states = query_states.broadcast_add(&bias.bias_q).unwrap();
                        }

                        let mut query = query_states
                            .reshape((b_sz, seq_len, num_heads, head_dim))
                            .unwrap()
                            .transpose(1, 2)
                            .unwrap();

                        if let Some(norm) = &self.attention_q_norm {
                            query = norm.forward(&query.contiguous().unwrap()).unwrap();
                        }

                        Ok(query)
                    });
                let key_states: std::thread::ScopedJoinHandle<'_, candle_core::Result<Tensor>> = s
                    .spawn(|| {
                        let mut key_states = self.attention_wk.forward(hidden_states).unwrap();

                        if let Some(bias) = &self.bias {
                            key_states = key_states.broadcast_add(&bias.bias_k).unwrap();
                        }

                        let mut key = key_states
                            .reshape((b_sz, seq_len, num_key_value_heads, head_dim))
                            .unwrap()
                            .transpose(1, 2)
                            .unwrap();

                        if let Some(norm) = &self.attention_k_norm {
                            key = norm.forward(&key.contiguous().unwrap()).unwrap();
                        }

                        Ok(key)
                    });
                let value_states = s.spawn(|| {
                    let mut value_states = self.attention_wv.forward(hidden_states).unwrap();

                    if let Some(bias) = &self.bias {
                        value_states = value_states.broadcast_add(&bias.bias_v).unwrap();
                    }

                    value_states
                        .reshape((b_sz, seq_len, num_key_value_heads, head_dim))
                        .unwrap()
                        .transpose(1, 2)
                });

                let query_states = query_states
                    .join()
                    .map_err(|_| candle_core::Error::Msg("failed to join query states".to_string()))
                    .unwrap()
                    .unwrap();
                let key_states = key_states
                    .join()
                    .map_err(|_| candle_core::Error::Msg("failed to join key states".to_string()))
                    .unwrap()
                    .unwrap();

                let (query_states, key_states) = rope_cache
                    .forward(
                        &query_states,
                        &key_states,
                        start_pos,
                        pos_ids,
                        self.interleaved_rope,
                    )
                    .unwrap();

                let value_states = value_states
                    .join()
                    .map_err(|_| candle_core::Error::Msg("failed to join value states".to_string()))
                    .unwrap()
                    .unwrap();

                Ok((query_states, key_states, value_states))
            })
        } else {
            let query_states = {
                let mut query_states = self.attention_wq.forward(hidden_states).unwrap();

                if let Some(bias) = &self.bias {
                    query_states = query_states.broadcast_add(&bias.bias_q).unwrap();
                }

                let mut query = query_states
                    .reshape((b_sz, seq_len, num_heads, head_dim))
                    .unwrap()
                    .transpose(1, 2)
                    .unwrap();

                if let Some(norm) = &self.attention_q_norm {
                    query = norm.forward(&query.contiguous().unwrap()).unwrap();
                }

                query
            };
            let key_states = {
                let mut key_states = self.attention_wk.forward(hidden_states).unwrap();

                if let Some(bias) = &self.bias {
                    key_states = key_states.broadcast_add(&bias.bias_k).unwrap();
                }

                let mut key = key_states
                    .reshape((b_sz, seq_len, num_key_value_heads, head_dim))
                    .unwrap()
                    .transpose(1, 2)
                    .unwrap();

                if let Some(norm) = &self.attention_k_norm {
                    key = norm.forward(&key.contiguous().unwrap()).unwrap();
                }

                key
            };
            let value_states = {
                let mut value_states = self.attention_wv.forward(hidden_states).unwrap();

                if let Some(bias) = &self.bias {
                    value_states = value_states.broadcast_add(&bias.bias_v).unwrap();
                }

                value_states
                    .reshape((b_sz, seq_len, num_key_value_heads, head_dim))
                    .unwrap()
                    .transpose(1, 2)
                    .unwrap()
            };

            let (query_states, key_states) = rope_cache
                .forward(
                    &query_states,
                    &key_states,
                    start_pos,
                    pos_ids,
                    self.interleaved_rope,
                )
                .unwrap();

            Ok((query_states, key_states, value_states))
        }
    }
}

pub struct GroupedAttention {
    pub attention_qkv: QMatMul,
}

impl GroupedAttention {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        num_heads: usize,
        head_dim: usize,
        num_key_value_heads: usize,
        x: &Tensor,
        rope_cache: &RopeImplementation,
        start_pos: usize,
        pos_ids: Option<&Tensor>,
    ) -> candle_core::Result<(Tensor, Tensor, Tensor)> {
        let b_sz = x.dims()[0];
        let seq_len = x.dims()[1];
        let qkv = self.attention_qkv.forward(x).unwrap();

        let query_pos = num_heads * head_dim;
        let query_states = qkv.narrow(D::Minus1, 0, query_pos).unwrap();
        let key_states = qkv
            .narrow(D::Minus1, query_pos, num_key_value_heads * head_dim)
            .unwrap();
        let value_states = qkv
            .narrow(
                D::Minus1,
                query_pos + num_key_value_heads * head_dim,
                num_key_value_heads * head_dim,
            )
            .unwrap();

        let query_states = query_states
            .reshape((b_sz, seq_len, num_heads, head_dim))
            .unwrap()
            .transpose(1, 2)
            .unwrap();
        let key_states = key_states
            .reshape((b_sz, seq_len, num_key_value_heads, head_dim))
            .unwrap()
            .transpose(1, 2)
            .unwrap();
        let value_states = value_states
            .reshape((b_sz, seq_len, num_key_value_heads, head_dim))
            .unwrap()
            .transpose(1, 2)
            .unwrap();

        let (query_states, key_states) = rope_cache
            .forward(&query_states, &key_states, start_pos, pos_ids, false)
            .unwrap();

        Ok((query_states, key_states, value_states))
    }
}

pub struct LlamaAttention {
    pub attention_variant: AttentionVariant,
    pub attention_wo: Linear,
    pub attention_norm: RmsNorm,
    pub post_attention_norm: Option<RmsNorm>,
    pub feed_forward_variant: FeedForwardVariant,
    pub ffn_norm: RmsNorm,
    pub post_ffn_norm: Option<RmsNorm>,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    pub hidden_size: usize,
    pub rope_cache: RopeImplementation,
    pub(crate) sliding_window_size: Option<usize>,
}

impl LlamaAttention {
    pub(crate) fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&AttentionMask>,
        start_pos: usize,
        pos_ids: Option<&Tensor>,
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
            AttentionVariant::Separate(ref attention) => attention
                .forward(
                    num_heads,
                    head_dim,
                    num_key_value_heads,
                    hidden_states,
                    &self.rope_cache,
                    start_pos,
                    pos_ids,
                )
                .unwrap(),
            AttentionVariant::Grouped(ref attention) => attention
                .forward(
                    num_heads,
                    head_dim,
                    num_key_value_heads,
                    hidden_states,
                    &self.rope_cache,
                    start_pos,
                    pos_ids,
                )
                .unwrap(),
        };
        debug_assert_none_nan(&query_states);
        debug_assert_none_nan(&key_states);
        debug_assert_none_nan(&value_states);

        let key_states = repeat_kv(key_states.clone(), num_key_value_groups).unwrap();
        let value_states = repeat_kv(value_states, num_key_value_groups).unwrap();

        let (key_states, value_states) = match cache {
            None => (key_states, value_states),
            Some(cache) => cache.append(&key_states, &value_states).unwrap(),
        };

        forward_attention_qkv(
            &query_states,
            &key_states,
            &value_states,
            &self.attention_wo,
            attention_mask,
            num_heads,
            head_dim,
            bsz,
            q_len,
            hidden_size,
        )
    }
}

fn repeat_kv(x: Tensor, num_key_value_groups: usize) -> candle_core::Result<Tensor> {
    if num_key_value_groups == 1 {
        Ok(x)
    } else {
        let (b_sz, n_kv_head, seq_len, head_dim) = x.dims4().unwrap();
        Tensor::cat(&vec![&x; num_key_value_groups], 2)
            .unwrap()
            .reshape((b_sz, n_kv_head * num_key_value_groups, seq_len, head_dim))
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn forward_attention_qkv(
    query_states: &Tensor,
    key_states: &Tensor,
    value_states: &Tensor,
    attention_wo: &Linear,
    attention_mask: Option<&AttentionMask>,
    num_heads: usize,
    head_dim: usize,
    bsz: usize,
    q_len: usize,
    hidden_size: usize,
) -> candle_core::Result<Tensor> {
    let scale = 1. / (head_dim as f64).sqrt();
    let mut attn_output = {
        let mut attn_weights =
            (query_states.matmul(&key_states.t().unwrap()).unwrap() * scale).unwrap();
        debug_assert_none_nan(&attn_weights);

        if let Some(attention_mask) = attention_mask {
            attention_mask.forward(&mut attn_weights).unwrap();
            debug_assert_none_nan(&attn_weights);
        }

        attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights).unwrap();
        debug_assert_none_nan(&attn_weights);

        attn_weights.matmul(value_states).unwrap()
    };

    debug_assert_none_nan(&attn_output);

    if attn_output.dims() != [bsz, num_heads, q_len, head_dim] {
        return Err(candle_core::Error::Msg(format!(
            "`attn_output` should be of size {:?}, but is {:?}",
            [bsz, num_heads, q_len, head_dim],
            attn_output.dims()
        )));
    }

    attn_output = attn_output.transpose(1, 2).unwrap();

    attn_output = attn_output.reshape(&[bsz, q_len, hidden_size]).unwrap();

    attn_output = attention_wo.forward(&attn_output).unwrap();

    debug_assert_none_nan(&attn_output);

    Ok(attn_output)
}
