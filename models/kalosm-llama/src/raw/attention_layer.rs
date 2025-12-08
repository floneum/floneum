use crate::raw::rope::RopeImplementation;

use fusor_core::cache::AttentionMask;
use fusor_core::cache::KvCache;
use fusor_core::layers::Linear;
use fusor_core::layers::RmsNorm;
use fusor_core::QMatrix;
use fusor_core::Tensor;
use fusor_core::D;

pub enum FeedForwardVariant {
    // Used by the Llama, Qwen, and Gemma models
    Llama(LlamaFeedForward),
    // Used by the Phi models
    Phi(PhiFeedForward),
}

impl FeedForwardVariant {
    pub(crate) fn forward(&self, x: &Tensor<3, f32>) -> Tensor<3, f32> {
        match self {
            FeedForwardVariant::Llama(ffn) => ffn.forward(x),
            FeedForwardVariant::Phi(ffn) => ffn.forward(x),
        }
    }
}

pub struct PhiFeedForward {
    pub up: QMatrix,
    pub down: QMatrix,
    pub feed_forward_length: usize,
}

impl PhiFeedForward {
    pub(crate) fn forward(&self, x: &Tensor<3, f32>) -> Tensor<3, f32> {
        let up_states = x.q_mat_mul(&self.up);
        let gate = up_states.narrow(D::Minus1, 0, self.feed_forward_length);
        let up_states = up_states.narrow(
            D::Minus1,
            self.feed_forward_length,
            self.feed_forward_length,
        );
        let gate = gate.silu();
        let up_states = up_states * gate;
        up_states.q_mat_mul(&self.down)
    }
}

pub struct LlamaFeedForward {
    gate: QMatrix,
    gate_bias: Option<Tensor<1, f32>>,
    down: QMatrix,
    down_bias: Option<Tensor<1, f32>>,
    up: QMatrix,
    up_bias: Option<Tensor<1, f32>>,
}

impl LlamaFeedForward {
    pub(crate) fn new(gate: QMatrix, down: QMatrix, up: QMatrix) -> Self {
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
        gate: QMatrix,
        gate_bias: Option<Tensor<1, f32>>,
        down: QMatrix,
        down_bias: Option<Tensor<1, f32>>,
        up: QMatrix,
        up_bias: Option<Tensor<1, f32>>,
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

    fn forward(&self, x: &Tensor<3, f32>) -> Tensor<3, f32> {
        let mut w1 = x.q_mat_mul(&self.gate);
        if let Some(ref bias) = self.gate_bias {
            w1 = w1.add_(bias);
        }
        let w1 = w1.silu();

        let mut w3 = x.q_mat_mul(&self.up);
        if let Some(ref bias) = self.up_bias {
            w3 = w3.add_(bias);
        }

        let mut up = (w1 * w3).q_mat_mul(&self.down);
        if let Some(ref bias) = self.down_bias {
            up = up.add_(bias);
        }
        up
    }
}

pub enum AttentionVariant {
    Separate(SeparateAttention),
    Grouped(GroupedAttention),
}

pub struct AttentionBias {
    bias_q: Tensor<1, f32>,
    bias_k: Tensor<1, f32>,
    bias_v: Tensor<1, f32>,
}

impl AttentionBias {
    pub fn new(q: Tensor<1, f32>, k: Tensor<1, f32>, v: Tensor<1, f32>) -> Self {
        Self {
            bias_q: q,
            bias_k: k,
            bias_v: v,
        }
    }

    pub fn from_qtensor(q: &QMatrix, k: &QMatrix, v: &QMatrix) -> Self {
        Self::new(q.dequantize(), k.dequantize(), v.dequantize())
    }
}

pub struct SeparateAttention {
    pub attention_wq: QMatrix,
    pub attention_q_norm: Option<RmsNorm<1, f32>>,
    pub attention_wk: QMatrix,
    pub attention_k_norm: Option<RmsNorm<1, f32>>,
    pub attention_wv: QMatrix,
    pub bias: Option<AttentionBias>,
    pub interleaved_rope: bool,
}

impl SeparateAttention {
    fn forward(
        &self,
        num_heads: usize,
        head_dim: usize,
        num_key_value_heads: usize,
        hidden_states: &Tensor<3, f32>,
        rope_cache: &RopeImplementation,
        start_pos: usize,
        pos_ids: Option<&Tensor<2, f32>>,
    ) -> (Tensor<4, f32>, Tensor<4, f32>, Tensor<4, f32>) {
        let [b_sz, seq_len, _] = *hidden_states.shape();

        let query_states = {
            let mut query_states = hidden_states.q_mat_mul(&self.attention_wq);

            if let Some(bias) = &self.bias {
                query_states = query_states.add_(&bias.bias_q);
            }

            let mut query = query_states
                .reshape([b_sz, seq_len, num_heads, head_dim])
                .transpose(1, 2);
            if let Some(norm) = &self.attention_q_norm {
                query = norm.forward(&query);
            }
            query
        };
        let key_states = {
            let mut key_states = hidden_states.q_mat_mul(&self.attention_wk);

            if let Some(bias) = &self.bias {
                key_states = key_states.add_(&bias.bias_k);
            }

            let mut key = key_states
                .reshape([b_sz, seq_len, num_key_value_heads, head_dim])
                .transpose(1, 2);
            if let Some(norm) = &self.attention_k_norm {
                key = norm.forward(&key);
            }

            key
        };
        let value_states = {
            let mut value_states = hidden_states.q_mat_mul(&self.attention_wv);

            if let Some(bias) = &self.bias {
                value_states = value_states.add_(&bias.bias_v);
            }

            value_states
                .reshape([b_sz, seq_len, num_key_value_heads, head_dim])
                .transpose(1, 2)
        };

        let (query_states, key_states) = rope_cache.forward(
            &query_states,
            &key_states,
            start_pos,
            pos_ids,
            self.interleaved_rope,
        );
        (query_states, key_states, value_states)
    }
}

pub struct GroupedAttention {
    pub attention_qkv: QMatrix,
}

impl GroupedAttention {
    fn forward(
        &self,
        num_heads: usize,
        head_dim: usize,
        num_key_value_heads: usize,
        x: &Tensor<3, f32>,
        rope_cache: &RopeImplementation,
        start_pos: usize,
        pos_ids: Option<&Tensor<2, f32>>,
    ) -> (Tensor<4, f32>, Tensor<4, f32>, Tensor<4, f32>) {
        let [b_sz, seq_len, _] = *x.shape();
        let qkv = x.q_mat_mul(&self.attention_qkv);

        let query_pos = num_heads * head_dim;
        let query_states = qkv.narrow(D::Minus1, 0, query_pos);
        let key_states = qkv.narrow(D::Minus1, query_pos, num_key_value_heads * head_dim);
        let value_states = qkv.narrow(
            D::Minus1,
            query_pos + num_key_value_heads * head_dim,
            num_key_value_heads * head_dim,
        );

        let query_states = query_states
            .reshape([b_sz, seq_len, num_heads, head_dim])
            .transpose(1, 2);
        let key_states = key_states
            .reshape([b_sz, seq_len, num_key_value_heads, head_dim])
            .transpose(1, 2);
        let value_states = value_states
            .reshape([b_sz, seq_len, num_key_value_heads, head_dim])
            .transpose(1, 2);

        let (query_states, key_states) =
            rope_cache.forward(&query_states, &key_states, start_pos, pos_ids, false);

        (query_states, key_states, value_states)
    }
}

pub struct LlamaAttention {
    pub attention_variant: AttentionVariant,
    pub attention_wo: Linear<f32>,
    pub attention_norm: RmsNorm<1, f32>,
    pub post_attention_norm: Option<RmsNorm<1, f32>>,
    pub feed_forward_variant: FeedForwardVariant,
    pub ffn_norm: RmsNorm<1, f32>,
    pub post_ffn_norm: Option<RmsNorm<1, f32>>,
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
        hidden_states: &Tensor<3, f32>,
        attention_mask: Option<&AttentionMask<f32>>,
        start_pos: usize,
        pos_ids: Option<&Tensor<2, f32>>,
        cache: Option<&mut KvCache<f32>>,
    ) -> Tensor<3, f32> {
        let [b_sz, q_len, _] = *hidden_states.shape();
        let hidden_size = self.hidden_size;
        let num_heads = self.n_head;
        let head_dim = self.head_dim;
        let num_key_value_heads = self.n_kv_head;

        let (query_states, key_states, value_states) = match self.attention_variant {
            AttentionVariant::Separate(ref attention) => attention.forward(
                num_heads,
                head_dim,
                num_key_value_heads,
                hidden_states,
                &self.rope_cache,
                start_pos,
                pos_ids,
            ),
            AttentionVariant::Grouped(ref attention) => attention.forward(
                num_heads,
                head_dim,
                num_key_value_heads,
                hidden_states,
                &self.rope_cache,
                start_pos,
                pos_ids,
            ),
        };

        let (key_states, value_states) = match cache {
            None => (key_states, value_states),
            Some(cache) => cache.append(&key_states, &value_states),
        };

        forward_attention_qkv(
            &query_states,
            &key_states,
            &value_states,
            &self.attention_wo,
            attention_mask,
            head_dim,
            b_sz,
            q_len,
            hidden_size,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn forward_attention_qkv(
    query_states: &Tensor<4, f32>,
    key_states: &Tensor<4, f32>,
    value_states: &Tensor<4, f32>,
    attention_wo: &Linear<f32>,
    attention_mask: Option<&AttentionMask<f32>>,
    head_dim: usize,
    b_sz: usize,
    q_len: usize,
    hidden_size: usize,
) -> Tensor<3, f32> {
    let scale = 1. / (head_dim as f64).sqrt();
    let attn_output = {
        // let mut attn_weights = query_states.mat_mul(&key_states.t()) * scale as f32;

        // if let Some(attention_mask) = attention_mask {
        //     attention_mask.forward(&mut attn_weights);
        // }

        // attn_weights = attn_weights.softmax_last_dim();

        // attn_weights.mat_mul(value_states)
        query_states.flash_attention(
            key_states,
            value_states,
            scale as f32,
            attention_mask.map(|m| m.mask()),
        )
    };

    let attn_output = attn_output.transpose(1, 2);

    let attn_output = attn_output.reshape([b_sz, q_len, hidden_size]);

    let attn_output = attention_wo.forward(&attn_output);

    attn_output
}
