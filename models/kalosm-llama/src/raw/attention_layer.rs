use super::cache::KvCache;
use super::mask::AttentionMask;
use super::rope::RopeCache;
use super::RmsNorm;
use fusor_core::QMatrix;
use fusor_core::Tensor;

pub enum FeedForwardVariant {
    Llama(LlamaFeedForward),
    Phi(PhiFeedForward),
}

impl FeedForwardVariant {
    pub(crate) fn forward(&self, x: &Tensor<2, f32>) -> Tensor<2, f32> {
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
    pub(crate) fn forward(&self, x: &Tensor<2, f32>) -> Tensor<2, f32> {
        let up_states = x.q_mat_mul(&self.up);
        let last_dim = *up_states.shape().last().unwrap();
        let gate = up_states.narrow(last_dim, 0, self.feed_forward_length);
        let up_states =
            up_states.narrow(last_dim, self.feed_forward_length, self.feed_forward_length);
        let gate = gate.silu();
        let up_states = up_states * gate;
        up_states.q_mat_mul(&self.down)
    }
}

pub struct LlamaFeedForward {
    pub feed_forward_w1: QMatrix,
    pub feed_forward_w2: QMatrix,
    pub feed_forward_w3: QMatrix,
}

impl LlamaFeedForward {
    fn forward(&self, x: &Tensor<2, f32>) -> Tensor<2, f32> {
        let w1 = x.q_mat_mul(&self.feed_forward_w1);
        let w1 = w1.silu();

        let w3 = x.q_mat_mul(&self.feed_forward_w3);

        (w1 * w3).q_mat_mul(&self.feed_forward_w2)
    }
}

pub enum AttentionVariant {
    Separate(SeparateAttention),
    Grouped(GroupedAttention),
}

pub struct AttentionBias {
    pub bias_q: Tensor<2, f32>,
    pub bias_k: Tensor<2, f32>,
    pub bias_v: Tensor<2, f32>,
}

pub struct SeparateAttention {
    pub attention_wq: QMatrix,
    pub attention_wk: QMatrix,
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
        hidden_states: &Tensor<2, f32>,
        rope_cache: &RopeCache,
        start_pos: usize,
    ) -> (Tensor<3, f32>, Tensor<3, f32>, Tensor<3, f32>) {
        let seq_len = hidden_states.shape()[0];

        let query_states = {
            let mut query_states = hidden_states.q_mat_mul(&self.attention_wq);

            if let Some(bias) = &self.bias {
                query_states = &query_states + &bias.bias_q;
            }

            query_states
                .reshape([seq_len, num_heads, head_dim])
                .transpose(0, 1)
        };
        let key_states = {
            let mut key_states = hidden_states.q_mat_mul(&self.attention_wk);

            if let Some(bias) = &self.bias {
                key_states = &key_states + &bias.bias_k;
            }

            key_states
                .reshape([seq_len, num_key_value_heads, head_dim])
                .transpose(0, 1)
        };
        let value_states = {
            let mut value_states = hidden_states.q_mat_mul(&self.attention_wv);

            if let Some(bias) = &self.bias {
                value_states = &value_states + &bias.bias_v;
            }

            value_states
                .reshape([seq_len, num_key_value_heads, head_dim])
                .transpose(0, 1)
        };

        let (query_states, key_states) = if self.interleaved_rope {
            rope_cache.forward_i(query_states, key_states, start_pos)
        } else {
            rope_cache.forward(query_states, key_states, start_pos)
        };

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
        x: &Tensor<2, f32>,
        rope_cache: &RopeCache,
        start_pos: usize,
    ) -> (Tensor<3, f32>, Tensor<3, f32>, Tensor<3, f32>) {
        let seq_len = x.shape()[0];
        let qkv = x.q_mat_mul(&self.attention_qkv);

        let query_pos = num_heads * head_dim;
        let last_dim = *qkv.shape().last().unwrap();
        let query_states = qkv.narrow(last_dim, 0, query_pos);
        let key_states = qkv.narrow(last_dim, query_pos, num_key_value_heads * head_dim);
        let value_states = qkv.narrow(
            last_dim,
            query_pos + num_key_value_heads * head_dim,
            num_key_value_heads * head_dim,
        );

        let query_states = query_states
            .reshape([seq_len, num_heads, head_dim])
            .transpose(1, 2);
        let key_states = key_states
            .reshape([seq_len, num_key_value_heads, head_dim])
            .transpose(1, 2);
        let value_states = value_states
            .reshape([seq_len, num_key_value_heads, head_dim])
            .transpose(1, 2);

        let (query_states, key_states) = rope_cache.forward(query_states, key_states, start_pos);

        (query_states, key_states, value_states)
    }
}

pub struct LlamaAttention {
    pub attention_variant: AttentionVariant,
    pub attention_wo: QMatrix,
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
        hidden_states: &Tensor<2, f32>,
        attention_mask: Option<&AttentionMask>,
        start_pos: usize,
        cache: Option<&mut KvCache>,
    ) -> Tensor<2, f32> {
        let q_len = hidden_states.shape()[0];
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
            ),
            AttentionVariant::Grouped(ref attention) => attention.forward(
                num_heads,
                head_dim,
                num_key_value_heads,
                hidden_states,
                &self.rope_cache,
                start_pos,
            ),
        };
        let key_states = repeat_kv(key_states.clone(), num_key_value_groups);
        let value_states = repeat_kv(value_states, num_key_value_groups);

        let (key_states, value_states) = match cache {
            None => (key_states, value_states),
            Some(cache) => cache.append(&key_states, &value_states),
        };

        debug_assert!(head_dim > 0);
        let scale = 1. / (head_dim as f32).sqrt();

        let attn_output = {
            let mut attn_weights = query_states.mat_mul(&key_states.t()) * scale;

            if let Some(attention_mask) = attention_mask {
                attention_mask.forward(&mut attn_weights);
            }
            attn_weights = attn_weights.softmax_last_dim();

            attn_weights.mat_mul(&value_states)
        };

        let attn_output = attn_output.transpose(0, 1);

        let attn_output = attn_output.reshape([q_len, hidden_size]);

        let attn_output = attn_output.q_mat_mul(&self.attention_wo);

        attn_output
    }
}

fn repeat_kv(x: Tensor<3, f32>, num_key_value_groups: usize) -> Tensor<3, f32> {
    if num_key_value_groups == 1 {
        x
    } else {
        // Could this just be a stride transformation?
        let [n_kv_head, seq_len, head_dim] = *x.shape();
        Tensor::cat(vec![x; num_key_value_groups], 1).reshape([
            n_kv_head * num_key_value_groups,
            seq_len,
            head_dim,
        ])
    }
}
