use crate::raw::rope::RopeImplementation;

use fusor_core::cache::AttentionMask;
use fusor_core::cache::KvCache;
use fusor_core::layers::Linear;
use fusor_core::layers::RmsNorm;
use fusor_core::CastTensor;
use fusor_core::FloatDataType;
use fusor_core::QMatrix;
use fusor_core::Tensor;
use fusor_core::D;

pub enum FeedForwardVariant<F: FloatDataType = f32> {
    // Used by the Llama, Qwen, and Gemma models
    Llama(LlamaFeedForward<F>),
    // Used by the Phi models
    Phi(PhiFeedForward),
}

impl<F: FloatDataType> FeedForwardVariant<F> {
    pub(crate) fn forward(&self, x: &Tensor<3, F>) -> Tensor<3, F> {
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
    pub(crate) fn forward<F: FloatDataType>(&self, x: &Tensor<3, F>) -> Tensor<3, F> {
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

pub struct LlamaFeedForward<F: FloatDataType = f32> {
    gate: QMatrix,
    gate_bias: Option<Tensor<1, F>>,
    down: QMatrix,
    down_bias: Option<Tensor<1, F>>,
    up: QMatrix,
    up_bias: Option<Tensor<1, F>>,
}

impl<F: FloatDataType> LlamaFeedForward<F> {
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
        gate_bias: Option<Tensor<1, F>>,
        down: QMatrix,
        down_bias: Option<Tensor<1, F>>,
        up: QMatrix,
        up_bias: Option<Tensor<1, F>>,
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

    pub(crate) fn forward(&self, x: &Tensor<3, F>) -> Tensor<3, F> {
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

pub enum AttentionVariant<F: FloatDataType = f32> {
    Separate(SeparateAttention<F>),
    Grouped(GroupedAttention),
}

pub struct AttentionBias<F: FloatDataType = f32> {
    bias_q: Tensor<1, F>,
    bias_k: Tensor<1, F>,
    bias_v: Tensor<1, F>,
}

impl<F: FloatDataType> AttentionBias<F> {
    pub fn new(q: Tensor<1, F>, k: Tensor<1, F>, v: Tensor<1, F>) -> Self {
        Self {
            bias_q: q,
            bias_k: k,
            bias_v: v,
        }
    }
}

pub struct SeparateAttention<F: FloatDataType = f32> {
    pub attention_wq: QMatrix,
    pub attention_q_norm: Option<RmsNorm<1, F>>,
    pub attention_wk: QMatrix,
    pub attention_k_norm: Option<RmsNorm<1, F>>,
    pub attention_wv: QMatrix,
    pub bias: Option<AttentionBias<F>>,
    pub interleaved_rope: bool,
}

impl<F: FloatDataType> SeparateAttention<F>
where
    F: CastTensor<f32>,
    f32: CastTensor<F>,
{
    fn forward(
        &self,
        num_heads: usize,
        head_dim: usize,
        num_key_value_heads: usize,
        hidden_states: &Tensor<3, F>,
        rope_cache: &RopeImplementation<F>,
        start_pos: usize,
        pos_ids: Option<&Tensor<2, F>>,
    ) -> (Tensor<4, F>, Tensor<4, F>, Tensor<4, F>) {
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
    fn forward<F: FloatDataType>(
        &self,
        num_heads: usize,
        head_dim: usize,
        num_key_value_heads: usize,
        x: &Tensor<3, F>,
        rope_cache: &RopeImplementation<F>,
        start_pos: usize,
        pos_ids: Option<&Tensor<2, F>>,
    ) -> (Tensor<4, F>, Tensor<4, F>, Tensor<4, F>)
    where
        f32: CastTensor<F>,
    {
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

pub struct LlamaAttention<F: FloatDataType = f32> {
    pub attention_variant: AttentionVariant<F>,
    pub attention_wo: Linear<F>,
    pub attention_norm: RmsNorm<1, F>,
    pub post_attention_norm: Option<RmsNorm<1, F>>,
    pub feed_forward_variant: FeedForwardVariant<F>,
    pub ffn_norm: RmsNorm<1, F>,
    pub post_ffn_norm: Option<RmsNorm<1, F>>,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    pub hidden_size: usize,
    pub rope_cache: RopeImplementation<F>,
    pub(crate) sliding_window_size: Option<usize>,
}

impl<F: FloatDataType> LlamaAttention<F>
where
    F: CastTensor<f32>,
    f32: CastTensor<F>,
{
    pub(crate) fn forward(
        &self,
        hidden_states: &Tensor<3, F>,
        attention_mask: Option<&AttentionMask<F>>,
        start_pos: usize,
        pos_ids: Option<&Tensor<2, F>>,
        cache: Option<&mut KvCache<F>>,
    ) -> Tensor<3, F> {
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
pub(crate) fn forward_attention_qkv<F: FloatDataType>(
    query_states: &Tensor<4, F>,
    key_states: &Tensor<4, F>,
    value_states: &Tensor<4, F>,
    attention_wo: &Linear<F>,
    attention_mask: Option<&AttentionMask<F>>,
    head_dim: usize,
    b_sz: usize,
    q_len: usize,
    hidden_size: usize,
) -> Tensor<3, F> {
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
