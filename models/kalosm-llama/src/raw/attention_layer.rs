use super::cache::{AttentionCache, AttentionCacheValue};
use super::rope::RopeCache;
use candle_core::{quantized::QMatMul, Module, Tensor};
use candle_nn::LayerNorm;

pub struct LlamaAttention {
    pub attention_wq: QMatMul,
    pub attention_wk: QMatMul,
    pub attention_wv: QMatMul,
    pub attention_wo: QMatMul,
    pub attention_norm: LayerNorm,
    pub feed_forward_w1: QMatMul,
    pub feed_forward_w2: QMatMul,
    pub feed_forward_w3: QMatMul,
    pub ffn_norm: LayerNorm,
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
        attention_mask: Option<&Tensor>,
        start_pos: usize,
        cache: Option<&mut AttentionCache>,
    ) -> candle_core::Result<Tensor> {
        let bsz = hidden_states.dims()[0];
        let q_len = hidden_states.dims()[1];
        let hidden_size = self.hidden_size;
        let num_heads = self.n_head;
        let head_dim = self.head_dim;
        let num_key_value_heads = self.n_kv_head;
        let num_key_value_groups = num_heads / num_key_value_heads;

        let (query_states, key_states, value_states) =
            std::thread::scope(|s| -> Result<_, candle_core::Error> {
                let query_states = s.spawn(|| {
                    let query_states = self.attention_wq.forward(hidden_states)?;
                    query_states
                        .reshape((bsz, q_len, num_heads, head_dim))?
                        .transpose(1, 2)
                });
                let key_states = s.spawn(|| {
                    let key_states = self.attention_wk.forward(hidden_states)?;
                    key_states
                        .reshape((bsz, q_len, num_key_value_heads, head_dim))?
                        .transpose(1, 2)
                });
                let value_states = s.spawn(|| {
                    let value_states = self.attention_wv.forward(hidden_states)?;

                    value_states
                        .reshape((bsz, q_len, num_key_value_heads, head_dim))?
                        .transpose(1, 2)
                });

                let query_states = query_states.join().unwrap()?;
                let key_states = key_states.join().unwrap()?;

                let (query_states, key_states) =
                    self.rope_cache
                        .forward(&query_states, &key_states, start_pos)?;

                let value_states = value_states.join().unwrap()?;

                Ok((query_states, key_states, value_states))
            })?;

        let key_states = repeat_kv(key_states.clone(), num_key_value_groups)?;
        let value_states = repeat_kv(value_states, num_key_value_groups)?;

        let (key_states, value_states) = match cache {
            None => (key_states, value_states),
            Some(cache) => match &mut cache.0 {
                Some(AttentionCacheValue { key, value }) => {
                    let kv_seq_len = key_states.dim(candle_core::D::Minus2)?;
                    let (k, v) = if kv_seq_len == 0 {
                        (key_states, value_states)
                    } else {
                        let key_states = Tensor::cat(&[&*key, &key_states], 2)?.contiguous()?;
                        let value_states =
                            Tensor::cat(&[&*value, &value_states], 2)?.contiguous()?;
                        (key_states, value_states)
                    };

                    *key = k.clone();
                    *value = v.clone();

                    (k, v)
                }
                None => {
                    cache.0 = Some(AttentionCacheValue {
                        key: key_states.clone(),
                        value: value_states.clone(),
                    });
                    (key_states, value_states)
                }
            },
        };

        let mut attn_weights = (query_states.matmul(&key_states.t()?)? / (head_dim as f64).sqrt())?;

        if let Some(attention_mask) = attention_mask {
            let shape = attn_weights.shape();
            let attention_mask = attention_mask.broadcast_as(shape)?;
            let on_true =
                Tensor::new(f32::NEG_INFINITY, attn_weights.device())?.broadcast_as(shape)?;
            attn_weights = attention_mask.where_cond(&on_true, &attn_weights)?;
        }

        attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;

        let mut attn_output = attn_weights.matmul(&value_states.contiguous()?)?;

        if attn_output.dims() != [bsz, num_heads, q_len, head_dim] {
            return Err(candle_core::Error::Msg(format!(
                "`attn_output` should be of size {:?}, but is {:?}",
                [bsz, self.n_head, q_len, head_dim],
                attn_weights.dims()
            )));
        }

        attn_output = attn_output.transpose(1, 2)?;

        attn_output = attn_output.reshape(&[bsz, q_len, hidden_size])?;

        attn_output = self.attention_wo.forward(&attn_output)?;

        Ok(attn_output)
    }
}

fn repeat_kv(x: Tensor, num_key_value_groups: usize) -> candle_core::Result<Tensor> {
    if num_key_value_groups == 1 {
        Ok(x)
    } else {
        let (b_sz, n_kv_head, seq_len, head_dim) = x.dims4()?;
        let x = x
            .unsqueeze(2)?
            .expand((b_sz, n_kv_head, num_key_value_groups, seq_len, head_dim))?
            .reshape((b_sz, n_kv_head * num_key_value_groups, seq_len, head_dim))?;
        Ok(x)
    }
}
