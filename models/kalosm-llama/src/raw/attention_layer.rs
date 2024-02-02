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

        let query_states = self.attention_wq.forward(hidden_states).unwrap();
        let key_states = self.attention_wk.forward(hidden_states).unwrap();
        let value_states = self.attention_wv.forward(hidden_states).unwrap();

        let query_states = query_states
            .reshape((bsz, q_len, num_heads, head_dim))
            .unwrap()
            .transpose(1, 2)
            .unwrap();
        let key_states = key_states
            .reshape((bsz, q_len, num_key_value_heads, head_dim))
            .unwrap()
            .transpose(1, 2)
            .unwrap();
        let v = value_states
            .reshape((bsz, q_len, num_key_value_heads, head_dim))
            .unwrap()
            .transpose(1, 2)
            .unwrap();

        let kv_seq_len = key_states.dim(candle_core::D::Minus2).unwrap();
        let (q, k) = self
            .rope_cache
            .forward(&query_states, &key_states, start_pos)
            .unwrap();

        let (key_states, value_states) = match cache {
            None => (k, v),
            Some(cache) => match &mut cache.0 {
                Some(AttentionCacheValue { key, value }) => {
                    let (k, v) = if kv_seq_len == 0 {
                        (k, v)
                    } else {
                        let k = Tensor::cat(&[&*key, &k], 2).unwrap().contiguous().unwrap();
                        let v = Tensor::cat(&[&*value, &v], 2)
                            .unwrap()
                            .contiguous()
                            .unwrap();
                        (k, v)
                    };

                    *key = k.clone();
                    *value = v.clone();

                    (k, v)
                }
                None => {
                    cache.0 = Some(AttentionCacheValue {
                        key: k.clone(),
                        value: v.clone(),
                    });
                    (k, v)
                }
            },
        };

        let key_states = repeat_kv(key_states.clone(), num_key_value_groups).unwrap();
        let value_states = repeat_kv(value_states, num_key_value_groups).unwrap();

        let mut attn_weights = (q.matmul(&key_states.transpose(2, 3).unwrap()).unwrap()
            / (head_dim as f64).sqrt())
        .unwrap();

        if let Some(attention_mask) = attention_mask {
            attn_weights = attn_weights.broadcast_add(attention_mask).unwrap();
        }

        attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights).unwrap();

        let mut attn_output = attn_weights.matmul(&value_states).unwrap();

        if attn_output.dims() != [bsz, num_heads, q_len, head_dim] {
            return Err(candle_core::Error::Msg(format!(
                "`attn_output` should be of size {:?}, but is {:?}",
                [bsz, self.n_head, q_len, head_dim],
                attn_weights.dims()
            )));
        }

        attn_output = attn_output.transpose(1, 2).unwrap().contiguous().unwrap();

        attn_output = attn_output.reshape(&[bsz, q_len, hidden_size]).unwrap();

        attn_output = self.attention_wo.forward(&attn_output).unwrap();

        Ok(attn_output)
    }
}

fn repeat_kv(x: Tensor, num_key_value_groups: usize) -> candle_core::Result<Tensor> {
    if num_key_value_groups == 1 {
        Ok(x)
    } else {
        let (b_sz, n_kv_head, seq_len, head_dim) = x.dims4().unwrap();
        let x = x
            .unsqueeze(2)
            .unwrap()
            .expand((b_sz, n_kv_head, num_key_value_groups, seq_len, head_dim))
            .unwrap()
            .reshape((b_sz, n_kv_head * num_key_value_groups, seq_len, head_dim))
            .unwrap();
        Ok(x)
    }
}
