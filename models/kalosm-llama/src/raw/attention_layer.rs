use super::cache::AttentionCache;
use super::mask::AttentionMask;
use super::rope::RopeCache;
use candle_core::Device;
use candle_core::{quantized::QMatMul, Module, Tensor};
use candle_transformers::quantized_nn::RmsNorm;

pub struct LlamaAttention {
    pub attention_wq: QMatMul,
    pub attention_wk: QMatMul,
    pub attention_wv: QMatMul,
    pub attention_wo: QMatMul,
    pub attention_norm: RmsNorm,
    pub feed_forward_w1: QMatMul,
    pub feed_forward_w2: QMatMul,
    pub feed_forward_w3: QMatMul,
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
        cache: Option<&mut AttentionCache>,
    ) -> candle_core::Result<Tensor> {
        let bsz = hidden_states.dims()[0];
        let q_len = hidden_states.dims()[1];
        let hidden_size = self.hidden_size;
        let num_heads = self.n_head;
        let head_dim = self.head_dim;
        let num_key_value_heads = self.n_kv_head;
        let num_key_value_groups = num_heads / num_key_value_heads;
        let device = hidden_states.device();

        let (query_states, key_states, value_states) = if matches!(device, Device::Cpu) {
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
            })?
        } else {
            let query_states = {
                let query_states = self.attention_wq.forward(hidden_states)?;
                query_states
                    .reshape((bsz, q_len, num_heads, head_dim))?
                    .transpose(1, 2)?
            };
            let key_states = {
                let key_states = self.attention_wk.forward(hidden_states)?;
                key_states
                    .reshape((bsz, q_len, num_key_value_heads, head_dim))?
                    .transpose(1, 2)?
            };
            let value_states = {
                let value_states = self.attention_wv.forward(hidden_states)?;

                value_states
                    .reshape((bsz, q_len, num_key_value_heads, head_dim))?
                    .transpose(1, 2)?
            };

            let (query_states, key_states) =
                self.rope_cache
                    .forward(&query_states, &key_states, start_pos)?;

            (query_states, key_states, value_states)
        };

        let key_states = repeat_kv(key_states.clone(), num_key_value_groups)?;
        let value_states = repeat_kv(value_states, num_key_value_groups)?;

        let (key_states, value_states) = match cache {
            None => (key_states, value_states),
            Some(cache) => cache.0.append(&key_states, &value_states)?,
        };

        let mut attn_weights = (query_states.matmul(&key_states.t()?)? / (head_dim as f64).sqrt())?;

        if let Some(attention_mask) = attention_mask {
            attention_mask.forward(&mut attn_weights)?;
        }

        attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;

        let mut attn_output = attn_weights.matmul(&value_states)?;

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
        Tensor::cat(&vec![&x; num_key_value_groups], 2)?.reshape((
            b_sz,
            n_kv_head * num_key_value_groups,
            seq_len,
            head_dim,
        ))
    }
}
