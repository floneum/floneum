use candle_core::{quantized::QMatMul, DType, Result, Tensor, D};
use candle_nn::Module;
use candle_transformers::{
    quantized_nn::{Linear, RmsNorm},
    quantized_var_builder::VarBuilder,
};
use kalosm_common::{AttentionMask, KvCache};

use crate::raw::{
    attention_layer::{forward_attention_qkv, LlamaFeedForward},
    rope::RopeCache,
};

pub(crate) struct VisionBlock {
    norm1: RmsNorm,
    norm2: RmsNorm,
    mlp: LlamaFeedForward,
    attn: VisionAttention,
}

impl VisionBlock {
    pub(crate) fn new(
        vb: &VarBuilder,
        head_count: usize,
        head_dim: usize,
        embed_dim: usize,
        layer_norm_eps: f64,
    ) -> Result<Self> {
        let device = vb.device();
        let norm1 = RmsNorm::new(embed_dim, layer_norm_eps, vb.pp("ln1")).unwrap();
        let norm2 = RmsNorm::new(embed_dim, layer_norm_eps, vb.pp("ln2")).unwrap();

        let mlp = LlamaFeedForward::new_with_bias(
            QMatMul::from_arc(vb.get_no_shape("ffn_gate.weight").unwrap()).unwrap(),
            Some(
                vb.get_no_shape("ffn_gate.bias")
                    .unwrap()
                    .dequantize(device)
                    .unwrap(),
            ),
            QMatMul::from_arc(vb.get_no_shape("ffn_down.weight").unwrap()).unwrap(),
            Some(
                vb.get_no_shape("ffn_down.bias")
                    .unwrap()
                    .dequantize(device)
                    .unwrap(),
            ),
            QMatMul::from_arc(vb.get_no_shape("ffn_up.weight").unwrap()).unwrap(),
            Some(
                vb.get_no_shape("ffn_up.bias")
                    .unwrap()
                    .dequantize(device)
                    .unwrap(),
            ),
        );

        let attn = VisionAttention::new(vb, head_count, head_dim, embed_dim).unwrap();

        Ok(Self {
            norm1,
            norm2,
            mlp,
            attn,
        })
    }

    pub(crate) fn forward(
        &self,
        xs: &Tensor,
        cu_seqlens: &[u32],
        rope_cache: &RopeCache,
        cache: Option<&mut KvCache>,
    ) -> Result<Tensor> {
        let xs = xs.to_dtype(DType::F32).unwrap();
        let after_norm = self.norm1.forward(&xs).unwrap();
        let after_attention = self
            .attn
            .forward(&after_norm, cu_seqlens, rope_cache, cache)
            .unwrap();
        let xs = (&xs + after_attention).unwrap();
        &xs + self.mlp.forward(&self.norm2.forward(&xs).unwrap()).unwrap()
    }
}

struct VisionAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    proj: Linear,
    head_count: usize,
    head_dim: usize,
    embed_dim: usize,
}

impl VisionAttention {
    fn new(vb: &VarBuilder, head_count: usize, head_dim: usize, embed_dim: usize) -> Result<Self> {
        let q = vb.get_no_shape("attn_q.weight").unwrap();
        let q_bias = vb
            .get_no_shape("attn_q.bias")
            .unwrap()
            .dequantize(vb.device())
            .unwrap();
        let q = Linear::from_arc(q, Some(q_bias)).unwrap();
        let k = vb.get_no_shape("attn_k.weight").unwrap();
        let k_bias = vb
            .get_no_shape("attn_k.bias")
            .unwrap()
            .dequantize(vb.device())
            .unwrap();
        let k = Linear::from_arc(k, Some(k_bias)).unwrap();
        let v = vb.get_no_shape("attn_v.weight").unwrap();
        let v_bias = vb
            .get_no_shape("attn_v.bias")
            .unwrap()
            .dequantize(vb.device())
            .unwrap();
        let v = Linear::from_arc(v, Some(v_bias)).unwrap();
        let proj = vb.get_no_shape("attn_out.weight").unwrap();
        let proj_bias = vb
            .get_no_shape("attn_out.bias")
            .unwrap()
            .dequantize(vb.device())
            .unwrap();
        let proj = Linear::from_arc(proj, Some(proj_bias)).unwrap();

        Ok(Self {
            q,
            k,
            v,
            proj,
            head_count,
            head_dim,
            embed_dim,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        cu_seqlens: &[u32],
        rope_cache: &RopeCache,
        cache: Option<&mut KvCache>,
    ) -> Result<Tensor> {
        let seq_len = xs.dim(0).unwrap();

        // First, pass the input through the qkv layer
        let q = xs
            .apply(&self.q)
            .unwrap()
            .reshape((seq_len, self.head_count, ()))
            .unwrap();

        let k = xs
            .apply(&self.k)
            .unwrap()
            .reshape((seq_len, self.head_count, ()))
            .unwrap();

        let v = xs
            .apply(&self.v)
            .unwrap()
            .reshape((seq_len, self.head_count, ()))
            .unwrap();

        fn apply_rotary_pos_emb_vision(
            rope_cache: &RopeCache,
            q: &Tensor,
            k: &Tensor,
        ) -> candle_core::Result<(Tensor, Tensor)> {
            let sin = rope_cache.sin();
            let cos = rope_cache.cos();
            let sin = Tensor::cat(&[sin, sin], D::Minus1)?.unsqueeze(1)?;
            let cos = Tensor::cat(&[cos, cos], D::Minus1)?.unsqueeze(1)?;

            Ok((
                (q.broadcast_mul(&cos)? + rotate_half(q)?.broadcast_mul(&sin)?)?,
                (k.broadcast_mul(&cos)? + rotate_half(k)?.broadcast_mul(&sin)?)?,
            ))
        }

        fn rotate_half(xs: &Tensor) -> Result<Tensor> {
            let last_dim = xs.dim(D::Minus1)?;
            let xs1 = xs.narrow(D::Minus1, 0, last_dim / 2)?;
            let xs2 = xs.narrow(D::Minus1, last_dim / 2, last_dim - last_dim / 2)?;
            Tensor::cat(&[&xs2.neg()?, &xs1], D::Minus1)
        }

        let (q, k) = apply_rotary_pos_emb_vision(rope_cache, &q, &k).unwrap();

        let q = q.squeeze(0).unwrap();
        let k = k.squeeze(0).unwrap();

        // Convert from [seq_len, head_count, batch_size] to [head_count, seq_len, batch_size]
        let query_states = q
            .transpose(0, 1)
            .unwrap()
            .unsqueeze(0)
            .unwrap()
            .contiguous()
            .unwrap();
        let key_states = k
            .transpose(0, 1)
            .unwrap()
            .unsqueeze(0)
            .unwrap()
            .contiguous()
            .unwrap();
        let value_states = v
            .transpose(0, 1)
            .unwrap()
            .unsqueeze(0)
            .unwrap()
            .contiguous()
            .unwrap();

        let (key_states, value_states) = match cache {
            None => (key_states, value_states),
            Some(cache) => cache.append(&key_states, &value_states).unwrap(),
        };

        let bsz = 1;

        let mut attention_mask = vec![vec![1u32; seq_len]; seq_len];
        for pair in cu_seqlens.windows(2) {
            let [last, next] = pair else { unreachable!() };
            let last = *last as usize;
            let next = *next as usize;
            #[allow(clippy::needless_range_loop)]
            for i in last..next {
                for j in last..next {
                    attention_mask[i][j] = 0;
                }
            }
        }
        let attention_mask = AttentionMask::new(
            Tensor::from_iter(
                attention_mask.iter().flatten().copied(),
                query_states.device(),
            )
            .unwrap()
            .reshape((1, 1, seq_len, seq_len))
            .unwrap(),
        );

        forward_attention_qkv(
            &query_states,
            &key_states,
            &value_states,
            &self.proj,
            Some(&attention_mask),
            self.head_count,
            self.head_dim,
            bsz,
            seq_len,
            self.embed_dim,
        )
        .unwrap()
        .squeeze(0)
    }
}
