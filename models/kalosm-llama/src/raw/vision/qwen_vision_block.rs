use candle_core::{quantized::QMatMul, DType, Result, Tensor};
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

use super::QWEN_EPS;

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
    ) -> Result<Self> {
        let device = vb.device();
        let norm1 = RmsNorm::new(embed_dim, QWEN_EPS, vb.pp("ln1"))?;
        let norm2 = RmsNorm::new(embed_dim, QWEN_EPS, vb.pp("ln2"))?;

        let mlp = LlamaFeedForward::new_with_bias(
            QMatMul::from_arc(vb.get_no_shape("ffn_gate.weight")?)?,
            Some(vb.get_no_shape("ffn_gate.bias")?.dequantize(&device)?),
            QMatMul::from_arc(vb.get_no_shape("ffn_down.weight")?)?,
            Some(vb.get_no_shape("ffn_down.bias")?.dequantize(&device)?),
            QMatMul::from_arc(vb.get_no_shape("ffn_up.weight")?)?,
            Some(vb.get_no_shape("ffn_up.bias")?.dequantize(&device)?),
        );

        let attn = VisionAttention::new(&vb, head_count, head_dim, embed_dim)?;

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
        start_pos: usize,
        cache: Option<&mut KvCache>,
    ) -> Result<Tensor> {
        let xs = xs.to_dtype(DType::F32)?;
        let xs = (&xs
            + self.attn.forward(
                &self.norm1.forward(&xs)?,
                cu_seqlens,
                rope_cache,
                start_pos,
                cache,
            )?)?;
        &xs + self.mlp.forward(&self.norm2.forward(&xs)?)?
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
        let q = vb.get_no_shape("attn_q.weight")?;
        let q_bias = vb.get_no_shape("attn_q.bias")?.dequantize(&vb.device())?;
        let q = Linear::from_arc(q, Some(q_bias))?;
        let k = vb.get_no_shape("attn_k.weight")?;
        let k_bias = vb.get_no_shape("attn_k.bias")?.dequantize(&vb.device())?;
        let k = Linear::from_arc(k, Some(k_bias))?;
        let v = vb.get_no_shape("attn_v.weight")?;
        let v_bias = vb.get_no_shape("attn_v.bias")?.dequantize(&vb.device())?;
        let v = Linear::from_arc(v, Some(v_bias))?;
        let proj = vb.get_no_shape("attn_out.weight")?;
        let proj_bias = vb.get_no_shape("attn_out.bias")?.dequantize(&vb.device())?;
        let proj = Linear::from_arc(proj, Some(proj_bias))?;

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
        start_pos: usize,
        cache: Option<&mut KvCache>,
    ) -> Result<Tensor> {
        let seq_len = xs.dim(0)?;

        // First, pass the input through the qkv layer
        let q = xs
            .apply(&self.q)?
            .reshape((seq_len, self.head_count, ()))?;

        let k = xs
            .apply(&self.k)?
            .reshape((seq_len, self.head_count, ()))?;

        let v = xs
            .apply(&self.v)?
            .reshape((seq_len, self.head_count, ()))?;

        let (q, k) = rope_cache.forward(&q.unsqueeze(0)?, &k.unsqueeze(0)?, start_pos)?;
        let q = q.squeeze(0)?;
        let k = k.squeeze(0)?;

        // Convert from [seq_len, head_count, batch_size] to [head_count, seq_len, batch_size]
        let query_states = q.transpose(0, 1)?.unsqueeze(0)?.contiguous()?;
        let key_states = k.transpose(0, 1)?.unsqueeze(0)?.contiguous()?;
        let value_states = v.transpose(0, 1)?.unsqueeze(0)?.contiguous()?;

        let (key_states, value_states) = match cache {
            None => (key_states, value_states),
            Some(cache) => cache.append(&key_states, &value_states)?,
        };

        let bsz = 1;

        let mut attention_mask = vec![vec![1u32; seq_len]; seq_len];
        for pair in cu_seqlens.windows(2) {
            let [last, next] = pair else { unreachable!() };
            let last = *last as usize;
            let next = *next as usize;
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
            )?
            .reshape((1, 1, seq_len, seq_len))?,
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
        )?
        .squeeze(0)
    }
}

fn unbind(tensor: &Tensor, dim: usize) -> Result<Vec<Tensor>> {
    tensor
        .chunk(tensor.dim(dim)?, dim)?
        .into_iter()
        .map(|t| t.squeeze(dim))
        .collect()
}
