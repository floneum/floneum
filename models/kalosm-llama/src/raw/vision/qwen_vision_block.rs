use candle_core::{quantized::QMatMul, Result, Tensor};
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
        hidden_size: usize,
    ) -> Result<Self> {
        let device = vb.device();
        let norm1 = RmsNorm::new(hidden_size, QWEN_EPS, vb.pp("norm1"))?;
        let norm2 = RmsNorm::new(hidden_size, QWEN_EPS, vb.pp("norm2"))?;

        let mlp = LlamaFeedForward::new_with_bias(
            QMatMul::from_arc(vb.get_no_shape("gate_proj.weight")?)?,
            Some(vb.get_no_shape("gate_proj.bias")?.dequantize(&device)?),
            QMatMul::from_arc(vb.get_no_shape("up_proj.weight")?)?,
            Some(vb.get_no_shape("up_proj.bias")?.dequantize(&device)?),
            QMatMul::from_arc(vb.get_no_shape("down_proj.weight")?)?,
            Some(vb.get_no_shape("down_proj.bias")?.dequantize(&device)?),
        );

        let attn = VisionAttention::new(vb, head_count, head_dim, hidden_size)?;

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
        let xs = (xs
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
    qkv: Linear,
    proj: Linear,
    head_count: usize,
    head_dim: usize,
    hidden_size: usize,
}

impl VisionAttention {
    fn new(
        vb: &VarBuilder,
        head_count: usize,
        head_dim: usize,
        hidden_size: usize,
    ) -> Result<Self> {
        let qkv = vb.get_no_shape("qkv.weight")?;
        let qkv_bias = vb.get_no_shape("qkv.bias")?.dequantize(&vb.device())?;
        let qkv = Linear::from_arc(qkv, Some(qkv_bias))?;
        let proj = vb.get_no_shape("proj.weight")?;
        let proj_bias = vb.get_no_shape("proj.bias")?.dequantize(&vb.device())?;
        let proj = Linear::from_arc(proj, Some(proj_bias))?;

        Ok(Self {
            qkv,
            proj,
            head_count,
            head_dim,
            hidden_size,
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
        let qkv = xs.apply(&self.qkv)?;

        // Then split up the qkv tensor into q, k, and v
        let qkv = qkv
            .reshape((seq_len, 3, self.head_count, ()))?
            .permute((1, 0, 2, 3))?;

        let unbound = unbind(&qkv, 0)?;
        let [q, k, v] = unbound.as_slice() else {
            return Err(candle_core::Error::msg(
                "Failed to unbind qkv tensor into q, k, and v",
            ));
        };

        let (q, k) = rope_cache.forward(&q.unsqueeze(0)?, &k.unsqueeze(0)?, start_pos)?;

        // Convert from [seq_len, head_count, batch_size] to [head_count, seq_len, batch_size]
        let query_states = q.transpose(0, 1)?;
        let key_states = k.transpose(0, 1)?;
        let value_states = v.transpose(0, 1)?;

        let (key_states, value_states) = match cache {
            None => (key_states, value_states),
            Some(cache) => cache.append(&key_states, &value_states)?,
        };

        let bsz = query_states.dim(2)?;

        let mut attention_mask = vec![vec![f32::NEG_INFINITY; seq_len]; seq_len];
        for pair in cu_seqlens.windows(2) {
            let [last, next] = pair else { unreachable!() };
            let last = *last as usize;
            let next = *next as usize;
            for i in last..next {
                for j in last..next {
                    attention_mask[i][j] = 0.0;
                }
            }
        }
        let attention_mask = AttentionMask::new(
            Tensor::from_iter(
                attention_mask.iter().flatten().copied(),
                query_states.device(),
            )?
            .reshape((1, seq_len, seq_len))?,
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
            self.hidden_size,
        )
    }
}

fn unbind(tensor: &Tensor, dim: usize) -> Result<Vec<Tensor>> {
    tensor
        .chunk(dim, tensor.dim(dim)?)?
        .into_iter()
        .map(|t| t.squeeze(dim))
        .collect()
}
