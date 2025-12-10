use fusor_core::{
    CastTensor, D, Device, FloatDataType, Tensor, VarBuilder, cache::{AttentionMask, KvCache}, layers::{Linear, RmsNorm}
};

use crate::raw::{
    attention_layer::{forward_attention_qkv, LlamaFeedForward},
    rope::RopeCache,
};

pub(crate) struct VisionBlock<F: FloatDataType> {
    norm1: RmsNorm<1, F>,
    norm2: RmsNorm<1, F>,
    mlp: LlamaFeedForward<F>,
    attn: VisionAttention<F>,
}

impl<F: FloatDataType> VisionBlock<F>
where
    f32: CastTensor<F>,
    F: CastTensor<f32>,
{
    pub(crate) fn new(
        vb: &mut VarBuilder,
        device: &Device,
        head_count: usize,
        head_dim: usize,
        embed_dim: usize,
        layer_norm_eps: f64,
    ) -> fusor_core::Result<Self> {
        // norm1, norm2
        let norm1_weight = vb.get("ln1.weight", device)?.dequantize();
        let norm1 = RmsNorm::new(norm1_weight, None, layer_norm_eps as f32);

        let norm2_weight = vb.get("ln2.weight", device)?.dequantize();
        let norm2 = RmsNorm::new(norm2_weight, None, layer_norm_eps as f32);

        // MLP
        let gate = vb.get("ffn_gate.weight", device)?;
        let gate_bias = vb.get("ffn_gate.bias", device)?.dequantize();
        let down = vb.get("ffn_down.weight", device)?;
        let down_bias = vb.get("ffn_down.bias", device)?.dequantize();
        let up = vb.get("ffn_up.weight", device)?;
        let up_bias = vb.get("ffn_up.bias", device)?.dequantize();
        let mlp = LlamaFeedForward::new_with_bias(
            gate,
            Some(gate_bias),
            down,
            Some(down_bias),
            up,
            Some(up_bias),
        );

        let attn = VisionAttention::new(vb, device, head_count, head_dim, embed_dim)?;

        Ok(Self {
            norm1,
            norm2,
            mlp,
            attn,
        })
    }

    pub(crate) fn forward(
        &self,
        xs: &Tensor<2, F>,
        cu_seqlens: &[u32],
        rope_cache: &RopeCache<F>,
        cache: Option<&mut KvCache<F>>,
    ) -> fusor_core::Result<Tensor<2, F>> {
        let xs_3d = xs.unsqueeze(0); // [1, seq, dim]
        let after_norm = self.norm1.forward(&xs_3d);
        let after_attention = self
            .attn
            .forward(&after_norm, cu_seqlens, rope_cache, cache)?;

        let xs_3d = xs_3d + after_attention;
        let after_norm2 = self.norm2.forward(&xs_3d);
        let mlp_out = self.mlp.forward(&after_norm2); // LlamaFeedForward expects Tensor<3, F>

        let out = xs_3d + mlp_out;

        Ok(out.squeeze(0))
    }
}

struct VisionAttention<F: FloatDataType> {
    q: Linear<F>,
    k: Linear<F>,
    v: Linear<F>,
    proj: Linear<F>,
    head_count: usize,
    head_dim: usize,
    embed_dim: usize,
}

impl<F: FloatDataType> VisionAttention<F>
where
    f32: CastTensor<F>,
    F: CastTensor<f32>,
{
    fn new(
        vb: &mut VarBuilder,
        device: &Device,
        head_count: usize,
        head_dim: usize,
        embed_dim: usize,
    ) -> fusor_core::Result<Self> {
        let q = Linear::new(
            vb.get("attn_q.weight", device)?,
            Some(vb.get("attn_q.bias", device)?.dequantize()),
        );
        let k = Linear::new(
            vb.get("attn_k.weight", device)?,
            Some(vb.get("attn_k.bias", device)?.dequantize()),
        );
        let v = Linear::new(
            vb.get("attn_v.weight", device)?,
            Some(vb.get("attn_v.bias", device)?.dequantize()),
        );
        let proj = Linear::new(
            vb.get("attn_out.weight", device)?,
            Some(vb.get("attn_out.bias", device)?.dequantize()),
        );

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
        xs: &Tensor<3, F>, // [1, seq, dim]
        cu_seqlens: &[u32],
        rope_cache: &RopeCache<F>,
        cache: Option<&mut KvCache<F>>,
    ) -> fusor_core::Result<Tensor<3, F>> {
        let [bsz, seq_len, _] = *xs.shape();

        // qkv
        let q = self
            .q
            .forward(xs)
            .reshape([seq_len, self.head_count, self.head_dim]);
        let k = self
            .k
            .forward(xs)
            .reshape([seq_len, self.head_count, self.head_dim]);
        let v = self
            .v
            .forward(xs)
            .reshape([seq_len, self.head_count, self.head_dim]);

        let sin = rope_cache.sin();
        let cos = rope_cache.cos();

        // sin/cos: [total_seq, head_dim/2]
        // Expand to [total_seq, head_count, head_dim]
        let sin = Tensor::cat(vec![sin.clone(); 2], D::Minus1)
            .unsqueeze(1)
            .expand([seq_len, self.head_count, self.head_dim]);
        let cos = Tensor::cat(vec![cos.clone(); 2], D::Minus1)
            .unsqueeze(1)
            .expand([seq_len, self.head_count, self.head_dim]);

        // Rotate half
        let rotate_half = |x: &Tensor<3, F>| {
            // x: [seq, heads, dim]
            let last_dim = x.shape().last().copied().unwrap();
            let half = last_dim / 2;
            let x1 = x.narrow(D::Minus1, 0, half);
            let x2 = x.narrow(D::Minus1, half, half);
            Tensor::cat([-x2, x1], D::Minus1)
        };

        let q_embed = (q.clone() * cos.clone()) + (rotate_half(&q) * sin.clone());
        let k_embed = (k.clone() * cos) + (rotate_half(&k) * sin);

        // Transpose to [heads, seq, dim] -> [1, heads, seq, dim] (batch=1)
        let query_states = q_embed.transpose(0, 1).unsqueeze(0);
        let key_states = k_embed.transpose(0, 1).unsqueeze(0);
        let value_states = v.transpose(0, 1).unsqueeze(0);

        // Cache append
        let (key_states, value_states) = match cache {
            None => (key_states, value_states),
            Some(cache) => cache.append(&key_states, &value_states),
        };

        // Mask
        let mut mask_vec = vec![f32::NEG_INFINITY; seq_len * seq_len];
        for pair in cu_seqlens.windows(2) {
            let last = pair[0] as usize;
            let next = pair[1] as usize;
            for i in last..next {
                for j in last..next {
                    mask_vec[i * seq_len + j] = 0.0;
                }
            }
        }

        let mask_tensor = Tensor::new(xs.device(), &mask_vec).reshape([seq_len, seq_len]);
        let mask = AttentionMask::new(mask_tensor.cast());

        let output = forward_attention_qkv(
            &query_states,
            &key_states,
            &value_states,
            &self.proj,
            Some(&mask),
            self.head_dim,
            bsz,
            seq_len,
            self.embed_dim,
        );

        Ok(output)
    }
}