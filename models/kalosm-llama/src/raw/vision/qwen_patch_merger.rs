use fusor_core::{
    layers::{Linear, RmsNorm},
    CastTensor, Device, FloatDataType, Tensor, VarBuilder,
};

pub(crate) struct Qwen2VLPatchMerger<F: FloatDataType> {
    hidden_size: usize,
    ln_q: RmsNorm<1, F>,
    mlp: [Linear<F>; 2],
}

impl<F: FloatDataType> Qwen2VLPatchMerger<F>
where
    f32: CastTensor<F>,
    F: CastTensor<f32>,
{
    pub(crate) fn new(
        in_channels: usize,
        _out_channels: usize,
        spatial_merge_size: usize,
        layer_norm_eps: f64,
        vb: &mut VarBuilder,
        device: &Device,
    ) -> fusor_core::Result<Self> {
        let hidden_size = in_channels * spatial_merge_size.pow(2);

        let ln_q_weight = vb.get("v.post_ln.weight", device)?.dequantize();
        let ln_q = RmsNorm::new(ln_q_weight, None, layer_norm_eps as f32);

        let mlp_0_weight = vb.get("mm.0.weight", device)?;
        let mlp_0_bias = vb.get("mm.0.bias", device)?.dequantize();
        let mlp_0 = Linear::new(mlp_0_weight, Some(mlp_0_bias));

        let mlp_2_weight = vb.get("mm.2.weight", device)?;
        let mlp_2_bias = vb.get("mm.2.bias", device)?.dequantize();
        let mlp_2 = Linear::new(mlp_2_weight, Some(mlp_2_bias));

        Ok(Self {
            hidden_size,
            ln_q,
            mlp: [mlp_0, mlp_2],
        })
    }

    pub(crate) fn forward(&self, x: &Tensor<2, F>) -> Tensor<2, F> {
        let [seq_len, _] = *x.shape();
        let x = self.ln_q.forward(x);
        let x = x.reshape([
            seq_len / (self.hidden_size / x.shape()[1]),
            self.hidden_size,
        ]);

        // Linear expects 3D input: (batch, seq, dim)
        // We have (seq_len, dim). Treat as (1, seq_len, dim).
        let x_3d = x.unsqueeze(0);

        let x_3d = self.mlp[0].forward(&x_3d);
        let x_3d = x_3d.gelu();
        let x_3d = self.mlp[1].forward(&x_3d);

        // Squeeze back to 2D
        let x = x_3d.squeeze(0);
        x
    }
}
