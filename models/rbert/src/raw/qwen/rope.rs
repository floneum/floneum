use fusor_core::{Device, Result, Tensor};

use super::model::QwenConfig;

fn create_inverse_frequency(
    dim: usize,
    rope_theta: f32,
    device: &Device,
) -> Tensor<2, f32> {
    let inverse_frequency: Vec<f32> = (0..dim)
        .step_by(2)
        .map(|i| 1. / (rope_theta.powf(i as f32 / dim as f32)))
        .collect();
    let inverse_frequency_len = inverse_frequency.len();
    Tensor::new(device, &inverse_frequency).reshape([1, inverse_frequency_len])
}

#[derive(Debug, Clone)]
pub struct RopeCache {
    sin: Tensor<2, f32>,
    cos: Tensor<2, f32>,
}

impl RopeCache {
    pub fn new(config: &QwenConfig, device: &Device) -> Result<Self> {
        let inverse_frequency = create_inverse_frequency(
            config.head_dimension,
            config.rope_theta,
            device,
        );

        let context_indices: Tensor<2, f32> =
            Tensor::arange(device, 0f32, config.context_length as f32)
                .reshape([config.context_length, 1]);

        let outer_product = context_indices.mat_mul(&inverse_frequency);

        let sin = outer_product.sin();
        let cos = outer_product.cos();

        Ok(Self { sin, cos })
    }

    /// Apply non-interleaved RoPE (Qwen style) to query and key tensors
    pub fn forward(
        &self,
        q: &Tensor<4, f32>,
        k: &Tensor<4, f32>,
        start_pos: usize,
    ) -> (Tensor<4, f32>, Tensor<4, f32>) {
        let [_b_sz, _n_head, seq_len, _n_embd] = *q.shape();
        let cos = self.cos.narrow(0, start_pos, seq_len);
        let sin = self.sin.narrow(0, start_pos, seq_len);

        let q = q.rope_normal_fused(&cos, &sin);
        let k = k.rope_normal_fused(&cos, &sin);

        (q, k)
    }
}
