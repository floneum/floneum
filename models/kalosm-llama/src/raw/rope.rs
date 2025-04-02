use std::f32::consts::PI;

use super::LlamaConfig;
use fusor_core::{Device, Tensor};

#[derive(Clone)]
pub struct RopeCache {
    sin: Tensor<2, f32>,
    cos: Tensor<2, f32>,
}

impl RopeCache {
    pub fn new(config: &LlamaConfig, device: &Device) -> Self {
        let mut inverse_frequency = (0..config.head_dimension)
            .step_by(2)
            .map(|i| {
                1. / (config
                    .rope_theta
                    .powf(i as f32 / config.head_dimension as f32))
            })
            .collect::<Vec<_>>();
        if let Some(scaling_config) = &config.rope_scaling {
            let original_max_position_embeddings = scaling_config.original_max_position_embeddings;
            let factor = scaling_config.factor;
            let high_freq_factor = scaling_config.high_freq_factor;
            let low_freq_factor = scaling_config.low_freq_factor;
            let low_freq_wavelen = original_max_position_embeddings as f32 / low_freq_factor;
            let high_freq_wavelen = original_max_position_embeddings as f32 / high_freq_factor;
            for freq in inverse_frequency.iter_mut() {
                let wavelen = 2. * PI / *freq;
                if wavelen > low_freq_wavelen {
                    *freq /= factor
                } else if wavelen == high_freq_wavelen {
                    let smooth = (original_max_position_embeddings as f32 / wavelen
                        - low_freq_factor)
                        / (high_freq_factor - low_freq_factor);
                    *freq = (1. - smooth) * *freq / factor + smooth * *freq
                }
            }
        }
        let inverse_frequency_len = inverse_frequency.len();
        let mut inverse_frequency =
            Tensor::new(device, &inverse_frequency).reshape([1, inverse_frequency_len]);
        if let Some(weight) = &config.rope_freq_weight {
            inverse_frequency = inverse_frequency * weight.reshape([1, inverse_frequency_len]);
        }

        let llama_context_length_indices =
            Tensor::arange(device, 0f32, config.context_length as f32)
                .reshape([config.context_length, 1]);

        let outer_product = llama_context_length_indices.mat_mul(&inverse_frequency);

        let sin = outer_product.sin();
        let cos = outer_product.cos();

        Self { sin, cos }
    }

    fn forward_with_embed(
        &self,
        q: Tensor<3, f32>,
        k: Tensor<3, f32>,
        start_pos: usize,
        apply_rotary_emb: fn(Tensor<3, f32>, Tensor<2, f32>, Tensor<2, f32>) -> Tensor<3, f32>,
    ) -> (Tensor<3, f32>, Tensor<3, f32>) {
        let apply_rotary_emb =
            |sin: &Tensor<2, f32>, cos: &Tensor<2, f32>, x: Tensor<3, f32>, index_pos| {
                let [_n_head, seq_len, _n_embd] = *x.shape();
                let cos = cos.narrow(0, index_pos, seq_len);
                let sin = sin.narrow(0, index_pos, seq_len);
                apply_rotary_emb(x, cos, sin)
            };

        let q = apply_rotary_emb(&self.sin, &self.cos, q, start_pos);
        let k = apply_rotary_emb(&self.sin, &self.cos, k, start_pos);
        (q, k)
    }

    pub fn forward(
        &self,
        q: Tensor<3, f32>,
        k: Tensor<3, f32>,
        start_pos: usize,
    ) -> (Tensor<3, f32>, Tensor<3, f32>) {
        self.forward_with_embed(q, k, start_pos, Tensor::rope)
    }

    pub fn forward_i(
        &self,
        q: Tensor<3, f32>,
        k: Tensor<3, f32>,
        start_pos: usize,
    ) -> (Tensor<3, f32>, Tensor<3, f32>) {
        todo!()
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_rope_cache() {
    use fusor_core::Sum;

    let config = LlamaConfig::mock_test();
    let device = Device::new().await.unwrap();
    let cache = RopeCache::new(&config, &device);

    println!("cache cos: {:?}", cache.cos.as_slice().await.unwrap());
    println!("cache sin: {:?}", cache.sin.as_slice().await.unwrap());

    let expected_cos = Tensor::new(
        &device,
        vec![
            &[1.0000f32],
            &[0.5403f32],
            &[-0.4161f32],
            &[-0.9900f32],
            &[-0.6536f32],
            &[0.2837f32],
        ],
    );
    let expected_sin = Tensor::new(
        &device,
        vec![
            &[0.0000f32],
            &[0.8415f32],
            &[0.9093f32],
            &[0.1411f32],
            &[-0.7568f32],
            &[-0.9589f32],
        ],
    );

    let cos_difference = (cache.cos - expected_cos).abs();
    println!(
        "cos_difference: {:?}",
        cos_difference.as_slice().await.unwrap()
    );

    let cos_error: f32 = cos_difference.sum(0).sum(0).as_slice().await.unwrap()[[]];
    assert!(cos_error < 1e-2);

    let sin_difference = (cache.sin - expected_sin).abs();
    println!(
        "sin_difference: {:?}",
        sin_difference.as_slice().await.unwrap()
    );
    let sin_error: f32 = sin_difference.sum(0).sum(0).as_slice().await.unwrap()[[]];
    assert!(sin_error < 1e-2);
}
