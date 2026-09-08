use std::f32::consts::PI;

use super::LlamaConfig;
use candle_core::{DType, Device, Tensor};

#[derive(Debug, Clone)]
pub struct RopeCache {
    sin: Tensor,
    cos: Tensor,
}

impl RopeCache {
    pub fn new(config: &LlamaConfig, dtype: DType, device: &Device) -> candle_core::Result<Self> {
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
            Tensor::from_vec(inverse_frequency, (1, inverse_frequency_len), device)?
                .to_dtype(dtype)?;
        if let Some(weight) = &config.rope_freq_weight {
            inverse_frequency = inverse_frequency.mul(&weight.reshape((1, ()))?)?;
        }

        let llama_context_length_indices =
            Tensor::arange(0f32, config.context_length as f32, device)?
                .reshape((config.context_length, 1))?
                .to_dtype(dtype)?;

        let outer_product = llama_context_length_indices.matmul(&inverse_frequency)?;

        let sin = outer_product.sin()?;
        let cos = outer_product.cos()?;

        Ok(Self { sin, cos })
    }

    fn forward_with_embed(
        &self,
        q: &Tensor,
        k: &Tensor,
        start_pos: usize,
        apply_rotary_emb: fn(&Tensor, &Tensor, &Tensor) -> candle_core::Result<Tensor>,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let apply_rotary_emb = |sin: &Tensor, cos: &Tensor, x: &Tensor, index_pos| {
            let (_b_sz, _n_head, seq_len, _n_embd) = x.dims4()?;
            let cos = cos.narrow(0, index_pos, seq_len)?;
            let sin = sin.narrow(0, index_pos, seq_len)?;
            apply_rotary_emb(&x.contiguous()?, &cos, &sin)
        };
        let device = q.device();
        let (q, k) = if matches!(device, Device::Cpu) {
            std::thread::scope(|s| {
                let q = s.spawn(|| apply_rotary_emb(&self.sin, &self.cos, q, start_pos));
                let k = apply_rotary_emb(&self.sin, &self.cos, k, start_pos)?;
                candle_core::Result::Ok((
                    q.join()
                        .map_err(|e| candle_core::Error::Msg(format!("Error in q: {:?}", e)))??,
                    k,
                ))
            })?
        } else {
            let q = apply_rotary_emb(&self.sin, &self.cos, q, start_pos)?;
            let k = apply_rotary_emb(&self.sin, &self.cos, k, start_pos)?;
            (q, k)
        };

        Ok((q, k))
    }

    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        start_pos: usize,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        self.forward_with_embed(q, k, start_pos, candle_nn::rotary_emb::rope)
    }

    pub fn forward_i(
        &self,
        q: &Tensor,
        k: &Tensor,
        start_pos: usize,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        self.forward_with_embed(q, k, start_pos, candle_nn::rotary_emb::rope_i)
    }
}

#[test]
fn test_rope_cache() {
    let config = LlamaConfig::mock_test();
    let device = Device::cuda_if_available(0).unwrap();
    let cache = RopeCache::new(&config, DType::F32, &device).unwrap();

    let expected_cos = Tensor::new(
        vec![
            vec![1.0000f32],
            vec![0.5403f32],
            vec![-0.4161f32],
            vec![-0.9900f32],
            vec![-0.6536f32],
            vec![0.2837f32],
        ],
        &device,
    )
    .unwrap();
    let expected_sin = Tensor::new(
        vec![
            vec![0.0000f32],
            vec![0.8415f32],
            vec![0.9093f32],
            vec![0.1411f32],
            vec![-0.7568f32],
            vec![-0.9589f32],
        ],
        &device,
    )
    .unwrap();

    let cos_error: f32 = (cache.cos - expected_cos)
        .unwrap()
        .abs()
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar()
        .unwrap();
    assert!(cos_error < 1e-2);
    let sin_error: f32 = (cache.sin - expected_sin)
        .unwrap()
        .abs()
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar()
        .unwrap();
    assert!(sin_error < 1e-2);
}
