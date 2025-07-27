use std::f32::consts::PI;

use super::{LlamaConfig, RopeScalingConfig};
use candle_core::{shape::Dim, DType, Device, IndexOp, Result, Tensor, D};

pub(crate) fn create_inverse_frequency(
    rope_scaling: Option<&RopeScalingConfig>,
    rope_freq_weight: Option<&Tensor>,
    dtype: DType,
    dim: usize,
    rope_theta: f32,
    device: &Device,
) -> candle_core::Result<Tensor> {
    let mut inverse_frequency = (0..dim)
        .step_by(2)
        .map(|i| 1. / (rope_theta.powf(i as f32 / dim as f32)))
        .collect::<Vec<_>>();
    if let Some(scaling_config) = &rope_scaling {
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
                let smooth = (original_max_position_embeddings as f32 / wavelen - low_freq_factor)
                    / (high_freq_factor - low_freq_factor);
                *freq = (1. - smooth) * *freq / factor + smooth * *freq
            }
        }
    }
    let inverse_frequency_len = inverse_frequency.len();
    let mut inverse_frequency =
        Tensor::from_vec(inverse_frequency, (1, inverse_frequency_len), device)?.to_dtype(dtype)?;
    if let Some(weight) = &rope_freq_weight {
        inverse_frequency = inverse_frequency.mul(&weight.reshape((1, ()))?)?;
    }

    Ok(inverse_frequency)
}

#[derive(Debug, Clone)]
pub(crate) enum RopeImplementation {
    QwenVL(QwenVLRopeCache),
    Llama(RopeCache),
}

impl RopeImplementation {
    pub fn new(
        config: &LlamaConfig,
        dtype: DType,
        rope_theta: f32,
        device: &Device,
    ) -> candle_core::Result<Self> {
        if let Some(mrope_sections) = &config.mrope_sections {
            let cache = QwenVLRopeCache::new(config, dtype, rope_theta, mrope_sections, device)?;
            Ok(Self::QwenVL(cache))
        } else {
            let cache = RopeCache::new(config, dtype, rope_theta, device)?;
            Ok(Self::Llama(cache))
        }
    }

    pub fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        start_pos: usize,
        position_ids: Option<&Tensor>,
        interleaved_rope: bool,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        match self {
            Self::QwenVL(cache) => cache.forward(position_ids.unwrap(), query, key),
            Self::Llama(cache) => {
                if interleaved_rope {
                    cache.forward_i(query, key, start_pos)
                } else {
                    cache.forward(query, key, start_pos)
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct QwenVLRopeCache {
    inverse_frequency: Tensor,
    mrope_sections: Vec<usize>,
}

impl QwenVLRopeCache {
    pub fn new(
        config: &LlamaConfig,
        dtype: DType,
        rope_theta: f32,
        mrope_sections: &[usize],
        device: &Device,
    ) -> candle_core::Result<Self> {
        let inverse_frequency = create_inverse_frequency(
            config.rope_scaling.as_ref(),
            config.rope_freq_weight.as_ref(),
            dtype,
            config.head_dimension,
            rope_theta,
            device,
        )?;
        let mrope_sections = mrope_sections.to_vec();
        Ok(Self {
            inverse_frequency,
            mrope_sections,
        })
    }

    fn forward_sin_cos(&self, position_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let inv_freq_expanded = self
            .inverse_frequency
            .reshape(((),))?
            .repeat((3, 1, 1, 1))?
            .reshape((3, 1, (), 1))?;
        let position_ids_expanded = position_ids.unsqueeze(1)?.unsqueeze(1)?;
        let freqs = inv_freq_expanded
            .matmul(&position_ids_expanded.to_dtype(inv_freq_expanded.dtype())?)?
            .transpose(2, 3)?;
        let cos = freqs.cos()?;
        let sin = freqs.sin()?;

        let cos = Tensor::cat(
            &split(&cos, D::Minus1, &self.mrope_sections)?
                .iter()
                .enumerate()
                .map(|(i, m)| m.i(i % 3))
                .collect::<Result<Vec<_>>>()?,
            D::Minus1,
        )?
        .squeeze(0)?
        .contiguous()?;
        let sin = Tensor::cat(
            &split(&sin, D::Minus1, &self.mrope_sections)?
                .iter()
                .enumerate()
                .map(|(i, m)| m.i(i % 3))
                .collect::<Result<Vec<_>>>()?,
            D::Minus1,
        )?
        .squeeze(0)?
        .contiguous()?;

        Ok((cos, sin))
    }

    pub(crate) fn forward(
        &self,
        position_ids: &Tensor,
        query: &Tensor,
        key: &Tensor,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let (cos, sin) = self.forward_sin_cos(position_ids)?;
        let key = candle_nn::rotary_emb::rope(&key.contiguous()?, &cos, &sin)?;
        let query = candle_nn::rotary_emb::rope(&query.contiguous()?, &cos, &sin)?;
        Ok((query, key))
    }
}

fn split(
    tensor: &Tensor,
    dim: impl Dim + Copy,
    split_at: &[usize],
) -> candle_core::Result<Vec<Tensor>> {
    let mut result = Vec::new();
    let mut start = 0;
    for len in split_at.iter().copied() {
        let slice = tensor.narrow(dim, start, len)?;
        result.push(slice);
        start += len;
    }
    Ok(result)
}

#[derive(Debug, Clone)]
pub struct RopeCache {
    sin: Tensor,
    cos: Tensor,
}

impl RopeCache {
    pub fn new(
        config: &LlamaConfig,
        dtype: DType,
        rope_theta: f32,
        device: &Device,
    ) -> candle_core::Result<Self> {
        let inverse_frequency = create_inverse_frequency(
            config.rope_scaling.as_ref(),
            config.rope_freq_weight.as_ref(),
            dtype,
            config.head_dimension,
            rope_theta,
            device,
        )?;

        let llama_context_length_indices =
            Tensor::arange(0f32, config.context_length as f32, device)?
                .reshape((config.context_length, 1))?
                .to_dtype(dtype)?;

        let outer_product = llama_context_length_indices.matmul(&inverse_frequency)?;

        let sin = outer_product.sin()?;
        let cos = outer_product.cos()?;

        Ok(Self { sin, cos })
    }

    pub(crate) fn from_parts(cos: Tensor, sin: Tensor) -> candle_core::Result<Self> {
        let cos = cos.contiguous()?;
        let sin = sin.contiguous()?;
        Ok(Self { cos, sin })
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
                        .map_err(|e| candle_core::Error::Msg(format!("Error in q: {e:?}")))??,
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

    pub(crate) fn sin(&self) -> &Tensor {
        &self.sin
    }

    pub(crate) fn cos(&self) -> &Tensor {
        &self.cos
    }
}

#[test]
fn test_rope_cache() {
    let config = LlamaConfig::mock_test();
    let device = Device::cuda_if_available(0).unwrap();
    let cache = RopeCache::new(&config, DType::F32, config.rope_theta, &device).unwrap();

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
