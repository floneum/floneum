use super::{LlamaConfig, RopeScalingConfig};
use fusor_core::{CastTensor, DataType, Device, Dim, FloatDataType, Tensor, D};
use std::f32::consts::PI;

pub(crate) fn create_inverse_frequency<F: FloatDataType>(
    rope_scaling: Option<&RopeScalingConfig>,
    rope_freq_weight: Option<&Tensor<1, F>>,
    dim: usize,
    rope_theta: f32,
    device: &Device,
) -> Tensor<2, F>
where
    f32: CastTensor<F>,
{
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
    let mut inverse_frequency: Tensor<2, F> = Tensor::new(device, &inverse_frequency)
        .reshape([1, inverse_frequency_len])
        .cast();
    if let Some(weight) = &rope_freq_weight {
        inverse_frequency = inverse_frequency * weight.reshape((1, ()));
    }

    inverse_frequency
}

#[derive(Debug, Clone)]
pub(crate) enum RopeImplementation<F: FloatDataType = f32> {
    QwenVL(QwenVLRopeCache<F>),
    Llama(RopeCache<F>),
}

impl<F: FloatDataType> RopeImplementation<F>
where
    f32: CastTensor<F>,
{
    pub fn new(
        config: &LlamaConfig<F>,
        rope_theta: f32,
        device: &Device,
    ) -> fusor_core::Result<Self> {
        if let Some(mrope_sections) = &config.mrope_sections {
            let cache = QwenVLRopeCache::new(config, rope_theta, mrope_sections, device)?;
            Ok(Self::QwenVL(cache))
        } else {
            let cache = RopeCache::new(config, rope_theta, device)?;
            Ok(Self::Llama(cache))
        }
    }

    pub fn forward(
        &self,
        query: &Tensor<4, F>,
        key: &Tensor<4, F>,
        start_pos: usize,
        position_ids: Option<&Tensor<2, F>>,
        interleaved_rope: bool,
    ) -> (Tensor<4, F>, Tensor<4, F>) {
        match self {
            Self::QwenVL(cache) => cache.forward(
                position_ids.expect("qwen vl requires position ids"),
                query,
                key,
            ),
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
pub(crate) struct QwenVLRopeCache<F: FloatDataType = f32> {
    inverse_frequency: Tensor<2, F>,
    mrope_sections: Vec<usize>,
}

impl<F: FloatDataType> QwenVLRopeCache<F>
where
    f32: CastTensor<F>,
{
    pub fn new(
        config: &LlamaConfig<F>,
        rope_theta: f32,
        mrope_sections: &[usize],
        device: &Device,
    ) -> fusor_core::Result<Self> {
        let inverse_frequency = create_inverse_frequency(
            config.rope_scaling.as_ref(),
            config.rope_freq_weight.as_ref(),
            config.head_dimension,
            rope_theta,
            device,
        );
        let mrope_sections = mrope_sections.to_vec();
        Ok(Self {
            inverse_frequency,
            mrope_sections,
        })
    }

    fn forward_sin_cos(&self, position_ids: &Tensor<2, F>) -> (Tensor<2, F>, Tensor<2, F>) {
        let inv_freq_expanded =
            self.inverse_frequency
                .reshape(((),))
                .repeat([3])
                .reshape((3, 1, (), 1));
        let position_ids_expanded = position_ids.unsqueeze(1).unsqueeze(1);
        let freqs = inv_freq_expanded
            .mat_mul(&position_ids_expanded)
            .transpose(2, 3);
        let cos = freqs.cos();
        let sin = freqs.sin();

        let cos = Tensor::cat(
            split(&cos, D::Minus1, &self.mrope_sections)
                .iter()
                .enumerate()
                .map(|(i, m)| m.i((i % 3, .., .., ..)))
                .collect::<Vec<_>>(),
            D::Minus1,
        )
        .squeeze(0);
        let sin = Tensor::cat(
            split(&sin, D::Minus1, &self.mrope_sections)
                .iter()
                .enumerate()
                .map(|(i, m)| m.i((i % 3, .., .., ..)))
                .collect::<Vec<_>>(),
            D::Minus1,
        )
        .squeeze(0);

        (cos, sin)
    }

    pub(crate) fn forward(
        &self,
        position_ids: &Tensor<2, F>,
        query: &Tensor<4, F>,
        key: &Tensor<4, F>,
    ) -> (Tensor<4, F>, Tensor<4, F>) {
        let (cos, sin) = self.forward_sin_cos(position_ids);
        let key = key.rope(&cos, &sin);
        let query = query.rope(&cos, &sin);
        (query, key)
    }
}

fn split<const R: usize, T: DataType>(
    tensor: &Tensor<R, T>,
    dim: impl Dim<R> + Copy,
    split_at: &[usize],
) -> Vec<Tensor<R, T>> {
    let mut result = Vec::new();
    let mut start = 0;
    for len in split_at.iter().copied() {
        let slice = tensor.narrow(dim, start, len);
        result.push(slice);
        start += len;
    }
    result
}

#[derive(Debug, Clone)]
pub struct RopeCache<F: FloatDataType = f32> {
    sin: Tensor<2, F>,
    cos: Tensor<2, F>,
}

impl<F: FloatDataType> RopeCache<F>
where
    f32: CastTensor<F>,
{
    pub fn new(
        config: &LlamaConfig<F>,
        rope_theta: f32,
        device: &Device,
    ) -> fusor_core::Result<Self> {
        let inverse_frequency: Tensor<2, F> = create_inverse_frequency(
            config.rope_scaling.as_ref(),
            config.rope_freq_weight.as_ref(),
            config.head_dimension,
            rope_theta,
            device,
        );

        let llama_context_length_indices: Tensor<2, F> =
            Tensor::arange(device, 0f32, config.context_length as f32)
                .reshape([config.context_length, 1])
                .cast();

        let outer_product = llama_context_length_indices.mat_mul(&inverse_frequency);

        let sin = outer_product.sin();
        let cos = outer_product.cos();

        Ok(Self { sin, cos })
    }

    pub(crate) fn from_parts(cos: Tensor<2, F>, sin: Tensor<2, F>) -> fusor_core::Result<Self> {
        Ok(Self { cos, sin })
    }

    fn forward_with_embed(
        &self,
        q: &Tensor<4, F>,
        k: &Tensor<4, F>,
        start_pos: usize,
        apply_rotary_emb: fn(&Tensor<4, F>, &Tensor<2, F>, &Tensor<2, F>) -> Tensor<4, F>,
    ) -> (Tensor<4, F>, Tensor<4, F>) {
        let apply_rotary_emb =
            |sin: &Tensor<2, F>, cos: &Tensor<2, F>, x: &Tensor<4, F>, index_pos| {
                let [_b_sz, _n_head, seq_len, _n_embd] = *x.shape();
                let cos = cos.narrow(0, index_pos, seq_len);
                let sin = sin.narrow(0, index_pos, seq_len);
                apply_rotary_emb(&x, &cos, &sin)
            };
        let q = apply_rotary_emb(&self.sin, &self.cos, q, start_pos);
        let k = apply_rotary_emb(&self.sin, &self.cos, k, start_pos);

        (q, k)
    }

    pub fn forward(
        &self,
        q: &Tensor<4, F>,
        k: &Tensor<4, F>,
        start_pos: usize,
    ) -> (Tensor<4, F>, Tensor<4, F>) {
        self.forward_with_embed(q, k, start_pos, Tensor::rope_normal_fused)
    }

    pub fn forward_i(
        &self,
        q: &Tensor<4, F>,
        k: &Tensor<4, F>,
        start_pos: usize,
    ) -> (Tensor<4, F>, Tensor<4, F>) {
        self.forward_with_embed(q, k, start_pos, Tensor::rope_fused)
    }

    pub(crate) fn sin(&self) -> &Tensor<2, F> {
        &self.sin
    }

    pub(crate) fn cos(&self) -> &Tensor<2, F> {
        &self.cos
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_rope_cache() {
    use fusor_core::{Device, Tensor};

    let config: LlamaConfig<f32> = LlamaConfig::mock_test();
    let device = Device::new().await.unwrap();
    let cache: RopeCache<f32> = RopeCache::new(&config, config.rope_theta, &device).unwrap();

    let expected_cos = Tensor::new(
        &device,
        &[
            [1.0000f32],
            [0.5403f32],
            [-0.4161f32],
            [-0.9900f32],
            [-0.6536f32],
            [0.2837f32],
        ],
    );
    let expected_sin = Tensor::new(
        &device,
        &[
            [0.0000f32],
            [0.8415f32],
            [0.9093f32],
            [0.1411f32],
            [-0.7568f32],
            [-0.9589f32],
        ],
    );

    let cos_error: f32 = (cache.cos - expected_cos)
        .abs()
        .sum(0)
        .sum(0)
        .to_scalar()
        .await
        .unwrap();
    assert!(cos_error < 1e-2);
    let sin_error: f32 = (cache.sin - expected_sin)
        .abs()
        .sum(0)
        .sum(0)
        .to_scalar()
        .await
        .unwrap();
    assert!(sin_error < 1e-2);
}
