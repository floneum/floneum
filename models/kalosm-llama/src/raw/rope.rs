use super::LlamaConfig;
use candle_core::IndexOp;
use candle_core::{DType, Device, Tensor, D};

#[derive(Debug, Clone)]
pub struct RopeCache {
    sin: Tensor,
    cos: Tensor,
}

impl RopeCache {
    pub fn new(config: &LlamaConfig, dtype: DType, device: &Device) -> candle_core::Result<Self> {
        let half_heads = config.rope_dimension / 2;

        let inverse_frequency = (0..config.rope_dimension)
            .step_by(2)
            .map(|i| {
                1. / (config
                    .rope_theta
                    .powf(i as f32 / config.rope_dimension as f32))
            })
            .collect::<Vec<_>>();
        let inverse_frequency = Tensor::new(inverse_frequency, device)?;

        let llama_context_length_indices =
            Tensor::arange(0f32, config.context_length as f32, device)?;

        let new_shape = (config.context_length, half_heads);

        let llama_context_length_indices = llama_context_length_indices
            .reshape((new_shape.0, 1))?
            .broadcast_as(new_shape)?;

        let inverse_frequency = inverse_frequency
            .reshape((1, new_shape.1))?
            .broadcast_as(new_shape)?;

        let outer_product = (llama_context_length_indices * inverse_frequency)?;

        let sin = outer_product.sin()?.to_dtype(dtype)?;
        let cos = outer_product.cos()?.to_dtype(dtype)?;

        Ok(Self {
            sin: Tensor::cat(&[&sin, &sin], D::Minus1)?,
            cos: Tensor::cat(&[&cos, &cos], D::Minus1)?,
        })
    }

    fn get(&self, seq_len: usize) -> candle_core::Result<(Tensor, Tensor)> {
        Ok((self.cos.i(..seq_len)?, self.sin.i(..seq_len)?))
    }

    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        seq_len: usize,
        start_pos: usize,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let (cos, sin) = self.get(seq_len)?;
        let cos = cos.i(start_pos..)?;
        let sin = sin.i(start_pos..)?;

        let q_embed = ((q * &cos)? + (rotate_half(q)? * &sin)?)?;
        let k_embed = ((k * &cos)? + (rotate_half(k)? * &sin)?)?;
        Ok((q_embed, k_embed))
    }
}

fn rotate_half(x: &Tensor) -> candle_core::Result<Tensor> {
    let last_dim = x.dim(D::Minus1)?;
    let xs1 = x.narrow(D::Minus1, 0, last_dim / 2)?;
    let xs2 = x.narrow(D::Minus1, last_dim / 2, last_dim - last_dim / 2)?;
    Tensor::cat(&[&xs2.neg()?, &xs1], D::Minus1)
}

#[test]
fn test_rope_cache() {
    let config = LlamaConfig {
        rope_theta: 5000.,
        context_length: 6,
        rope_dimension: 2,
        head_dimension: 0,
        n_head: 0,
        n_kv_head: 0,
        n_layer: 0,
    };
    let device = Device::cuda_if_available(0).unwrap();
    let cache = RopeCache::new(&config, DType::F32, &device).unwrap();

    let expected_cos = Tensor::new(
        vec![
            vec![1.0000f32, 1.0000f32],
            vec![0.5403f32, 0.5403f32],
            vec![-0.4161f32, -0.4161f32],
            vec![-0.9900f32, -0.9900f32],
            vec![-0.6536f32, -0.6536f32],
            vec![0.2837f32, 0.2837f32],
        ],
        &device,
    )
    .unwrap();
    let expected_sin = Tensor::new(
        vec![
            vec![0.0000f32, 0.0000f32],
            vec![0.8415f32, 0.8415f32],
            vec![0.9093f32, 0.9093f32],
            vec![0.1411f32, 0.1411f32],
            vec![-0.7568f32, -0.7568f32],
            vec![-0.9589f32, -0.9589f32],
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
