use super::LlamaConfig;
use candle_core::{DType, Device, Tensor, D};

#[derive(Debug, Clone)]
pub struct RopeCache {
    sin: Tensor,
    cos: Tensor,
}

impl RopeCache {
    pub fn new(config: &LlamaConfig, dtype: DType, device: &Device) -> candle_core::Result<Self> {
        let inverse_frequency = (0..config.head_dimension)
            .step_by(2)
            .map(|i| {
                1. / (config
                    .rope_theta
                    .powf(i as f32 / config.head_dimension as f32))
            })
            .collect::<Vec<_>>();
        let inverse_frequency = Tensor::new(inverse_frequency, device)?;

        let llama_context_length_indices =
            Tensor::arange(0f32, config.context_length as f32, device)?;

        let new_shape = (
            llama_context_length_indices.dim(D::Minus1)?,
            inverse_frequency.dim(D::Minus1)?,
        );

        let llama_context_length_indices = llama_context_length_indices
            .reshape((new_shape.0, 1))
            .unwrap()
            .broadcast_as(new_shape)
            .unwrap();

        let inverse_frequency = inverse_frequency
            .reshape((1, new_shape.1))
            .unwrap()
            .broadcast_as(new_shape)
            .unwrap();

        let outer_product = (llama_context_length_indices * inverse_frequency)?;

        let outer_product = Tensor::cat(&[&outer_product, &outer_product], D::Minus1)?.to_dtype(dtype)?;
        let sin = outer_product.sin()?;
        let cos = outer_product.cos()?;

        Ok(Self { sin, cos })
    }

    fn get(
        &self,
        index_pos: usize,
        seq_len: usize,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let cos = self
            .cos
            .narrow(0, index_pos, seq_len)
            .unwrap().unsqueeze(0)?.unsqueeze(0)?;
        let sin = self
            .sin
            .narrow(0, index_pos, seq_len)
            .unwrap().unsqueeze(0)?.unsqueeze(0)?;
        
        Ok((cos, sin))
    }

    fn apply_rotary_emb(cos: &Tensor, sin: &Tensor, x: &Tensor) -> candle_core::Result<Tensor> {
        x.broadcast_mul(cos)? + rotate_half(x)?.broadcast_mul(sin)
    }

    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        start_pos: usize,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let (_, _n_head, seq_len, _) = q.dims4().unwrap();
        let (cos, sin) = self.get(start_pos, seq_len)?;
        let q = {
            let cos = cos.clone();
            let sin = sin.clone();
            let q = q.clone();
            std::thread::spawn(
            move || Self::apply_rotary_emb(&cos, &sin, &q))
        
            };
        let k = Self::apply_rotary_emb(&cos, &sin, k)?;
        Ok((q.join().map_err(
            |_| candle_core::Error::Msg("Error in rotary emb thread".to_string())
        )??, k))
    }
}

fn rotate_half(xs: &Tensor) -> candle_core::Result<Tensor> {
    let last_dim = xs.dim(D::Minus1)?;
    let xs1 = xs.narrow(D::Minus1, 0, last_dim / 2)?;
    let xs2 = xs.narrow(D::Minus1, last_dim / 2, last_dim - last_dim / 2)?;
    Tensor::cat(&[&xs2.neg()?, &xs1], D::Minus1)
}

#[test]
fn test_rope_cache() {
    let config = LlamaConfig {
        rope_theta: 5000.,
        context_length: 6,
        rope_dimension: 2,
        head_dimension: 2,
        n_head: 0,
        n_kv_head: 0,
        n_layer: 0,
    };
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
