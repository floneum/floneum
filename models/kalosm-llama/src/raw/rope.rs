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
        let inverse_frequency_len = inverse_frequency.len();
        let inverse_frequency =
            Tensor::from_vec(inverse_frequency, (1, inverse_frequency_len), device)?.to_dtype(dtype)?;

        let llama_context_length_indices =
            Tensor::arange(0f32, config.context_length as f32, device)?
                .reshape((config.context_length, 1))?.to_dtype(dtype)?;

        let outer_product = llama_context_length_indices.matmul(&inverse_frequency)?;

        let sin = outer_product.sin()?;
        let cos = outer_product.cos()?;

        Ok(Self { sin, cos })
    }

    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        start_pos: usize,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        fn apply_rotary_emb(sin: &Tensor, cos: &Tensor, x: &Tensor, index_pos: usize) -> candle_core::Result<Tensor> {
            let (b_sz, n_head, seq_len, n_embd) = x.dims4()?;
            let cos = cos
                .narrow(0, index_pos, seq_len)?
                .reshape((seq_len, n_embd / 2, 1))?;
            let sin = sin
                .narrow(0, index_pos, seq_len)?
                .reshape((seq_len, n_embd / 2, 1))?;
            let cos = cos.broadcast_as((b_sz, 1, seq_len, n_embd / 2, 1))?;
            let sin = sin.broadcast_as((b_sz, 1, seq_len, n_embd / 2, 1))?;
            // This mimics the llama.cpp behavior.
            // https://github.com/ggerganov/llama.cpp/blob/1f0bccb27929e261744c979bc75114955da49e98/ggml.c#L12104-L12105
            // The x0 and x1 value are interleaved on the n_embd (= head_dim) dimension.
            // The resulting y0 and y1 are also interleaved with:
            //   y0 = x0*cos - x1*sin
            //   y1 = x0*sin + x1*cos
            let x = x.reshape((b_sz, n_head, seq_len, n_embd / 2, 2))?;
            let x0 = x.narrow(D::Minus1, 0, 1)?;
            let x1 = x.narrow(D::Minus1, 1, 1)?;
            let y0 = (x0.broadcast_mul(&cos)? - x1.broadcast_mul(&sin)?)?;
            let y1 = (x0.broadcast_mul(&sin)? + x1.broadcast_mul(&cos)?)?;
            let rope = Tensor::cat(&[y0, y1], D::Minus1)?;
            let rope = rope.flatten_from(D::Minus2)?;
            Ok(rope)
        }
        
        let q = apply_rotary_emb(&self.sin , &self.cos, q, start_pos)?;
        let k = apply_rotary_emb(&self.sin , &self.cos, k, start_pos)?;
        Ok((q, k))
    }
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
