
use candle_core::{Device, Result, Tensor, D};

use crate::raw::rope::create_inverse_frequency;

struct VisionRotaryEmbedding {
    inv_freq: Tensor,
}

impl VisionRotaryEmbedding {
    fn new(dim: usize, rope_theta: f32, device: &Device) -> Result<Self> {
        Ok(Self {
            inv_freq: create_inverse_frequency(
                None,
                None,
                candle_core::DType::F32,
                dim,
                rope_theta,
                device,
            )?,
        })
    }

    fn make_embeds(&self, sequence_length: usize) -> Result<Tensor> {
        let seq =
            Tensor::arange(0f32, sequence_length as f32, self.inv_freq.device())?.unsqueeze(D::Minus1)?;
        seq.broadcast_matmul(&self.inv_freq)
    }
}
