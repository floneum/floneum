//! Attention mask implementation.

use crate::{ConcreteTensor, Device, GpuOr, SimdElement};
use fusor_core::FloatDataType;

/// Attention mask for causal (decoder) attention
///
/// Prevents attending to future positions
#[derive(Clone)]
pub struct AttentionMask<D: SimdElement> {
    mask: GpuOr<2, D, ConcreteTensor<D, 2>>,
}

impl<D: SimdElement + FloatDataType + Default> AttentionMask<D>
where
    crate::AddOp: fusor_cpu::SimdBinaryOp<D>,
{
    /// Create a new attention mask
    pub fn new(mask: GpuOr<2, D, ConcreteTensor<D, 2>>) -> Self {
        Self { mask }
    }

    /// Create a causal mask for the given sequence length
    ///
    /// mask[i, j] = -inf if j > i (can't attend to future), 0 otherwise
    pub fn causal(device: &Device, seq_len: usize) -> Self {
        // Create a lower triangular matrix of 0s and upper triangular of -inf
        let mut mask_data = vec![D::zero(); seq_len * seq_len];
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                mask_data[i * seq_len + j] = D::from_f32(f32::NEG_INFINITY);
            }
        }

        let mask: GpuOr<2, D> = match device {
            Device::Cpu => GpuOr::Cpu(fusor_cpu::Tensor::from_slice([seq_len, seq_len], &mask_data)),
            Device::Gpu(gpu) => {
                let data_chunks: Vec<&[D]> = mask_data.chunks(seq_len).collect();
                GpuOr::Gpu(fusor_core::Tensor::new(gpu, data_chunks))
            }
        };
        Self::new(mask)
    }

    /// Apply the mask to attention scores
    ///
    /// attention_scores: (batch, heads, seq_len, seq_len) or similar ranks
    /// Returns: masked attention scores
    ///
    /// The mask will be broadcast to match the attention scores shape
    pub fn apply<const R: usize>(
        &self,
        attention_scores: &GpuOr<R, D>,
    ) -> GpuOr<R, D>
    where
        D: std::ops::Add<Output = D>,
        (fusor_core::Tensor<2, D>, fusor_core::Tensor<R, D>): fusor_core::MaxRank<R, D>,
    {
        // Broadcast the mask to match the attention scores shape
        let mask_broadcast: GpuOr<R, D, _> = self.mask.broadcast_as(attention_scores.shape());
        match (&mask_broadcast, attention_scores) {
            (GpuOr::Cpu(m), GpuOr::Cpu(a)) => GpuOr::Cpu((m + a).eval()),
            (GpuOr::Gpu(m), GpuOr::Gpu(a)) => GpuOr::Gpu(m.add_(a)),
            _ => panic!("Cannot mix CPU and GPU tensors"),
        }
    }

    pub fn forward<const R: usize>(
        &self,
        attention_scores: &mut GpuOr<R, D>,
    ) where
        D: std::ops::Add<Output = D>,
        (fusor_core::Tensor<2, D>, fusor_core::Tensor<R, D>): fusor_core::MaxRank<R, D>,
    {
        *attention_scores = self.apply(attention_scores);
    }

    pub fn mask(&self) -> &GpuOr<2, D, ConcreteTensor<D, 2>> {
        &self.mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_attention_mask_causal() {
        let device = Device::cpu();

        let seq_len = 3;
        let mask: AttentionMask<f32> = AttentionMask::causal(&device, seq_len);

        assert_eq!(mask.mask().shape(), [3, 3]);

        let mask_data = mask.mask().clone().as_slice().await.unwrap();

        // Lower triangular should be 0, upper triangular should be -inf
        assert_eq!(mask_data[[0, 0]], 0.0);
        assert_eq!(mask_data[[0, 1]], f32::NEG_INFINITY);
        assert_eq!(mask_data[[0, 2]], f32::NEG_INFINITY);

        assert_eq!(mask_data[[1, 0]], 0.0);
        assert_eq!(mask_data[[1, 1]], 0.0);
        assert_eq!(mask_data[[1, 2]], f32::NEG_INFINITY);

        assert_eq!(mask_data[[2, 0]], 0.0);
        assert_eq!(mask_data[[2, 1]], 0.0);
        assert_eq!(mask_data[[2, 2]], 0.0);
    }

    #[tokio::test]
    async fn test_attention_mask_apply_4d() {
        let device = Device::cpu();

        let mask: AttentionMask<f32> = AttentionMask::causal(&device, 2);

        // Create attention scores: (1, 1, 2, 2)
        let scores_data = [1.0f32, 2.0, 3.0, 4.0];
        let scores: GpuOr<4, f32> =
            GpuOr::Cpu(fusor_cpu::Tensor::from_slice([1, 1, 2, 2], &scores_data));

        let masked = mask.apply(&scores);

        let output = masked.as_slice().await.unwrap();

        // [0][0] = 1.0 + 0 = 1.0
        // [0][1] = 2.0 + -inf = -inf
        // [1][0] = 3.0 + 0 = 3.0
        // [1][1] = 4.0 + 0 = 4.0
        assert_eq!(output[[0, 0, 0, 0]], 1.0);
        assert_eq!(output[[0, 0, 0, 1]], f32::NEG_INFINITY);
        assert_eq!(output[[0, 0, 1, 0]], 3.0);
        assert_eq!(output[[0, 0, 1, 1]], 4.0);
    }

    #[tokio::test]
    async fn test_attention_mask_apply_3d() {
        let device = Device::cpu();

        let mask: AttentionMask<f32> = AttentionMask::causal(&device, 2);

        // Create attention scores: (1, 2, 2)
        let scores_data = [1.0f32, 2.0, 3.0, 4.0];
        let scores: GpuOr<3, f32> =
            GpuOr::Cpu(fusor_cpu::Tensor::from_slice([1, 2, 2], &scores_data));

        let masked = mask.apply(&scores);

        let output = masked.as_slice().await.unwrap();

        assert_eq!(output[[0, 0, 0]], 1.0);
        assert_eq!(output[[0, 0, 1]], f32::NEG_INFINITY);
        assert_eq!(output[[0, 1, 0]], 3.0);
        assert_eq!(output[[0, 1, 1]], 4.0);
    }
}
