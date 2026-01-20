//! Normalization operations that work on both CPU and GPU backends.

use crate::{
    AddOp, ConcreteTensor, DivOp, ExpOp, FloatOps, GpuOr, MulOp, SimdBinaryOp, SimdElement,
    SimdUnaryOp, SqrtOp, SubOp,
};
use fusor_core::{DataType, FloatDataType, LastRank as GpuLastRank, NextRankInner as GpuNextRankInner};
use fusor_cpu::{LastRank as CpuLastRank, MaxOp, SimdReduceOp, SumOp};

impl<const R: usize, D> GpuOr<R, D>
where
    D: SimdElement + DataType + FloatDataType + FloatOps + Default,
{
    /// Softmax along a specific axis.
    ///
    /// softmax(x)_i = exp(x_i - max(x)) / sum(exp(x - max(x)))
    ///
    /// The subtraction of max(x) is for numerical stability.
    pub fn softmax<const OUT_RANK: usize>(&self, axis: usize) -> Self
    where
        ConcreteTensor<D, R>: CpuLastRank<OUT_RANK, D>,
        fusor_core::Tensor<R, D>: GpuLastRank<OUT_RANK, D>,
        <fusor_core::Tensor<R, D> as fusor_core::LastRankInner>::LastRank:
            GpuNextRankInner<NextRank = fusor_core::Tensor<R, D>>,
        MaxOp: SimdReduceOp<D>,
        SumOp: SimdReduceOp<D>,
        D: std::ops::Sub<Output = D> + std::ops::Div<Output = D>,
        SubOp: SimdBinaryOp<D>,
        DivOp: SimdBinaryOp<D>,
        ExpOp: SimdUnaryOp<D>,
    {
        // max(x) with keepdim for broadcasting
        let max_val = self.max_keepdim::<OUT_RANK>(axis);

        // x - max(x) (broadcasts automatically since max has size 1 in reduced dim)
        let shifted = match (self, &max_val) {
            (GpuOr::Cpu(a), GpuOr::Cpu(b)) => GpuOr::Cpu((a - b).eval()),
            (GpuOr::Gpu(a), GpuOr::Gpu(b)) => GpuOr::Gpu(a - b),
            _ => panic!("Cannot mix CPU and GPU tensors"),
        };

        // exp(x - max(x))
        let exp_val = shifted.exp();

        // sum(exp(...)) with keepdim
        let sum_exp = exp_val.sum_keepdim::<OUT_RANK>(axis);

        // exp / sum (broadcasts)
        match (&exp_val, &sum_exp) {
            (GpuOr::Cpu(a), GpuOr::Cpu(b)) => GpuOr::Cpu((a / b).eval()),
            (GpuOr::Gpu(a), GpuOr::Gpu(b)) => GpuOr::Gpu(a / b),
            _ => panic!("Cannot mix CPU and GPU tensors"),
        }
    }

    /// RMS Normalization along the last axis.
    ///
    /// rms_norm(x) = x / sqrt(mean(x^2) + eps) * weight
    ///
    /// Note: This is a simplified implementation that assumes weight has the same
    /// rank as input. For more complex broadcasting, use the GPU's optimized kernels directly.
    pub fn rms_norm<const OUT_RANK: usize>(&self, weight: &Self, eps: D) -> Self
    where
        ConcreteTensor<D, R>: CpuLastRank<OUT_RANK, D>,
        fusor_core::Tensor<R, D>: GpuLastRank<OUT_RANK, D>,
        <fusor_core::Tensor<R, D> as fusor_core::LastRankInner>::LastRank:
            GpuNextRankInner<NextRank = fusor_core::Tensor<R, D>>,
        SumOp: SimdReduceOp<D>,
        D: std::ops::Mul<Output = D>
            + std::ops::Div<Output = D>
            + std::ops::Add<Output = D>,
        MulOp: SimdBinaryOp<D>,
        DivOp: SimdBinaryOp<D>,
        AddOp: SimdBinaryOp<D>,
        SqrtOp: SimdUnaryOp<D>,
    {
        let axis = R - 1; // Normalize along last axis

        // x^2
        let x_sq = self.sqr();

        // mean(x^2) with keepdim along last axis
        let mean_sq = x_sq.mean_keepdim::<OUT_RANK>(axis);

        // mean(x^2) + eps
        let mean_sq_eps = mean_sq.add_scalar(eps);

        // sqrt(mean(x^2) + eps)
        let rms = mean_sq_eps.sqrt();

        // x / rms
        let normalized = match (self, &rms) {
            (GpuOr::Cpu(a), GpuOr::Cpu(b)) => GpuOr::Cpu((a / b).eval()),
            (GpuOr::Gpu(a), GpuOr::Gpu(b)) => GpuOr::Gpu(a / b),
            _ => panic!("Cannot mix CPU and GPU tensors"),
        };

        // normalized * weight
        match (&normalized, weight) {
            (GpuOr::Cpu(a), GpuOr::Cpu(b)) => GpuOr::Cpu((a * b).eval()),
            (GpuOr::Gpu(a), GpuOr::Gpu(b)) => GpuOr::Gpu(a * b),
            _ => panic!("Cannot mix CPU and GPU tensors"),
        }
    }

    /// Layer Normalization along the last axis.
    ///
    /// layer_norm(x) = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias
    ///
    /// If remove_mean is false, skips the mean subtraction (becomes RMS-like).
    ///
    /// Note: This is a simplified implementation that assumes weight and bias have
    /// the same rank as input.
    pub fn layer_norm<const OUT_RANK: usize>(
        &self,
        weight: &Self,
        bias: Option<&Self>,
        eps: D,
        remove_mean: bool,
    ) -> Self
    where
        ConcreteTensor<D, R>: CpuLastRank<OUT_RANK, D>,
        fusor_core::Tensor<R, D>: GpuLastRank<OUT_RANK, D>,
        <fusor_core::Tensor<R, D> as fusor_core::LastRankInner>::LastRank:
            GpuNextRankInner<NextRank = fusor_core::Tensor<R, D>>,
        SumOp: SimdReduceOp<D>,
        D: std::ops::Mul<Output = D>
            + std::ops::Div<Output = D>
            + std::ops::Add<Output = D>
            + std::ops::Sub<Output = D>,
        MulOp: SimdBinaryOp<D>,
        DivOp: SimdBinaryOp<D>,
        AddOp: SimdBinaryOp<D>,
        SubOp: SimdBinaryOp<D>,
        SqrtOp: SimdUnaryOp<D>,
    {
        let axis = R - 1;

        // Optionally subtract mean
        let centered = if remove_mean {
            let mean = self.mean_keepdim::<OUT_RANK>(axis);
            match (self, &mean) {
                (GpuOr::Cpu(a), GpuOr::Cpu(b)) => GpuOr::Cpu((a - b).eval()),
                (GpuOr::Gpu(a), GpuOr::Gpu(b)) => GpuOr::Gpu(a - b),
                _ => panic!("Cannot mix CPU and GPU tensors"),
            }
        } else {
            self.clone()
        };

        // Compute variance: mean(centered^2)
        let centered_sq = centered.sqr();
        let var = centered_sq.mean_keepdim::<OUT_RANK>(axis);

        // sqrt(var + eps)
        let std = var.add_scalar(eps).sqrt();

        // centered / std
        let normalized = match (&centered, &std) {
            (GpuOr::Cpu(a), GpuOr::Cpu(b)) => GpuOr::Cpu((a / b).eval()),
            (GpuOr::Gpu(a), GpuOr::Gpu(b)) => GpuOr::Gpu(a / b),
            _ => panic!("Cannot mix CPU and GPU tensors"),
        };

        // normalized * weight
        let scaled = match (&normalized, weight) {
            (GpuOr::Cpu(a), GpuOr::Cpu(b)) => GpuOr::Cpu((a * b).eval()),
            (GpuOr::Gpu(a), GpuOr::Gpu(b)) => GpuOr::Gpu(a * b),
            _ => panic!("Cannot mix CPU and GPU tensors"),
        };

        // + bias if present
        if let Some(b) = bias {
            match (&scaled, b) {
                (GpuOr::Cpu(a), GpuOr::Cpu(c)) => GpuOr::Cpu((a + c).eval()),
                (GpuOr::Gpu(a), GpuOr::Gpu(c)) => GpuOr::Gpu(a + c),
                _ => panic!("Cannot mix CPU and GPU tensors"),
            }
        } else {
            scaled
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_softmax_cpu() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t: GpuOr<1, f32> = GpuOr::Cpu(fusor_cpu::Tensor::from_slice([6], &data));

        let result = t.softmax::<0>(0);
        let slice = result.as_slice().await.unwrap();

        // Compute expected softmax
        let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = data.iter().map(|x| (x - max_val).exp()).collect();
        let sum_exp: f32 = exp_vals.iter().sum();
        let expected: Vec<f32> = exp_vals.iter().map(|x| x / sum_exp).collect();

        for i in 0..6 {
            assert!(
                (slice[[i]] - expected[i]).abs() < 0.001,
                "Mismatch at index {}: got {}, expected {}",
                i,
                slice[[i]],
                expected[i]
            );
        }

        // Check that softmax sums to 1
        let sum: f32 = (0..6).map(|i| slice[[i]]).sum();
        assert!((sum - 1.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_softmax_2d_cpu() {
        // 2x3 tensor, softmax along axis 1
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t: GpuOr<2, f32> = GpuOr::Cpu(fusor_cpu::Tensor::from_slice([2, 3], &data));

        let result = t.softmax::<1>(1);
        let slice = result.as_slice().await.unwrap();

        // Each row should sum to 1
        let sum_row0: f32 = (0..3).map(|j| slice[[0, j]]).sum();
        let sum_row1: f32 = (0..3).map(|j| slice[[1, j]]).sum();
        assert!((sum_row0 - 1.0).abs() < 0.001);
        assert!((sum_row1 - 1.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_rms_norm_cpu() {
        // Simple test: normalize [1, 2, 3] with weight [1, 1, 1]
        let data = [1.0f32, 2.0, 3.0];
        let t: GpuOr<1, f32> = GpuOr::Cpu(fusor_cpu::Tensor::from_slice([3], &data));
        let weight: GpuOr<1, f32> =
            GpuOr::Cpu(fusor_cpu::Tensor::from_slice([3], &[1.0, 1.0, 1.0]));

        let result = t.rms_norm::<0>(&weight, 1e-5);
        let slice = result.as_slice().await.unwrap();

        // rms = sqrt(mean([1, 4, 9])) = sqrt(14/3) ~ 2.16
        let rms = ((1.0 + 4.0 + 9.0) / 3.0_f32 + 1e-5).sqrt();
        let expected: Vec<f32> = data.iter().map(|x| x / rms).collect();

        for i in 0..3 {
            assert!(
                (slice[[i]] - expected[i]).abs() < 0.01,
                "Mismatch at index {}: got {}, expected {}",
                i,
                slice[[i]],
                expected[i]
            );
        }
    }

    #[tokio::test]
    async fn test_rms_norm_2d_cpu() {
        // 2x3 tensor
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t: GpuOr<2, f32> = GpuOr::Cpu(fusor_cpu::Tensor::from_slice([2, 3], &data));
        let weight: GpuOr<2, f32> =
            GpuOr::Cpu(fusor_cpu::Tensor::from_slice([2, 3], &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]));

        let result = t.rms_norm::<1>(&weight, 1e-5);
        let slice = result.as_slice().await.unwrap();

        // Row 0: [1, 2, 3] -> rms = sqrt((1+4+9)/3 + eps) = sqrt(14/3 + eps)
        let rms0 = ((1.0 + 4.0 + 9.0) / 3.0_f32 + 1e-5).sqrt();
        assert!((slice[[0, 0]] - 1.0 / rms0).abs() < 0.01);
        assert!((slice[[0, 1]] - 2.0 / rms0).abs() < 0.01);
        assert!((slice[[0, 2]] - 3.0 / rms0).abs() < 0.01);

        // Row 1: [4, 5, 6] -> rms = sqrt((16+25+36)/3 + eps) = sqrt(77/3 + eps)
        let rms1 = ((16.0 + 25.0 + 36.0) / 3.0_f32 + 1e-5).sqrt();
        assert!((slice[[1, 0]] - 4.0 / rms1).abs() < 0.01);
        assert!((slice[[1, 1]] - 5.0 / rms1).abs() < 0.01);
        assert!((slice[[1, 2]] - 6.0 / rms1).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_layer_norm_cpu() {
        // Simple test with remove_mean=false (RMS-like)
        let data = [1.0f32, 2.0, 3.0];
        let t: GpuOr<1, f32> = GpuOr::Cpu(fusor_cpu::Tensor::from_slice([3], &data));
        let weight: GpuOr<1, f32> =
            GpuOr::Cpu(fusor_cpu::Tensor::from_slice([3], &[2.0, 2.0, 2.0]));

        let result = t.layer_norm::<0>(&weight, None, 1e-5, false);
        let slice = result.as_slice().await.unwrap();

        // With remove_mean=false, this should be like rms_norm but with weight=2
        let rms = ((1.0 + 4.0 + 9.0) / 3.0_f32 + 1e-5).sqrt();
        let expected: Vec<f32> = data.iter().map(|x| (x / rms) * 2.0).collect();

        for i in 0..3 {
            assert!(
                (slice[[i]] - expected[i]).abs() < 0.01,
                "Mismatch at index {}: got {}, expected {}",
                i,
                slice[[i]],
                expected[i]
            );
        }
    }

    #[tokio::test]
    async fn test_layer_norm_with_mean_removal() {
        // Test with remove_mean=true (standard layer norm)
        let data = [1.0f32, 2.0, 3.0];
        let t: GpuOr<1, f32> = GpuOr::Cpu(fusor_cpu::Tensor::from_slice([3], &data));
        let weight: GpuOr<1, f32> =
            GpuOr::Cpu(fusor_cpu::Tensor::from_slice([3], &[1.0, 1.0, 1.0]));

        let result = t.layer_norm::<0>(&weight, None, 1e-5, true);
        let slice = result.as_slice().await.unwrap();

        // mean = 2, centered = [-1, 0, 1]
        // var = mean([1, 0, 1]) = 2/3
        // std = sqrt(2/3 + eps)
        let var: f32 = 2.0 / 3.0;
        let std = (var + 1e-5).sqrt();
        let expected: Vec<f32> = vec![
            (-1.0) / std,
            0.0 / std,
            1.0 / std,
        ];

        for i in 0..3 {
            assert!(
                (slice[[i]] - expected[i]).abs() < 0.01,
                "Mismatch at index {}: got {}, expected {}",
                i,
                slice[[i]],
                expected[i]
            );
        }
    }
}
