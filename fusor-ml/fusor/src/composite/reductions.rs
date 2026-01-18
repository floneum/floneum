//! Axis reduction operations that work on both CPU and GPU backends.

use crate::{AddOp, ConcreteTensor, DivOp, Expr, FloatOps, GpuOr, SimdBinaryOp, SimdElement};
use fusor_core::{DataType, FloatDataType, LastRank as GpuLastRank, NextRankInner as GpuNextRankInner};
use fusor_cpu::{LastRank as CpuLastRank, MaxOp, MinOp, SimdReduceOp, SumOp};

impl<const R: usize, D> GpuOr<R, D, ConcreteTensor<D, R>>
where
    D: SimdElement + DataType + FloatDataType + FloatOps + Default,
{
    /// Sum along a specific axis, reducing the tensor rank by 1.
    ///
    /// # Arguments
    /// * `axis` - The axis to reduce along (0 to R-1)
    ///
    /// # Type Parameters
    /// - `OUT_RANK`: The output tensor rank (must be R - 1)
    pub fn sum<const OUT_RANK: usize>(
        &self,
        axis: usize,
    ) -> GpuOr<OUT_RANK, D, ConcreteTensor<D, OUT_RANK>>
    where
        ConcreteTensor<D, R>: CpuLastRank<OUT_RANK, D>,
        fusor_core::Tensor<R, D>: GpuLastRank<OUT_RANK, D>,
        SumOp: SimdReduceOp<D>,
    {
        match self {
            GpuOr::Cpu(t) => GpuOr::Cpu(t.sum_axis::<OUT_RANK>(axis)),
            GpuOr::Gpu(t) => GpuOr::Gpu(t.sum(axis)),
        }
    }

    /// Maximum along a specific axis, reducing the tensor rank by 1.
    pub fn max<const OUT_RANK: usize>(
        &self,
        axis: usize,
    ) -> GpuOr<OUT_RANK, D, ConcreteTensor<D, OUT_RANK>>
    where
        ConcreteTensor<D, R>: CpuLastRank<OUT_RANK, D>,
        fusor_core::Tensor<R, D>: GpuLastRank<OUT_RANK, D>,
        MaxOp: SimdReduceOp<D>,
    {
        match self {
            GpuOr::Cpu(t) => GpuOr::Cpu(t.max_axis::<OUT_RANK>(axis)),
            GpuOr::Gpu(t) => GpuOr::Gpu(t.max(axis)),
        }
    }

    /// Minimum along a specific axis, reducing the tensor rank by 1.
    pub fn min<const OUT_RANK: usize>(
        &self,
        axis: usize,
    ) -> GpuOr<OUT_RANK, D, ConcreteTensor<D, OUT_RANK>>
    where
        ConcreteTensor<D, R>: CpuLastRank<OUT_RANK, D>,
        fusor_core::Tensor<R, D>: GpuLastRank<OUT_RANK, D>,
        MinOp: SimdReduceOp<D>,
    {
        match self {
            GpuOr::Cpu(t) => GpuOr::Cpu(t.min_axis::<OUT_RANK>(axis)),
            GpuOr::Gpu(t) => GpuOr::Gpu(t.min(axis)),
        }
    }

    /// Sum along a specific axis, keeping the dimension (with size 1).
    ///
    /// Output has same rank as input, with the reduced dimension having size 1.
    pub fn sum_keepdim<const OUT_RANK: usize>(&self, axis: usize) -> Self
    where
        ConcreteTensor<D, R>: CpuLastRank<OUT_RANK, D>,
        fusor_core::Tensor<R, D>: GpuLastRank<OUT_RANK, D>,
        <fusor_core::Tensor<R, D> as fusor_core::LastRankInner>::LastRank:
            GpuNextRankInner<NextRank = fusor_core::Tensor<R, D>>,
        SumOp: SimdReduceOp<D>,
    {
        match self {
            GpuOr::Cpu(t) => {
                // CPU: sum then reshape to keep dim
                let reduced = t.sum_axis::<OUT_RANK>(axis);
                // Get shape and insert 1 at axis position
                let reduced_shape = Expr::shape(&reduced);
                let mut new_shape = [0usize; R];
                for i in 0..R {
                    if i < axis {
                        new_shape[i] = reduced_shape[i];
                    } else if i == axis {
                        new_shape[i] = 1;
                    } else {
                        new_shape[i] = reduced_shape[i - 1];
                    }
                }
                // Create output with correct shape by creating and assigning
                let total_elements: usize = new_shape.iter().product();
                let reduced_concrete = reduced.eval();
                let data: Vec<D> = (0..total_elements)
                    .map(|i| {
                        // Map linear index in new shape to reduced shape
                        let mut reduced_idx = [0usize; OUT_RANK];
                        let mut remainder = i;
                        for j in (0..R).rev() {
                            if j == axis {
                                remainder /= 1; // skip the size-1 dimension
                            } else {
                                let dim_idx = if j < axis { j } else { j - 1 };
                                reduced_idx[dim_idx] = remainder % new_shape[j];
                                remainder /= new_shape[j];
                            }
                        }
                        reduced_concrete.get(reduced_idx)
                    })
                    .collect();
                GpuOr::Cpu(fusor_cpu::Tensor::from_slice(new_shape, &data))
            }
            GpuOr::Gpu(t) => GpuOr::Gpu(t.sum_keepdim(axis)),
        }
    }

    /// Mean along a specific axis, reducing the tensor rank by 1.
    pub fn mean<const OUT_RANK: usize>(
        &self,
        axis: usize,
    ) -> GpuOr<OUT_RANK, D, ConcreteTensor<D, OUT_RANK>>
    where
        ConcreteTensor<D, R>: CpuLastRank<OUT_RANK, D>,
        fusor_core::Tensor<R, D>: GpuLastRank<OUT_RANK, D>,
        SumOp: SimdReduceOp<D>,
        D: std::ops::Div<Output = D>,
        DivOp: SimdBinaryOp<D>,
    {
        let shape = self.shape();
        let axis_size = shape[axis];
        let sum = self.sum::<OUT_RANK>(axis);
        sum.div_scalar(D::from_f32(axis_size as f32))
    }

    /// Mean along a specific axis, keeping the dimension (with size 1).
    pub fn mean_keepdim<const OUT_RANK: usize>(&self, axis: usize) -> Self
    where
        ConcreteTensor<D, R>: CpuLastRank<OUT_RANK, D>,
        fusor_core::Tensor<R, D>: GpuLastRank<OUT_RANK, D>,
        <fusor_core::Tensor<R, D> as fusor_core::LastRankInner>::LastRank:
            GpuNextRankInner<NextRank = fusor_core::Tensor<R, D>>,
        SumOp: SimdReduceOp<D>,
        D: std::ops::Add<Output = D> + std::ops::Div<Output = D>,
        AddOp: SimdBinaryOp<D>,
        DivOp: SimdBinaryOp<D>,
    {
        let shape = self.shape();
        let axis_size = shape[axis];
        let sum = self.sum_keepdim::<OUT_RANK>(axis);
        sum.div_scalar(D::from_f32(axis_size as f32))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_sum_cpu() {
        // 2x3 tensor
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t: GpuOr<2, f32> = GpuOr::Cpu(fusor_cpu::Tensor::from_slice([2, 3], &data));

        // Sum along axis 0: [1+4, 2+5, 3+6] = [5, 7, 9]
        let result: GpuOr<1, f32, _> = t.sum::<1>(0);
        let slice = result.as_slice().await.unwrap();
        assert!((slice[[0]] - 5.0).abs() < 0.001);
        assert!((slice[[1]] - 7.0).abs() < 0.001);
        assert!((slice[[2]] - 9.0).abs() < 0.001);

        // Sum along axis 1: [1+2+3, 4+5+6] = [6, 15]
        let result: GpuOr<1, f32, _> = t.sum::<1>(1);
        let slice = result.as_slice().await.unwrap();
        assert!((slice[[0]] - 6.0).abs() < 0.001);
        assert!((slice[[1]] - 15.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_max_cpu() {
        let data = [1.0f32, 5.0, 3.0, 4.0, 2.0, 6.0];
        let t: GpuOr<2, f32> = GpuOr::Cpu(fusor_cpu::Tensor::from_slice([2, 3], &data));

        // Max along axis 0: [max(1,4), max(5,2), max(3,6)] = [4, 5, 6]
        let result: GpuOr<1, f32, _> = t.max::<1>(0);
        let slice = result.as_slice().await.unwrap();
        assert!((slice[[0]] - 4.0).abs() < 0.001);
        assert!((slice[[1]] - 5.0).abs() < 0.001);
        assert!((slice[[2]] - 6.0).abs() < 0.001);

        // Max along axis 1: [max(1,5,3), max(4,2,6)] = [5, 6]
        let result: GpuOr<1, f32, _> = t.max::<1>(1);
        let slice = result.as_slice().await.unwrap();
        assert!((slice[[0]] - 5.0).abs() < 0.001);
        assert!((slice[[1]] - 6.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_min_cpu() {
        let data = [1.0f32, 5.0, 3.0, 4.0, 2.0, 6.0];
        let t: GpuOr<2, f32> = GpuOr::Cpu(fusor_cpu::Tensor::from_slice([2, 3], &data));

        // Min along axis 0: [min(1,4), min(5,2), min(3,6)] = [1, 2, 3]
        let result: GpuOr<1, f32, _> = t.min::<1>(0);
        let slice = result.as_slice().await.unwrap();
        assert!((slice[[0]] - 1.0).abs() < 0.001);
        assert!((slice[[1]] - 2.0).abs() < 0.001);
        assert!((slice[[2]] - 3.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_sum_keepdim_cpu() {
        // 2x3 tensor
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t: GpuOr<2, f32> = GpuOr::Cpu(fusor_cpu::Tensor::from_slice([2, 3], &data));

        // Sum along axis 1 with keepdim: [[1+2+3], [4+5+6]] = [[6], [15]]
        let result = t.sum_keepdim::<1>(1);
        assert_eq!(result.shape(), [2, 1]);
        let slice = result.as_slice().await.unwrap();
        assert!((slice[[0, 0]] - 6.0).abs() < 0.001);
        assert!((slice[[1, 0]] - 15.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_mean_cpu() {
        // 2x3 tensor
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t: GpuOr<2, f32> = GpuOr::Cpu(fusor_cpu::Tensor::from_slice([2, 3], &data));

        // Mean along axis 1: [(1+2+3)/3, (4+5+6)/3] = [2, 5]
        let result: GpuOr<1, f32, _> = t.mean::<1>(1);
        let slice = result.as_slice().await.unwrap();
        assert!((slice[[0]] - 2.0).abs() < 0.001);
        assert!((slice[[1]] - 5.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_mean_keepdim_cpu() {
        // 2x3 tensor
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t: GpuOr<2, f32> = GpuOr::Cpu(fusor_cpu::Tensor::from_slice([2, 3], &data));

        // Mean along axis 1 with keepdim: [[(1+2+3)/3], [(4+5+6)/3]] = [[2], [5]]
        let result = t.mean_keepdim::<1>(1);
        assert_eq!(result.shape(), [2, 1]);
        let slice = result.as_slice().await.unwrap();
        assert!((slice[[0, 0]] - 2.0).abs() < 0.001);
        assert!((slice[[1, 0]] - 5.0).abs() < 0.001);
    }
}
