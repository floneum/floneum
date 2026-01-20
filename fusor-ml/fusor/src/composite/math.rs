//! Math operations that work on both CPU and GPU backends.

use crate::{ConcreteTensor, FloatOps, GpuOr, MulOp, SimdBinaryOp, SimdElement};
use fusor_core::{DataType, FloatDataType};

impl<const R: usize, D> GpuOr<R, D>
where
    D: SimdElement + DataType + FloatDataType + FloatOps + Default,
{
    /// Square each element: sqr(x) = x * x
    pub fn sqr(&self) -> Self
    where
        D: std::ops::Mul<Output = D>,
        MulOp: SimdBinaryOp<D>,
    {
        match self {
            GpuOr::Cpu(t) => GpuOr::Cpu((t * t).eval()),
            GpuOr::Gpu(t) => GpuOr::Gpu(t * t),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_sqr_cpu() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t: GpuOr<1, f32> = GpuOr::Cpu(fusor_cpu::Tensor::from_slice([6], &data));
        let result = t.sqr();
        let slice = result.as_slice().await.unwrap();

        for i in 0..6 {
            let expected = data[i] * data[i];
            assert!((slice[[i]] - expected).abs() < 0.001, "Mismatch at index {}", i);
        }
    }
}
