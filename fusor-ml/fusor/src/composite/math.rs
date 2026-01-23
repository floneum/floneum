//! Math operations that work on both CPU and GPU backends.

use crate::{ConcreteTensor, FloatOps, Tensor, MulOp, ResolvedTensor, SimdBinaryOp, SimdElement};
use fusor_core::{DataType, FloatDataType};

impl<const R: usize, D> Tensor<R, D>
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
            Tensor::Cpu(t) => Tensor::Cpu((t * t).to_concrete()),
            Tensor::Gpu(t) => Tensor::Gpu(t * t),
        }
    }
}

impl<const R: usize, D> Tensor<R, D, ConcreteTensor<D, R>>
where
    D: SimdElement + DataType + FloatDataType + FloatOps + Default,
{
    /// Element-wise power: pow(self, other) computes self^other for each element.
    pub fn pow(&self, other: &Self) -> Self {
        match (self, other) {
            (Tensor::Cpu(a), Tensor::Cpu(b)) => {
                // Use element-wise powf via iterating
                let shape = self.shape();
                let a_data = ResolvedTensor::data(a.inner());
                let b_data = ResolvedTensor::data(b.inner());
                let result: Vec<D> = a_data
                    .iter()
                    .zip(b_data.iter())
                    .map(|(x, y)| x.powf(*y))
                    .collect();
                Tensor::Cpu(fusor_cpu::Tensor::new(fusor_cpu::ConcreteTensor::from_slice(shape, &result)))
            }
            (Tensor::Gpu(a), Tensor::Gpu(b)) => Tensor::Gpu(a.pow(b)),
            _ => panic!("Cannot mix CPU and GPU tensors in pow"),
        }
    }

    /// Resize tensor to new shape with padding/truncation.
    pub fn resize(&self, new_shape: [usize; R]) -> Self {
        match self {
            Tensor::Cpu(t) => {
                // CPU resize: create new tensor and copy elements
                let old_shape = self.shape();
                let src_data = ResolvedTensor::data(t.inner());
                let mut result = vec![D::default(); new_shape.iter().product()];

                // Calculate how many elements to copy per dimension
                let copy_shape: [usize; R] = std::array::from_fn(|i| old_shape[i].min(new_shape[i]));

                // Copy elements using nested iteration
                fn copy_recursive<D: Copy, const R: usize>(
                    src: &[D],
                    dst: &mut [D],
                    old_shape: &[usize; R],
                    new_shape: &[usize; R],
                    copy_shape: &[usize; R],
                    dim: usize,
                    src_offset: usize,
                    dst_offset: usize,
                ) {
                    if dim == R {
                        dst[dst_offset] = src[src_offset];
                    } else {
                        let old_stride: usize = old_shape[dim + 1..].iter().product();
                        let new_stride: usize = new_shape[dim + 1..].iter().product();
                        for i in 0..copy_shape[dim] {
                            copy_recursive(
                                src,
                                dst,
                                old_shape,
                                new_shape,
                                copy_shape,
                                dim + 1,
                                src_offset + i * old_stride,
                                dst_offset + i * new_stride,
                            );
                        }
                    }
                }

                copy_recursive(
                    src_data.as_ref(),
                    &mut result,
                    &old_shape,
                    &new_shape,
                    &copy_shape,
                    0,
                    0,
                    0,
                );

                Tensor::Cpu(fusor_cpu::Tensor::new(fusor_cpu::ConcreteTensor::from_slice(new_shape, &result)))
            }
            Tensor::Gpu(t) => Tensor::Gpu(t.resize(new_shape)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_sqr_cpu() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t: Tensor<1, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice([6], &data));
        let result = t.sqr();
        let slice = result.as_slice().await.unwrap();

        for i in 0..6 {
            let expected = data[i] * data[i];
            assert!((slice[[i]] - expected).abs() < 0.001, "Mismatch at index {}", i);
        }
    }
}
