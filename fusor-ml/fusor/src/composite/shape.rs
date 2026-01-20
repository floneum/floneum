//! Shape manipulation operations that work on both CPU and GPU backends.

use std::ops::Range;

use crate::{ConcreteTensor, Device, Expr, GpuOr, SimdElement};
use fusor_core::{DataType, ShapeWithOneHole};
use fusor_types::SlidingWindow;

impl<const R: usize, D> GpuOr<R, D>
where
    D: SimdElement + DataType + Default,
{
    /// Reshape the tensor to a new shape.
    ///
    /// The total number of elements must remain the same.
    pub fn reshape<const R2: usize>(
        &self,
        new_shape: impl ShapeWithOneHole<R2>,
    ) -> GpuOr<R2, D, ConcreteTensor<D, R2>> {
        match self {
            GpuOr::Cpu(t) => {
                let resolved_shape = new_shape.resolve_shape(Expr::shape(t));
                GpuOr::Cpu(t.reshape(resolved_shape))
            }
            GpuOr::Gpu(t) => GpuOr::Gpu(t.reshape(new_shape)),
        }
    }

    /// Transpose two dimensions of the tensor.
    ///
    /// # Arguments
    /// * `dim0` - First dimension to swap
    /// * `dim1` - Second dimension to swap
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Self {
        match self {
            GpuOr::Cpu(t) => GpuOr::Cpu(t.transpose(dim0, dim1)),
            GpuOr::Gpu(t) => GpuOr::Gpu(t.transpose(dim0, dim1)),
        }
    }

    /// Slice the tensor along all dimensions.
    ///
    /// Returns a view into the tensor's data with updated layout.
    pub fn slice(&self, slices: [Range<usize>; R]) -> Self {
        match self {
            GpuOr::Cpu(t) => GpuOr::Cpu(t.slice(slices.clone())),
            GpuOr::Gpu(t) => GpuOr::Gpu(t.slice(slices)),
        }
    }

    /// Permute the tensor dimensions according to the given axes order.
    ///
    /// # Arguments
    /// * `axes` - A permutation of [0, 1, ..., R-1] specifying the new order
    pub fn permute(&self, axes: [usize; R]) -> Self {
        match self {
            GpuOr::Cpu(t) => GpuOr::Cpu(t.permute(axes)),
            GpuOr::Gpu(t) => GpuOr::Gpu(t.permute(axes)),
        }
    }

    /// Broadcast the tensor to a larger shape.
    ///
    /// Broadcasting rules:
    /// - Dimensions are aligned from the right
    /// - A dimension can be broadcast if it's 1 or matches the target
    /// - New dimensions can be added on the left
    pub fn broadcast_as<const R2: usize>(
        &self,
        out_shape: [usize; R2],
    ) -> GpuOr<R2, D, ConcreteTensor<D, R2>> {
        match self {
            GpuOr::Cpu(t) => GpuOr::Cpu(t.broadcast_as(out_shape)),
            GpuOr::Gpu(t) => GpuOr::Gpu(t.broadcast_as(out_shape)),
        }
    }

    /// Expand the tensor to a larger shape (alias for broadcast_as).
    pub fn expand<const R2: usize>(
        &self,
        out_shape: [usize; R2],
    ) -> GpuOr<R2, D, ConcreteTensor<D, R2>> {
        self.broadcast_as(out_shape)
    }

    /// Flatten the tensor to 1D.
    pub fn flatten_all(&self) -> GpuOr<1, D, ConcreteTensor<D, 1>> {
        match self {
            GpuOr::Cpu(t) => GpuOr::Cpu(t.flatten_all()),
            GpuOr::Gpu(t) => GpuOr::Gpu(t.flatten_all()),
        }
    }

    /// Narrow the tensor along a given dimension.
    ///
    /// # Arguments
    /// * `dim` - The dimension to narrow
    /// * `start` - The starting index
    /// * `length` - The length of the slice
    pub fn narrow(&self, dim: usize, start: usize, length: usize) -> Self {
        match self {
            GpuOr::Cpu(t) => GpuOr::Cpu(t.narrow(dim, start, length)),
            GpuOr::Gpu(t) => {
                // GPU narrow is implemented via slice
                let shape = self.shape();
                let mut slices: [Range<usize>; R] = std::array::from_fn(|i| 0..shape[i]);
                slices[dim] = start..start + length;
                GpuOr::Gpu(t.slice(slices))
            }
        }
    }

    /// Split the tensor into chunks along a given dimension.
    ///
    /// # Arguments
    /// * `chunks` - Number of chunks to split into
    /// * `dim` - The dimension to split along
    pub fn chunk(&self, chunks: usize, dim: usize) -> Vec<Self> {
        let shape = self.shape();
        let dim_size = shape[dim];
        let chunk_size = (dim_size + chunks - 1) / chunks;

        let mut result = Vec::with_capacity(chunks);
        let mut start = 0;

        while start < dim_size {
            let length = chunk_size.min(dim_size - start);
            result.push(self.narrow(dim, start, length));
            start += length;
        }

        result
    }

    /// Repeat the tensor along each dimension.
    ///
    /// # Arguments
    /// * `repeats` - Number of times to repeat along each dimension
    pub fn repeat(&self, repeats: [usize; R]) -> Self {
        match self {
            GpuOr::Cpu(t) => GpuOr::Cpu(t.repeat(repeats)),
            GpuOr::Gpu(t) => GpuOr::Gpu(t.repeat(repeats)),
        }
    }

    /// Squeeze a dimension of size 1.
    ///
    /// # Arguments
    /// * `dim` - The dimension to squeeze (must have size 1)
    pub fn squeeze<const R2: usize>(&self, dim: usize) -> GpuOr<R2, D, ConcreteTensor<D, R2>>
    where
        ConcreteTensor<D, R>: fusor_cpu::LastRank<R2, D>,
        fusor_core::Tensor<R, D>: fusor_core::LastRank<R2, D>,
    {
        match self {
            GpuOr::Cpu(t) => GpuOr::Cpu(t.squeeze(dim)),
            GpuOr::Gpu(t) => GpuOr::Gpu(t.squeeze(dim)),
        }
    }

    /// Unsqueeze (add a dimension of size 1).
    ///
    /// # Arguments
    /// * `dim` - Where to insert the new dimension
    pub fn unsqueeze<const R2: usize>(&self, dim: usize) -> GpuOr<R2, D, ConcreteTensor<D, R2>>
    where
        ConcreteTensor<D, R>: fusor_cpu::NextRank<R2, D>,
        fusor_core::Tensor<R, D>: fusor_core::NextRank<R2, D>,
    {
        match self {
            GpuOr::Cpu(t) => GpuOr::Cpu(t.unsqueeze(dim)),
            GpuOr::Gpu(t) => GpuOr::Gpu(t.unsqueeze(dim)),
        }
    }

    /// Create a sliding window view of the tensor (zero-copy).
    ///
    /// This creates overlapping windows along specified dimensions without copying data.
    ///
    /// # Type Parameters
    /// * `DIFF` - Number of windows to create
    /// * `R2` - Output rank (must be R + DIFF)
    ///
    /// # Arguments
    /// * `windows` - Array of SlidingWindow configurations specifying axis, window size, and step
    pub fn sliding_window_view<const DIFF: usize, const R2: usize>(
        &self,
        windows: [SlidingWindow; DIFF],
    ) -> GpuOr<R2, D, ConcreteTensor<D, R2>>
    where
        ConcreteTensor<D, R>: fusor_cpu::LargerRank<R2, DIFF, D>,
        fusor_core::Tensor<R, D>: fusor_core::LargerRank<DIFF, R2, D>,
    {
        match self {
            GpuOr::Cpu(t) => GpuOr::Cpu(t.sliding_window_view(windows)),
            GpuOr::Gpu(t) => GpuOr::Gpu(t.sliding_window_view(windows)),
        }
    }
}

// Transpose for 2D tensors (convenience method)
impl<D> GpuOr<2, D, ConcreteTensor<D, 2>>
where
    D: SimdElement + DataType + Default,
{
    /// Transpose a 2D matrix (swap dimensions 0 and 1).
    pub fn t(&self) -> Self {
        self.transpose(0, 1)
    }
}

// Transpose for 3D tensors (convenience method)
impl<D> GpuOr<3, D, ConcreteTensor<D, 3>>
where
    D: SimdElement + DataType + Default,
{
    /// Transpose last two dimensions.
    pub fn t(&self) -> Self {
        self.transpose(1, 2)
    }
}

// Transpose for 4D tensors (convenience method)
impl<D> GpuOr<4, D, ConcreteTensor<D, 4>>
where
    D: SimdElement + DataType + Default,
{
    /// Transpose last two dimensions.
    pub fn t(&self) -> Self {
        self.transpose(2, 3)
    }
}

/// Concatenate multiple tensors along a given dimension.
///
/// # Arguments
/// * `tensors` - Iterator of tensors to concatenate
/// * `dim` - The dimension to concatenate along
pub fn cat<const R: usize, D>(
    tensors: impl IntoIterator<Item = GpuOr<R, D>>,
    dim: usize,
) -> GpuOr<R, D>
where
    D: SimdElement + DataType + Default,
{
    let tensors: Vec<_> = tensors.into_iter().collect();
    assert!(!tensors.is_empty(), "Cannot concatenate empty list of tensors");

    // Check if all tensors are on the same device
    let is_cpu = tensors[0].is_cpu();
    assert!(
        tensors.iter().all(|t| t.is_cpu() == is_cpu),
        "All tensors must be on the same device for concatenation"
    );

    if is_cpu {
        let cpu_tensors: Vec<_> = tensors
            .into_iter()
            .map(|t| t.unwrap_cpu().eval())
            .collect();
        GpuOr::Cpu(fusor_cpu::Tensor::cat(cpu_tensors, dim))
    } else {
        let gpu_tensors: Vec<_> = tensors.into_iter().map(|t| t.unwrap_gpu()).collect();
        GpuOr::Gpu(fusor_core::Tensor::cat(gpu_tensors, dim))
    }
}

/// Stack tensors along a new dimension.
///
/// # Arguments
/// * `tensors` - Iterator of tensors to stack
/// * `dim` - Where to insert the new stacking dimension
pub fn stack<const R: usize, const R2: usize, D>(
    tensors: impl IntoIterator<Item = GpuOr<R, D>>,
    dim: usize,
) -> GpuOr<R2, D, ConcreteTensor<D, R2>>
where
    D: SimdElement + DataType + Default,
    ConcreteTensor<D, R>: fusor_cpu::NextRank<R2, D>,
    fusor_core::Tensor<R, D>: fusor_core::NextRank<R2, D>,
{
    let tensors: Vec<_> = tensors.into_iter().collect();
    assert!(!tensors.is_empty(), "Cannot stack empty list of tensors");

    // Check if all tensors are on the same device
    let is_cpu = tensors[0].is_cpu();
    assert!(
        tensors.iter().all(|t| t.is_cpu() == is_cpu),
        "All tensors must be on the same device for stacking"
    );

    if is_cpu {
        let cpu_tensors: Vec<_> = tensors
            .into_iter()
            .map(|t| t.unwrap_cpu().eval())
            .collect();
        GpuOr::Cpu(fusor_cpu::Tensor::stack(cpu_tensors, dim))
    } else {
        let gpu_tensors: Vec<_> = tensors.into_iter().map(|t| t.unwrap_gpu()).collect();
        GpuOr::Gpu(fusor_core::Tensor::stack(gpu_tensors, dim))
    }
}

/// Create a range tensor from start (inclusive) to end (exclusive).
pub fn arange<D>(device: &Device, start: D, end: D) -> GpuOr<1, D, ConcreteTensor<D, 1>>
where
    D: SimdElement + DataType + Default + std::ops::Add<Output = D> + PartialOrd + From<u8>,
{
    arange_step(device, start, end, D::from(1u8))
}

/// Create a range tensor with a custom step.
pub fn arange_step<D>(
    device: &Device,
    start: D,
    end: D,
    step: D,
) -> GpuOr<1, D, ConcreteTensor<D, 1>>
where
    D: SimdElement + DataType + Default + std::ops::Add<Output = D> + PartialOrd,
{
    match device {
        Device::Cpu => GpuOr::Cpu(fusor_cpu::Tensor::arange_step(start, end, step)),
        Device::Gpu(gpu_device) => GpuOr::Gpu(fusor_core::Tensor::arange_step(gpu_device, start, end, step)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_reshape_cpu() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t: GpuOr<1, f32> = GpuOr::Cpu(fusor_cpu::Tensor::from_slice([6], &data));

        let reshaped: GpuOr<2, f32, _> = t.reshape([2, 3]);
        let slice = reshaped.as_slice().await.unwrap();

        assert_eq!(slice[[0, 0]], 1.0);
        assert_eq!(slice[[0, 1]], 2.0);
        assert_eq!(slice[[0, 2]], 3.0);
        assert_eq!(slice[[1, 0]], 4.0);
        assert_eq!(slice[[1, 1]], 5.0);
        assert_eq!(slice[[1, 2]], 6.0);
    }

    #[tokio::test]
    async fn test_transpose_cpu() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t: GpuOr<2, f32> = GpuOr::Cpu(fusor_cpu::Tensor::from_slice([2, 3], &data));

        let transposed = t.transpose(0, 1);
        let slice = transposed.as_slice().await.unwrap();

        assert_eq!(slice[[0, 0]], 1.0);
        assert_eq!(slice[[0, 1]], 4.0);
        assert_eq!(slice[[1, 0]], 2.0);
        assert_eq!(slice[[1, 1]], 5.0);
        assert_eq!(slice[[2, 0]], 3.0);
        assert_eq!(slice[[2, 1]], 6.0);
    }

    #[tokio::test]
    async fn test_slice_cpu() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let t: GpuOr<2, f32> = GpuOr::Cpu(fusor_cpu::Tensor::from_slice([3, 3], &data));

        let sliced = t.slice([1..3, 1..3]);
        let slice = sliced.as_slice().await.unwrap();

        assert_eq!(slice[[0, 0]], 5.0);
        assert_eq!(slice[[0, 1]], 6.0);
        assert_eq!(slice[[1, 0]], 8.0);
        assert_eq!(slice[[1, 1]], 9.0);
    }

    #[tokio::test]
    async fn test_permute_cpu() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t: GpuOr<2, f32> = GpuOr::Cpu(fusor_cpu::Tensor::from_slice([2, 3], &data));

        let permuted = t.permute([1, 0]);
        assert_eq!(permuted.shape(), [3, 2]);
    }

    #[tokio::test]
    async fn test_broadcast_as_cpu() {
        let data = [1.0f32, 2.0, 3.0];
        let t: GpuOr<1, f32> = GpuOr::Cpu(fusor_cpu::Tensor::from_slice([3], &data));

        let broadcasted: GpuOr<2, f32, _> = t.broadcast_as([2, 3]);
        let slice = broadcasted.as_slice().await.unwrap();

        assert_eq!(slice[[0, 0]], 1.0);
        assert_eq!(slice[[0, 2]], 3.0);
        assert_eq!(slice[[1, 0]], 1.0);
        assert_eq!(slice[[1, 2]], 3.0);
    }

    #[tokio::test]
    async fn test_flatten_all_cpu() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t: GpuOr<2, f32> = GpuOr::Cpu(fusor_cpu::Tensor::from_slice([2, 3], &data));

        let flattened = t.flatten_all();
        assert_eq!(flattened.shape(), [6]);
    }

    #[tokio::test]
    async fn test_narrow_cpu() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t: GpuOr<2, f32> = GpuOr::Cpu(fusor_cpu::Tensor::from_slice([2, 3], &data));

        let narrowed = t.narrow(1, 1, 2);
        let slice = narrowed.as_slice().await.unwrap();

        assert_eq!(slice[[0, 0]], 2.0);
        assert_eq!(slice[[0, 1]], 3.0);
        assert_eq!(slice[[1, 0]], 5.0);
        assert_eq!(slice[[1, 1]], 6.0);
    }

    #[tokio::test]
    async fn test_chunk_cpu() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t: GpuOr<1, f32> = GpuOr::Cpu(fusor_cpu::Tensor::from_slice([6], &data));

        let chunks = t.chunk(3, 0);
        assert_eq!(chunks.len(), 3);

        let chunk0 = chunks[0].clone().as_slice().await.unwrap();
        assert_eq!(chunk0[[0]], 1.0);
        assert_eq!(chunk0[[1]], 2.0);

        let chunk1 = chunks[1].clone().as_slice().await.unwrap();
        assert_eq!(chunk1[[0]], 3.0);
        assert_eq!(chunk1[[1]], 4.0);

        let chunk2 = chunks[2].clone().as_slice().await.unwrap();
        assert_eq!(chunk2[[0]], 5.0);
        assert_eq!(chunk2[[1]], 6.0);
    }

    #[tokio::test]
    async fn test_repeat_cpu() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let t: GpuOr<2, f32> = GpuOr::Cpu(fusor_cpu::Tensor::from_slice([2, 2], &data));

        let repeated = t.repeat([2, 3]);
        assert_eq!(repeated.shape(), [4, 6]);

        let slice = repeated.as_slice().await.unwrap();
        // Verify the pattern repeats correctly
        assert_eq!(slice[[0, 0]], 1.0);
        assert_eq!(slice[[0, 2]], 1.0);
        assert_eq!(slice[[2, 0]], 1.0);
    }

    #[tokio::test]
    async fn test_t_2d_cpu() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t: GpuOr<2, f32> = GpuOr::Cpu(fusor_cpu::Tensor::from_slice([2, 3], &data));

        let transposed = t.t();
        assert_eq!(transposed.shape(), [3, 2]);
    }

    #[tokio::test]
    async fn test_cat_cpu() {
        let a: GpuOr<2, f32> = GpuOr::Cpu(fusor_cpu::Tensor::from_slice([2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
        let b: GpuOr<2, f32> = GpuOr::Cpu(fusor_cpu::Tensor::from_slice([2, 3], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]));

        let catted = cat([a, b], 0);
        assert_eq!(catted.shape(), [4, 3]);

        let slice = catted.as_slice().await.unwrap();
        assert_eq!(slice[[0, 0]], 1.0);
        assert_eq!(slice[[2, 0]], 7.0);
    }

    #[tokio::test]
    async fn test_arange_cpu() {
        let device = Device::Cpu;
        let t = arange(&device, 0.0f32, 5.0);
        assert_eq!(t.shape(), [5]);

        let slice = t.as_slice().await.unwrap();
        assert_eq!(slice[[0]], 0.0);
        assert_eq!(slice[[1]], 1.0);
        assert_eq!(slice[[4]], 4.0);
    }

    #[tokio::test]
    async fn test_arange_step_cpu() {
        let device = Device::Cpu;
        let t = arange_step(&device, 0.0f32, 2.0, 0.5);
        assert_eq!(t.shape(), [4]);

        let slice = t.as_slice().await.unwrap();
        assert_eq!(slice[[0]], 0.0);
        assert_eq!(slice[[1]], 0.5);
        assert_eq!(slice[[2]], 1.0);
        assert_eq!(slice[[3]], 1.5);
    }
}
