//! GPU tensor wrapper types
//!
//! These types wrap fusor_core tensors and provide a unified interface.
//! Each operation type preserves the lazy nature of GPU computation.

use std::marker::PhantomData;

/// Trait for types that have a reference to a GPU device.
pub trait HasDevice {
    fn gpu_device(&self) -> &fusor_core::Device;
}

/// Helper function to collect all elements from a TensorSlice into a Vec
fn tensor_slice_to_vec<const R: usize, T: fusor_core::DataType + Copy>(
    slice: &fusor_core::TensorSlice<R, T>,
) -> Vec<T> {
    let shape = slice.shape();
    let total: usize = shape.iter().product();
    let mut result = Vec::with_capacity(total);

    // Iterate through all indices
    let mut indices = [0usize; R];
    for _ in 0..total {
        result.push(slice[indices]);
        // Increment indices
        for dim in (0..R).rev() {
            indices[dim] += 1;
            if indices[dim] < shape[dim] {
                break;
            }
            indices[dim] = 0;
        }
    }
    result
}

/// GPU tensor wrapper around fusor_core::Tensor.
#[derive(Clone)]
pub struct GpuTensor<const R: usize, T> {
    inner: fusor_core::Tensor<R, T>,
}

impl<const R: usize, T: fusor_core::DataType> GpuTensor<R, T> {
    /// Create a new GPU tensor from a fusor_core tensor.
    pub fn new(inner: fusor_core::Tensor<R, T>) -> Self {
        Self { inner }
    }

    /// Create a GPU tensor filled with zeros.
    pub fn zeros(device: &fusor_core::Device, shape: [usize; R]) -> Self {
        Self::new(fusor_core::Tensor::splat(device, T::zero(), shape))
    }

    /// Create a GPU tensor filled with a specific value.
    pub fn full(device: &fusor_core::Device, value: T, shape: [usize; R]) -> Self {
        Self::new(fusor_core::Tensor::splat(device, value, shape))
    }

    /// Get a reference to the inner fusor_core tensor.
    pub fn inner(&self) -> &fusor_core::Tensor<R, T> {
        &self.inner
    }

    /// Consume self and return the inner fusor_core tensor.
    pub fn into_inner(self) -> fusor_core::Tensor<R, T> {
        self.inner
    }

    /// Returns the shape of the tensor.
    pub fn shape(&self) -> &[usize; R] {
        self.inner.shape()
    }
}

impl<const R: usize, T: fusor_core::DataType + Copy> GpuTensor<R, T> {
    /// Get the data as a Vec asynchronously.
    pub async fn to_vec(&self) -> Vec<T> {
        let slice = self
            .inner
            .as_slice()
            .await
            .expect("Failed to get tensor slice");
        tensor_slice_to_vec(&slice)
    }

    /// Get the data as a Vec, blocking until complete.
    pub fn to_vec_blocking(&self) -> Vec<T> {
        pollster::block_on(self.to_vec())
    }
}

impl<const R: usize, T: fusor_core::DataType> From<fusor_core::Tensor<R, T>> for GpuTensor<R, T> {
    fn from(inner: fusor_core::Tensor<R, T>) -> Self {
        Self::new(inner)
    }
}

impl<const R: usize, T: fusor_core::DataType + std::fmt::Debug> std::fmt::Debug
    for GpuTensor<R, T>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GpuTensor({:?})", self.inner.shape())
    }
}

impl<const R: usize, T: fusor_core::DataType> HasDevice for GpuTensor<R, T> {
    fn gpu_device(&self) -> &fusor_core::Device {
        self.inner.device()
    }
}

// Trait for types that wrap a GPU tensor (for operation chaining)
pub trait GpuTensorLike<const R: usize, T: fusor_core::DataType>:
    Clone + Send + Sync + HasDevice
{
    fn as_core(&self) -> &fusor_core::Tensor<R, T>;
    fn into_core(self) -> fusor_core::Tensor<R, T>;
    fn shape(&self) -> &[usize; R];
}

impl<const R: usize, T: fusor_core::DataType + Send + Sync> GpuTensorLike<R, T>
    for GpuTensor<R, T>
{
    fn as_core(&self) -> &fusor_core::Tensor<R, T> {
        &self.inner
    }

    fn into_core(self) -> fusor_core::Tensor<R, T> {
        self.inner
    }

    fn shape(&self) -> &[usize; R] {
        self.inner.shape()
    }
}

// GPU operation wrappers that preserve the compute graph

/// GPU Add operation wrapper.
#[derive(Clone)]
pub struct GpuAdd<const R: usize, T, L, Ri> {
    inner: fusor_core::Tensor<R, T>,
    _phantom: PhantomData<(L, Ri)>,
}

impl<const R: usize, T, L, Ri> GpuAdd<R, T, L, Ri>
where
    T: fusor_core::DataType,
    L: GpuTensorLike<R, T>,
    Ri: GpuTensorLike<R, T>,
{
    pub fn new(lhs: &L, rhs: &Ri) -> Self {
        let result = lhs.as_core() + rhs.as_core();
        Self {
            inner: result,
            _phantom: PhantomData,
        }
    }
}

impl<const R: usize, T: fusor_core::DataType + Copy, L, Ri> GpuAdd<R, T, L, Ri> {
    /// Get the data as a Vec asynchronously.
    pub async fn to_vec(&self) -> Vec<T> {
        let slice = self
            .inner
            .as_slice()
            .await
            .expect("Failed to get tensor slice");
        tensor_slice_to_vec(&slice)
    }

    /// Get the data as a Vec, blocking until complete.
    pub fn to_vec_blocking(&self) -> Vec<T> {
        pollster::block_on(self.to_vec())
    }
}

impl<const R: usize, T, L, Ri> std::fmt::Debug for GpuAdd<R, T, L, Ri>
where
    T: fusor_core::DataType + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GpuAdd({:?})", self.inner.shape())
    }
}

impl<const R: usize, T, L, Ri> HasDevice for GpuAdd<R, T, L, Ri>
where
    T: fusor_core::DataType,
{
    fn gpu_device(&self) -> &fusor_core::Device {
        self.inner.device()
    }
}

impl<const R: usize, T, L, Ri> GpuTensorLike<R, T> for GpuAdd<R, T, L, Ri>
where
    T: fusor_core::DataType + Send + Sync,
    L: Clone + Send + Sync + HasDevice,
    Ri: Clone + Send + Sync + HasDevice,
{
    fn as_core(&self) -> &fusor_core::Tensor<R, T> {
        &self.inner
    }

    fn into_core(self) -> fusor_core::Tensor<R, T> {
        self.inner
    }

    fn shape(&self) -> &[usize; R] {
        self.inner.shape()
    }
}

/// GPU Sub operation wrapper.
#[derive(Clone)]
pub struct GpuSub<const R: usize, T, L, Ri> {
    inner: fusor_core::Tensor<R, T>,
    _phantom: PhantomData<(L, Ri)>,
}

impl<const R: usize, T, L, Ri> GpuSub<R, T, L, Ri>
where
    T: fusor_core::DataType,
    L: GpuTensorLike<R, T>,
    Ri: GpuTensorLike<R, T>,
{
    pub fn new(lhs: &L, rhs: &Ri) -> Self {
        let result = lhs.as_core() - rhs.as_core();
        Self {
            inner: result,
            _phantom: PhantomData,
        }
    }
}

impl<const R: usize, T: fusor_core::DataType + Copy, L, Ri> GpuSub<R, T, L, Ri> {
    pub async fn to_vec(&self) -> Vec<T> {
        let slice = self
            .inner
            .as_slice()
            .await
            .expect("Failed to get tensor slice");
        tensor_slice_to_vec(&slice)
    }

    pub fn to_vec_blocking(&self) -> Vec<T> {
        pollster::block_on(self.to_vec())
    }
}

impl<const R: usize, T, L, Ri> std::fmt::Debug for GpuSub<R, T, L, Ri>
where
    T: fusor_core::DataType + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GpuSub({:?})", self.inner.shape())
    }
}

impl<const R: usize, T, L, Ri> HasDevice for GpuSub<R, T, L, Ri>
where
    T: fusor_core::DataType,
{
    fn gpu_device(&self) -> &fusor_core::Device {
        self.inner.device()
    }
}

impl<const R: usize, T, L, Ri> GpuTensorLike<R, T> for GpuSub<R, T, L, Ri>
where
    T: fusor_core::DataType + Send + Sync,
    L: Clone + Send + Sync + HasDevice,
    Ri: Clone + Send + Sync + HasDevice,
{
    fn as_core(&self) -> &fusor_core::Tensor<R, T> {
        &self.inner
    }

    fn into_core(self) -> fusor_core::Tensor<R, T> {
        self.inner
    }

    fn shape(&self) -> &[usize; R] {
        self.inner.shape()
    }
}

/// GPU Mul operation wrapper.
#[derive(Clone)]
pub struct GpuMul<const R: usize, T, L, Ri> {
    inner: fusor_core::Tensor<R, T>,
    _phantom: PhantomData<(L, Ri)>,
}

impl<const R: usize, T, L, Ri> GpuMul<R, T, L, Ri>
where
    T: fusor_core::DataType,
    L: GpuTensorLike<R, T>,
    Ri: GpuTensorLike<R, T>,
{
    pub fn new(lhs: &L, rhs: &Ri) -> Self {
        let result = lhs.as_core() * rhs.as_core();
        Self {
            inner: result,
            _phantom: PhantomData,
        }
    }
}

impl<const R: usize, T: fusor_core::DataType + Copy, L, Ri> GpuMul<R, T, L, Ri> {
    pub async fn to_vec(&self) -> Vec<T> {
        let slice = self
            .inner
            .as_slice()
            .await
            .expect("Failed to get tensor slice");
        tensor_slice_to_vec(&slice)
    }

    pub fn to_vec_blocking(&self) -> Vec<T> {
        pollster::block_on(self.to_vec())
    }
}

impl<const R: usize, T, L, Ri> std::fmt::Debug for GpuMul<R, T, L, Ri>
where
    T: fusor_core::DataType + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GpuMul({:?})", self.inner.shape())
    }
}

impl<const R: usize, T, L, Ri> HasDevice for GpuMul<R, T, L, Ri>
where
    T: fusor_core::DataType,
{
    fn gpu_device(&self) -> &fusor_core::Device {
        self.inner.device()
    }
}

impl<const R: usize, T, L, Ri> GpuTensorLike<R, T> for GpuMul<R, T, L, Ri>
where
    T: fusor_core::DataType + Send + Sync,
    L: Clone + Send + Sync + HasDevice,
    Ri: Clone + Send + Sync + HasDevice,
{
    fn as_core(&self) -> &fusor_core::Tensor<R, T> {
        &self.inner
    }

    fn into_core(self) -> fusor_core::Tensor<R, T> {
        self.inner
    }

    fn shape(&self) -> &[usize; R] {
        self.inner.shape()
    }
}

/// GPU Div operation wrapper.
#[derive(Clone)]
pub struct GpuDiv<const R: usize, T, L, Ri> {
    inner: fusor_core::Tensor<R, T>,
    _phantom: PhantomData<(L, Ri)>,
}

impl<const R: usize, T, L, Ri> GpuDiv<R, T, L, Ri>
where
    T: fusor_core::DataType,
    L: GpuTensorLike<R, T>,
    Ri: GpuTensorLike<R, T>,
{
    pub fn new(lhs: &L, rhs: &Ri) -> Self {
        let result = lhs.as_core() / rhs.as_core();
        Self {
            inner: result,
            _phantom: PhantomData,
        }
    }
}

impl<const R: usize, T: fusor_core::DataType + Copy, L, Ri> GpuDiv<R, T, L, Ri> {
    pub async fn to_vec(&self) -> Vec<T> {
        let slice = self
            .inner
            .as_slice()
            .await
            .expect("Failed to get tensor slice");
        tensor_slice_to_vec(&slice)
    }

    pub fn to_vec_blocking(&self) -> Vec<T> {
        pollster::block_on(self.to_vec())
    }
}

impl<const R: usize, T, L, Ri> std::fmt::Debug for GpuDiv<R, T, L, Ri>
where
    T: fusor_core::DataType + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GpuDiv({:?})", self.inner.shape())
    }
}

impl<const R: usize, T, L, Ri> HasDevice for GpuDiv<R, T, L, Ri>
where
    T: fusor_core::DataType,
{
    fn gpu_device(&self) -> &fusor_core::Device {
        self.inner.device()
    }
}

impl<const R: usize, T, L, Ri> GpuTensorLike<R, T> for GpuDiv<R, T, L, Ri>
where
    T: fusor_core::DataType + Send + Sync,
    L: Clone + Send + Sync + HasDevice,
    Ri: Clone + Send + Sync + HasDevice,
{
    fn as_core(&self) -> &fusor_core::Tensor<R, T> {
        &self.inner
    }

    fn into_core(self) -> fusor_core::Tensor<R, T> {
        self.inner
    }

    fn shape(&self) -> &[usize; R] {
        self.inner.shape()
    }
}

/// GPU Neg operation wrapper.
#[derive(Clone)]
pub struct GpuNeg<const R: usize, T, I> {
    inner: fusor_core::Tensor<R, T>,
    _phantom: PhantomData<I>,
}

impl<const R: usize, T, I> GpuNeg<R, T, I>
where
    T: fusor_core::DataType + std::ops::Neg<Output = T>,
    I: GpuTensorLike<R, T>,
{
    pub fn new(input: &I) -> Self {
        let result = -input.as_core().clone();
        Self {
            inner: result,
            _phantom: PhantomData,
        }
    }
}

impl<const R: usize, T: fusor_core::DataType + Copy, I> GpuNeg<R, T, I> {
    pub async fn to_vec(&self) -> Vec<T> {
        let slice = self
            .inner
            .as_slice()
            .await
            .expect("Failed to get tensor slice");
        tensor_slice_to_vec(&slice)
    }

    pub fn to_vec_blocking(&self) -> Vec<T> {
        pollster::block_on(self.to_vec())
    }
}

impl<const R: usize, T, I> std::fmt::Debug for GpuNeg<R, T, I>
where
    T: fusor_core::DataType + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GpuNeg({:?})", self.inner.shape())
    }
}

impl<const R: usize, T, I> HasDevice for GpuNeg<R, T, I>
where
    T: fusor_core::DataType,
{
    fn gpu_device(&self) -> &fusor_core::Device {
        self.inner.device()
    }
}

impl<const R: usize, T, I> GpuTensorLike<R, T> for GpuNeg<R, T, I>
where
    T: fusor_core::DataType + Send + Sync,
    I: Clone + Send + Sync + HasDevice,
{
    fn as_core(&self) -> &fusor_core::Tensor<R, T> {
        &self.inner
    }

    fn into_core(self) -> fusor_core::Tensor<R, T> {
        self.inner
    }

    fn shape(&self) -> &[usize; R] {
        self.inner.shape()
    }
}

// Unary element-wise operation wrappers

macro_rules! define_gpu_unary_op {
    ($name:ident, $method:ident, $doc:literal) => {
        #[doc = $doc]
        #[derive(Clone)]
        pub struct $name<const R: usize, T, I> {
            inner: fusor_core::Tensor<R, T>,
            _phantom: PhantomData<I>,
        }

        impl<const R: usize, T, I> $name<R, T, I>
        where
            T: fusor_core::FloatDataType,
            I: GpuTensorLike<R, T>,
        {
            pub fn new(input: &I) -> Self {
                let result = input.as_core().$method();
                Self {
                    inner: result,
                    _phantom: PhantomData,
                }
            }
        }

        impl<const R: usize, T: fusor_core::DataType + Copy, I> $name<R, T, I> {
            pub async fn to_vec(&self) -> Vec<T> {
                let slice = self
                    .inner
                    .as_slice()
                    .await
                    .expect("Failed to get tensor slice");
                tensor_slice_to_vec(&slice)
            }

            pub fn to_vec_blocking(&self) -> Vec<T> {
                pollster::block_on(self.to_vec())
            }
        }

        impl<const R: usize, T, I> std::fmt::Debug for $name<R, T, I>
        where
            T: fusor_core::DataType + std::fmt::Debug,
        {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, concat!(stringify!($name), "({:?})"), self.inner.shape())
            }
        }

        impl<const R: usize, T, I> HasDevice for $name<R, T, I>
        where
            T: fusor_core::DataType,
        {
            fn gpu_device(&self) -> &fusor_core::Device {
                self.inner.device()
            }
        }

        impl<const R: usize, T, I> GpuTensorLike<R, T> for $name<R, T, I>
        where
            T: fusor_core::DataType + Send + Sync,
            I: Clone + Send + Sync + HasDevice,
        {
            fn as_core(&self) -> &fusor_core::Tensor<R, T> {
                &self.inner
            }

            fn into_core(self) -> fusor_core::Tensor<R, T> {
                self.inner
            }

            fn shape(&self) -> &[usize; R] {
                self.inner.shape()
            }
        }
    };
}

define_gpu_unary_op!(GpuAbs, abs, "GPU Abs operation wrapper.");
define_gpu_unary_op!(GpuSqrt, sqrt, "GPU Sqrt operation wrapper.");
define_gpu_unary_op!(GpuExp, exp, "GPU Exp operation wrapper.");
define_gpu_unary_op!(GpuExp2, exp2, "GPU Exp2 operation wrapper.");
define_gpu_unary_op!(GpuLog, log, "GPU Log operation wrapper.");
define_gpu_unary_op!(GpuLog2, log2, "GPU Log2 operation wrapper.");
define_gpu_unary_op!(GpuSin, sin, "GPU Sin operation wrapper.");
define_gpu_unary_op!(GpuCos, cos, "GPU Cos operation wrapper.");
define_gpu_unary_op!(GpuTan, tan, "GPU Tan operation wrapper.");
define_gpu_unary_op!(GpuTanh, tanh, "GPU Tanh operation wrapper.");
