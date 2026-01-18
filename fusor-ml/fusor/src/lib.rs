//! Unified CPU/GPU tensor abstraction
//!
//! This crate provides a unified interface over `fusor-cpu` (CPU tensors with SIMD fusion)
//! and `fusor-core` (GPU tensors with compute graph batching).
//!
//! The key design is:
//! - `GpuOr<CpuT, GpuT>` is a runtime dispatch enum holding either CPU or GPU version
//! - CPU kernel fusion is preserved (expression types stay lazy)
//! - GPU laziness is preserved (compute graph batching)

mod device;
mod error;

use std::ops::{Deref, Range};

pub use device::Device;
pub use error::Error;
use fusor_core::TensorSlice;
use fusor_cpu::TensorBacking;

// Re-export from fusor-cpu
pub use fusor_cpu::{
    Abs,
    // Op types for bounds
    AbsOp,
    Add,
    // Cast operations
    CastTo,
    ConcreteTensor,
    Cos,
    CosOp,
    Div,
    Exp,
    Exp2,
    Exp2Op,
    ExpOp,
    Expr,
    // Float operations
    FloatOps,
    // Conditional operations
    IsNonZero,
    Log,
    Log2,
    Log2Op,
    LogOp,
    // Matmul
    MatmulImpl,
    Mul,
    Neg,
    NegOp,
    ResolveTensor,
    ResolvedTensor,
    SimdElement,
    SimdUnaryOp,
    Sin,
    SinOp,
    Sqrt,
    SqrtOp,
    Sub,
    Tan,
    TanOp,
    Tanh,
    TanhOp,
    Tensor as CpuTensor,
};

pub use fusor_core::Tensor as GpuTensor;

// Re-export from fusor-core for GPU types
pub use fusor_core::{DataType, FloatDataType};

/// Runtime dispatch wrapper - holds either CPU or GPU version of an operation/tensor type.
///
/// This enum enables writing generic code that works with both CPU and GPU tensors
/// while preserving the benefits of each backend:
/// - CPU: Expression types stay lazy and fuse at resolve time
/// - GPU: Operations build a compute graph that batches at resolve time
#[derive(Clone)]
pub enum GpuOr<const R: usize, D, B: TensorBacking<R, Elem = D> = fusor_cpu::ConcreteTensor<D, R>> {
    Cpu(CpuTensor<R, B>),
    Gpu(GpuTensor<R, D>),
}

impl<const R: usize, D, B> GpuOr<R, D, B>
where
    B: TensorBacking<R, Elem = D>,
{
    /// Returns true if this is the CPU variant.
    #[inline]
    pub fn is_cpu(&self) -> bool {
        matches!(self, GpuOr::Cpu(_))
    }

    /// Returns true if this is the GPU variant.
    #[inline]
    pub fn is_gpu(&self) -> bool {
        matches!(self, GpuOr::Gpu(_))
    }

    /// Returns a reference to the CPU tensor if this is the CPU variant.
    #[inline]
    pub fn as_cpu(&self) -> Option<&CpuTensor<R, B>> {
        match self {
            GpuOr::Cpu(t) => Some(t),
            _ => None,
        }
    }

    /// Returns a reference to the GPU tensor if this is the GPU variant.
    #[inline]
    pub fn as_gpu(&self) -> Option<&GpuTensor<R, D>> {
        match self {
            GpuOr::Gpu(t) => Some(t),
            _ => None,
        }
    }

    /// Returns a mutable reference to the CPU tensor if this is the CPU variant.
    #[inline]
    pub fn as_cpu_mut(&mut self) -> Option<&mut CpuTensor<R, B>> {
        match self {
            GpuOr::Cpu(t) => Some(t),
            _ => None,
        }
    }

    /// Returns a mutable reference to the GPU tensor if this is the GPU variant.
    #[inline]
    pub fn as_gpu_mut(&mut self) -> Option<&mut GpuTensor<R, D>> {
        match self {
            GpuOr::Gpu(t) => Some(t),
            _ => None,
        }
    }

    /// Unwrap the CPU variant, panicking if this is a GPU tensor.
    #[inline]
    pub fn unwrap_cpu(self) -> CpuTensor<R, B> {
        match self {
            GpuOr::Cpu(t) => t,
            GpuOr::Gpu(_) => panic!("Expected CPU tensor, found GPU tensor"),
        }
    }

    /// Unwrap the GPU variant, panicking if this is a CPU tensor.
    #[inline]
    pub fn unwrap_gpu(self) -> GpuTensor<R, D> {
        match self {
            GpuOr::Gpu(t) => t,
            GpuOr::Cpu(_) => panic!("Expected GPU tensor, found CPU tensor"),
        }
    }

    #[inline]
    pub fn dispatch<const R2: usize, D2, B2>(
        self,
        cpu_fn: impl FnOnce(CpuTensor<R, B>) -> CpuTensor<R2, B2>,
        gpu_fn: impl FnOnce(GpuTensor<R, D>) -> GpuTensor<R2, D2>,
    ) -> GpuOr<R2, D2, B2>
    where
        B2: TensorBacking<R2, Elem = D2>,
    {
        match self {
            GpuOr::Cpu(t) => GpuOr::Cpu(cpu_fn(t)),
            GpuOr::Gpu(t) => GpuOr::Gpu(gpu_fn(t)),
        }
    }

    pub async fn as_slice(self) -> Result<TensorSlice<R, D, EitherMappedBuffer>, Error>
    where
        B: ResolveTensor<R>,
        D: fusor_cpu::SimdElement + DataType,
    {
        match self {
            GpuOr::Cpu(t) => Ok(t.as_slice().map_bytes(EitherMappedBuffer::Cpu)),
            GpuOr::Gpu(t) => {
                let mapped = t.as_slice().await.map_err(|err| Error::Gpu(err.into()))?;
                Ok(mapped.map_bytes(EitherMappedBuffer::Gpu))
            }
        }
    }

    /// Materialize the tensor to a concrete form.
    ///
    /// For CPU tensors, this evaluates any lazy expressions.
    /// For GPU tensors, this is a no-op as GPU tensors are already concrete.
    pub fn to_concrete(&self) -> GpuOr<R, D>
    where
        B: ResolveTensor<R>,
        D: SimdElement,
    {
        match self {
            GpuOr::Cpu(t) => GpuOr::Cpu(t.eval()),
            GpuOr::Gpu(t) => GpuOr::Gpu(t.clone()),
        }
    }
}

pub enum EitherMappedBuffer {
    Cpu(fusor_cpu::CpuMappedBuffer),
    Gpu(fusor_core::MappedBuffer),
}

impl Deref for EitherMappedBuffer {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        match self {
            EitherMappedBuffer::Cpu(buf) => buf.deref(),
            EitherMappedBuffer::Gpu(buf) => buf.deref(),
        }
    }
}

/// Macro to implement pairwise operators for GpuOr.
///
/// Generates all four combinations of owned/reference implementations:
/// - `GpuOr op GpuOr` (owned + owned)
/// - `&GpuOr op &GpuOr` (ref + ref)
/// - `GpuOr op &GpuOr` (owned + ref)
/// - `&GpuOr op GpuOr` (ref + owned)
macro_rules! impl_gpuor_pairwise_op {
    ($trait:ident, $method:ident, $op:tt, $panic_msg:literal) => {
        // Owned + Owned
        impl<const R: usize, D, B, B2, O> std::ops::$trait<GpuOr<R, D, O>> for GpuOr<R, D, B>
        where
            CpuTensor<R, B>: std::ops::$trait<CpuTensor<R, O>, Output = CpuTensor<R, B2>>,
            GpuTensor<R, D>: std::ops::$trait<Output = GpuTensor<R, D>>,
            B: TensorBacking<R, Elem = D>,
            O: TensorBacking<R, Elem = D>,
            B2: TensorBacking<R, Elem = D>,
        {
            type Output = GpuOr<R, D, B2>;

            fn $method(self, rhs: GpuOr<R, D, O>) -> Self::Output {
                match (self, rhs) {
                    (GpuOr::Cpu(lhs), GpuOr::Cpu(rhs)) => GpuOr::Cpu(lhs $op rhs),
                    (GpuOr::Gpu(lhs), GpuOr::Gpu(rhs)) => GpuOr::Gpu(lhs $op rhs),
                    _ => panic!($panic_msg),
                }
            }
        }

        // Ref + Ref
        impl<'a, const R: usize, D, B, B2, O> std::ops::$trait<&'a GpuOr<R, D, O>> for &'a GpuOr<R, D, B>
        where
            &'a CpuTensor<R, B>: std::ops::$trait<&'a CpuTensor<R, O>, Output = CpuTensor<R, B2>>,
            &'a GpuTensor<R, D>: std::ops::$trait<Output = GpuTensor<R, D>>,
            B: TensorBacking<R, Elem = D>,
            O: TensorBacking<R, Elem = D>,
            B2: TensorBacking<R, Elem = D>,
        {
            type Output = GpuOr<R, D, B2>;

            fn $method(self, rhs: &'a GpuOr<R, D, O>) -> Self::Output {
                match (self, rhs) {
                    (GpuOr::Cpu(lhs), GpuOr::Cpu(rhs)) => GpuOr::Cpu(lhs $op rhs),
                    (GpuOr::Gpu(lhs), GpuOr::Gpu(rhs)) => GpuOr::Gpu(lhs $op rhs),
                    _ => panic!($panic_msg),
                }
            }
        }

        // Ref + Owned
        impl<'a, const R: usize, D, B, B2, O> std::ops::$trait<GpuOr<R, D, O>> for &'a GpuOr<R, D, B>
        where
            &'a CpuTensor<R, B>: std::ops::$trait<CpuTensor<R, O>, Output = CpuTensor<R, B2>>,
            &'a GpuTensor<R, D>: std::ops::$trait<GpuTensor<R, D>, Output = GpuTensor<R, D>>,
            B: TensorBacking<R, Elem = D>,
            O: TensorBacking<R, Elem = D>,
            B2: TensorBacking<R, Elem = D>,
        {
            type Output = GpuOr<R, D, B2>;

            fn $method(self, rhs: GpuOr<R, D, O>) -> Self::Output {
                match (self, rhs) {
                    (GpuOr::Cpu(lhs), GpuOr::Cpu(rhs)) => GpuOr::Cpu(lhs $op rhs),
                    (GpuOr::Gpu(lhs), GpuOr::Gpu(rhs)) => GpuOr::Gpu(lhs $op rhs),
                    _ => panic!($panic_msg),
                }
            }
        }

        // Owned + Ref
        impl<'a, const R: usize, D, B, B2, O> std::ops::$trait<&'a GpuOr<R, D, O>> for GpuOr<R, D, B>
        where
            CpuTensor<R, B>: std::ops::$trait<&'a CpuTensor<R, O>, Output = CpuTensor<R, B2>>,
            GpuTensor<R, D>: std::ops::$trait<&'a GpuTensor<R, D>, Output = GpuTensor<R, D>>,
            B: TensorBacking<R, Elem = D>,
            O: TensorBacking<R, Elem = D>,
            B2: TensorBacking<R, Elem = D>,
        {
            type Output = GpuOr<R, D, B2>;

            fn $method(self, rhs: &'a GpuOr<R, D, O>) -> Self::Output {
                match (self, rhs) {
                    (GpuOr::Cpu(lhs), GpuOr::Cpu(rhs)) => GpuOr::Cpu(lhs $op rhs),
                    (GpuOr::Gpu(lhs), GpuOr::Gpu(rhs)) => GpuOr::Gpu(lhs $op rhs),
                    _ => panic!($panic_msg),
                }
            }
        }
    };
}

impl_gpuor_pairwise_op!(Add, add, +, "Cannot add CPU tensor to GPU tensor");
impl_gpuor_pairwise_op!(Sub, sub, -, "Cannot subtract CPU tensor from GPU tensor");
impl_gpuor_pairwise_op!(Mul, mul, *, "Cannot multiply CPU tensor with GPU tensor");
impl_gpuor_pairwise_op!(Div, div, /, "Cannot divide CPU tensor by GPU tensor");

// Neg trait implementation for GpuOr
impl<const R: usize, D, B, B2> std::ops::Neg for GpuOr<R, D, B>
where
    CpuTensor<R, B>: std::ops::Neg<Output = CpuTensor<R, B2>>,
    GpuTensor<R, D>: std::ops::Neg<Output = GpuTensor<R, D>>,
    B: TensorBacking<R, Elem = D>,
    B2: TensorBacking<R, Elem = D>,
{
    type Output = GpuOr<R, D, B2>;

    fn neg(self) -> Self::Output {
        match self {
            GpuOr::Cpu(t) => GpuOr::Cpu(-t),
            GpuOr::Gpu(t) => GpuOr::Gpu(-t),
        }
    }
}

impl<'a, const R: usize, D, B, B2> std::ops::Neg for &'a GpuOr<R, D, B>
where
    &'a CpuTensor<R, B>: std::ops::Neg<Output = CpuTensor<R, B2>>,
    &'a GpuTensor<R, D>: std::ops::Neg<Output = GpuTensor<R, D>>,
    B: TensorBacking<R, Elem = D>,
    B2: TensorBacking<R, Elem = D>,
{
    type Output = GpuOr<R, D, B2>;

    fn neg(self) -> Self::Output {
        match self {
            GpuOr::Cpu(t) => GpuOr::Cpu(-t),
            GpuOr::Gpu(t) => GpuOr::Gpu(-t),
        }
    }
}

/// Macro to implement unary element-wise operations for GpuOr.
macro_rules! impl_gpuor_unary_op {
    ($method:ident, $op:ident) => {
        impl<const R: usize, D> GpuOr<R, D, ConcreteTensor<D, R>>
        where
            D: SimdElement + DataType + FloatDataType + Default,
            fusor_cpu::$op: fusor_cpu::SimdUnaryOp<D>,
        {
            #[doc = concat!("Element-wise ", stringify!($method), " operation.")]
            pub fn $method(&self) -> GpuOr<R, D, ConcreteTensor<D, R>> {
                match self {
                    GpuOr::Cpu(t) => GpuOr::Cpu(t.$method()),
                    GpuOr::Gpu(t) => GpuOr::Gpu(t.$method()),
                }
            }
        }
    };
}

impl_gpuor_unary_op!(abs, AbsOp);
impl_gpuor_unary_op!(sqrt, SqrtOp);
impl_gpuor_unary_op!(exp, ExpOp);
impl_gpuor_unary_op!(exp2, Exp2Op);
impl_gpuor_unary_op!(log, LogOp);
impl_gpuor_unary_op!(log2, Log2Op);
impl_gpuor_unary_op!(sin, SinOp);
impl_gpuor_unary_op!(cos, CosOp);
impl_gpuor_unary_op!(tan, TanOp);
impl_gpuor_unary_op!(tanh, TanhOp);

// Conditional operation (where_cond)
impl<const R: usize, D> GpuOr<R, D, ConcreteTensor<D, R>>
where
    D: SimdElement + DataType + Default + IsNonZero,
{
    /// Conditional selection: where self != 0, select on_true, else on_false.
    pub fn where_cond(&self, on_true: &Self, on_false: &Self) -> Self {
        match (self, on_true, on_false) {
            (GpuOr::Cpu(c), GpuOr::Cpu(t), GpuOr::Cpu(f)) => GpuOr::Cpu(c.where_cond(t, f)),
            (GpuOr::Gpu(c), GpuOr::Gpu(t), GpuOr::Gpu(f)) => GpuOr::Gpu(c.clone().where_cond(t, f)),
            _ => panic!("Cannot mix CPU and GPU tensors in where_cond"),
        }
    }
}

// Float operations (pow_scalar, max_scalar, min_scalar, clamp)
impl<const R: usize, D> GpuOr<R, D, ConcreteTensor<D, R>>
where
    D: SimdElement + DataType + FloatDataType + FloatOps + Default,
{
    /// Raise each element to a power.
    pub fn pow_scalar(&self, exponent: D) -> Self {
        match self {
            GpuOr::Cpu(t) => GpuOr::Cpu(t.pow_scalar(exponent)),
            GpuOr::Gpu(t) => GpuOr::Gpu(t.pow_elementwise(exponent)),
        }
    }

    /// Element-wise maximum with a scalar.
    pub fn max_scalar(&self, scalar: D) -> Self {
        match self {
            GpuOr::Cpu(t) => GpuOr::Cpu(t.max_scalar(scalar)),
            GpuOr::Gpu(t) => GpuOr::Gpu(t.max_elementwise(scalar)),
        }
    }

    /// Element-wise minimum with a scalar.
    pub fn min_scalar(&self, scalar: D) -> Self {
        match self {
            GpuOr::Cpu(t) => GpuOr::Cpu(t.min_scalar(scalar)),
            GpuOr::Gpu(t) => GpuOr::Gpu(t.min_elementwise(scalar)),
        }
    }

    /// Clamp each element to a range [min, max].
    pub fn clamp(&self, min: D, max: D) -> Self {
        match self {
            GpuOr::Cpu(t) => GpuOr::Cpu(t.clamp(min, max)),
            GpuOr::Gpu(t) => GpuOr::Gpu(t.max_elementwise(min).min_elementwise(max)),
        }
    }
}

// Cast operation
impl<const R: usize, D> GpuOr<R, D, ConcreteTensor<D, R>>
where
    D: SimdElement + DataType + Default,
{
    /// Cast tensor to another element type.
    pub fn cast<D2>(&self) -> GpuOr<R, D2, ConcreteTensor<D2, R>>
    where
        D: CastTo<D2> + fusor_core::CastTensor<D2>,
        D2: SimdElement + DataType + Default,
    {
        match self {
            GpuOr::Cpu(t) => GpuOr::Cpu(t.cast()),
            GpuOr::Gpu(t) => GpuOr::Gpu(t.cast()),
        }
    }
}

// Index select operation
impl<const R: usize, D> GpuOr<R, D, ConcreteTensor<D, R>>
where
    D: SimdElement + DataType + Default,
{
    /// Select elements along a dimension using indices.
    pub fn index_select(
        &self,
        dimension: usize,
        indices: &GpuOr<1, u32, ConcreteTensor<u32, 1>>,
    ) -> Self {
        match (self, indices) {
            (GpuOr::Cpu(t), GpuOr::Cpu(idx)) => GpuOr::Cpu(t.index_select(dimension, idx)),
            (GpuOr::Gpu(t), GpuOr::Gpu(idx)) => GpuOr::Gpu(t.index_select(dimension, idx)),
            _ => panic!("Cannot mix CPU and GPU tensors in index_select"),
        }
    }
}

// Slice assign operation
impl<const R: usize, D> GpuOr<R, D, ConcreteTensor<D, R>>
where
    D: SimdElement + DataType + Default,
{
    /// Returns a new tensor with the slice region replaced by values from the value tensor.
    pub fn slice_assign(&self, slices: [Range<usize>; R], value: &Self) -> Self {
        match (self, value) {
            (GpuOr::Cpu(t), GpuOr::Cpu(v)) => GpuOr::Cpu(t.slice_assign(slices, v)),
            (GpuOr::Gpu(t), GpuOr::Gpu(v)) => GpuOr::Gpu(t.slice_assign(slices, v)),
            _ => panic!("Cannot mix CPU and GPU tensors in slice_assign"),
        }
    }
}

// Matrix multiplication for N-dimensional tensors (N >= 2)
impl<const R: usize, D> GpuOr<R, D, ConcreteTensor<D, R>>
where
    D: SimdElement + DataType + FloatDataType + Default + MatmulImpl,
{
    /// Matrix multiplication (batched for rank > 2)
    /// For 2D: [M, K] @ [K, N] -> [M, N]
    /// For ND: [...batch, M, K] @ [...batch, K, N] -> [...batch, M, N]
    /// Panics if R < 2
    pub fn matmul(&self, rhs: &Self) -> Self {
        match (self, rhs) {
            (GpuOr::Cpu(a), GpuOr::Cpu(b)) => GpuOr::Cpu(a.matmul(b)),
            (GpuOr::Gpu(a), GpuOr::Gpu(b)) => GpuOr::Gpu(a.mat_mul(b)),
            _ => panic!("Cannot multiply CPU tensor with GPU tensor"),
        }
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_gpu_or_add() {
    let a_cpu: CpuTensor<1, fusor_cpu::ConcreteTensor<f32, 1>> =
        fusor_cpu::Tensor::from_slice([3], &[1.0, 2.0, 3.0]);
    let b_cpu: CpuTensor<1, fusor_cpu::ConcreteTensor<f32, 1>> =
        fusor_cpu::Tensor::from_slice([3], &[4.0, 5.0, 6.0]);
    let device = fusor_core::Device::new().await.unwrap();
    let a_gpu: GpuTensor<1, f32> = GpuTensor::new(&device, &[1.0, 2.0, 3.0]);
    let b_gpu: GpuTensor<1, f32> = GpuTensor::new(&device, &[4.0, 5.0, 6.0]);

    let a_cpu_or: GpuOr<1, f32> = GpuOr::Cpu(a_cpu);
    let b_cpu_or: GpuOr<1, f32> = GpuOr::Cpu(b_cpu);
    let a_gpu_or: GpuOr<1, f32> = GpuOr::Gpu(a_gpu);
    let b_gpu_or: GpuOr<1, f32> = GpuOr::Gpu(b_gpu);

    let c_cpu_or = (&a_cpu_or + &b_cpu_or) * &b_cpu_or;
    println!("c_cpu_or: {:?}", c_cpu_or.as_slice().await.unwrap());
    let c_gpu_or = (&a_gpu_or + &b_gpu_or) * &b_gpu_or;
    println!("c_gpu_or: {:?}", c_gpu_or.as_slice().await.unwrap());
}
