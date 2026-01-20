//! Unified CPU/GPU tensor abstraction
//!
//! This crate provides a unified interface over `fusor-cpu` (CPU tensors with SIMD fusion)
//! and `fusor-core` (GPU tensors with compute graph batching).
//!
//! The key design is:
//! - `GpuOr<CpuT, GpuT>` is a runtime dispatch enum holding either CPU or GPU version
//! - CPU kernel fusion is preserved (expression types stay lazy)
//! - GPU laziness is preserved (compute graph batching)

pub mod cache;
mod composite;
mod device;
mod error;
pub mod layers;
pub mod quantized;

pub use quantized::QGpuOr;

use std::ops::{Deref, Range};

pub use composite::{arange, arange_step, cat, stack, ToVec1, ToVec2, ToVec3};
pub use device::Device;
pub use error::Error;
use fusor_core::TensorSlice;
use fusor_cpu::TensorBacking;

// Re-export from fusor-cpu
pub use fusor_cpu::{
    Abs,
    // Op types for bounds
    AbsOp,
    Acos,
    AcosOp,
    Acosh,
    AcoshOp,
    Add,
    AddOp,
    Asin,
    AsinOp,
    Asinh,
    AsinhOp,
    Atan,
    AtanOp,
    Atanh,
    AtanhOp,
    // Cast operations
    CastTo,
    ConcreteTensor,
    Cos,
    CosOp,
    Cosh,
    CoshOp,
    Div,
    DivOp,
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
    // Reduction ops
    MaxOp,
    MinOp,
    Mul,
    MulOp,
    Neg,
    NegOp,
    ResolveTensor,
    ResolvedTensor,
    SimdBinaryOp,
    SimdElement,
    SimdReduceOp,
    SimdUnaryOp,
    Sin,
    SinOp,
    Sinh,
    SinhOp,
    Sqrt,
    SqrtOp,
    Sub,
    SubOp,
    SumOp,
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

    /// Returns the shape of the tensor.
    pub fn shape(&self) -> [usize; R]
    where
        B: Expr<Elem = D>,
        D: SimdElement + DataType,
    {
        match self {
            GpuOr::Cpu(t) => Expr::shape(t).try_into().expect("Shape length mismatch"),
            GpuOr::Gpu(t) => *t.shape(),
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
impl_gpuor_pairwise_op!(Rem, rem, %, "Cannot perform remainder on CPU tensor with GPU tensor");

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

// Broadcasting binary operations that can work with tensors of different ranks
impl<const R: usize, D> GpuOr<R, D>
where
    D: SimdElement + DataType + Default,
{
    /// Broadcasting add: broadcasts both tensors to a common shape and adds them.
    pub fn add_<const R2: usize, const R3: usize>(
        &self,
        second: &GpuOr<R2, D>,
    ) -> GpuOr<R3, D, ConcreteTensor<D, R3>>
    where
        (fusor_core::Tensor<R, D>, fusor_core::Tensor<R2, D>): fusor_core::MaxRank<R3, D>,
        D: std::ops::Add<Output = D>,
        AddOp: SimdBinaryOp<D>,
    {
        self.broadcast_binary_op(second, |a, b| match (a, b) {
            (GpuOr::Cpu(a), GpuOr::Cpu(b)) => GpuOr::Cpu((&a + &b).eval()),
            (GpuOr::Gpu(a), GpuOr::Gpu(b)) => GpuOr::Gpu(&a + &b),
            _ => panic!("Cannot mix CPU and GPU tensors"),
        })
    }

    /// Broadcasting subtract: broadcasts both tensors to a common shape and subtracts them.
    pub fn sub_<const R2: usize, const R3: usize>(
        &self,
        second: &GpuOr<R2, D>,
    ) -> GpuOr<R3, D, ConcreteTensor<D, R3>>
    where
        (fusor_core::Tensor<R, D>, fusor_core::Tensor<R2, D>): fusor_core::MaxRank<R3, D>,
        D: std::ops::Sub<Output = D>,
        SubOp: SimdBinaryOp<D>,
    {
        self.broadcast_binary_op(second, |a, b| match (a, b) {
            (GpuOr::Cpu(a), GpuOr::Cpu(b)) => GpuOr::Cpu((&a - &b).eval()),
            (GpuOr::Gpu(a), GpuOr::Gpu(b)) => GpuOr::Gpu(&a - &b),
            _ => panic!("Cannot mix CPU and GPU tensors"),
        })
    }

    /// Broadcasting multiply: broadcasts both tensors to a common shape and multiplies them.
    pub fn mul_<const R2: usize, const R3: usize>(
        &self,
        second: &GpuOr<R2, D>,
    ) -> GpuOr<R3, D, ConcreteTensor<D, R3>>
    where
        (fusor_core::Tensor<R, D>, fusor_core::Tensor<R2, D>): fusor_core::MaxRank<R3, D>,
        D: std::ops::Mul<Output = D>,
        MulOp: SimdBinaryOp<D>,
    {
        self.broadcast_binary_op(second, |a, b| match (a, b) {
            (GpuOr::Cpu(a), GpuOr::Cpu(b)) => GpuOr::Cpu((&a * &b).eval()),
            (GpuOr::Gpu(a), GpuOr::Gpu(b)) => GpuOr::Gpu(&a * &b),
            _ => panic!("Cannot mix CPU and GPU tensors"),
        })
    }

    /// Broadcasting divide: broadcasts both tensors to a common shape and divides them.
    pub fn div_<const R2: usize, const R3: usize>(
        &self,
        second: &GpuOr<R2, D>,
    ) -> GpuOr<R3, D, ConcreteTensor<D, R3>>
    where
        (fusor_core::Tensor<R, D>, fusor_core::Tensor<R2, D>): fusor_core::MaxRank<R3, D>,
        D: std::ops::Div<Output = D>,
        DivOp: SimdBinaryOp<D>,
    {
        self.broadcast_binary_op(second, |a, b| match (a, b) {
            (GpuOr::Cpu(a), GpuOr::Cpu(b)) => GpuOr::Cpu((&a / &b).eval()),
            (GpuOr::Gpu(a), GpuOr::Gpu(b)) => GpuOr::Gpu(&a / &b),
            _ => panic!("Cannot mix CPU and GPU tensors"),
        })
    }

    /// Broadcasting power: broadcasts both tensors to a common shape and computes power.
    pub fn pow_<const R2: usize, const R3: usize>(
        &self,
        second: &GpuOr<R2, D>,
    ) -> GpuOr<R3, D, ConcreteTensor<D, R3>>
    where
        (fusor_core::Tensor<R, D>, fusor_core::Tensor<R2, D>): fusor_core::MaxRank<R3, D>,
        D: FloatDataType + FloatOps,
    {
        self.broadcast_binary_op(second, |a, b| match (&a, &b) {
            (GpuOr::Cpu(_), GpuOr::Cpu(_)) => {
                // Use the pow method from math module
                a.to_concrete().pow(&b.to_concrete())
            }
            (GpuOr::Gpu(a), GpuOr::Gpu(b)) => GpuOr::Gpu(a.pow(b)),
            _ => panic!("Cannot mix CPU and GPU tensors"),
        })
    }

    /// Helper function for broadcasting binary operations.
    fn broadcast_binary_op<const R2: usize, const R3: usize>(
        &self,
        second: &GpuOr<R2, D>,
        op: impl Fn(GpuOr<R3, D>, GpuOr<R3, D>) -> GpuOr<R3, D>,
    ) -> GpuOr<R3, D, ConcreteTensor<D, R3>>
    where
        (fusor_core::Tensor<R, D>, fusor_core::Tensor<R2, D>): fusor_core::MaxRank<R3, D>,
    {
        // Calculate the broadcasted shape
        let shape1 = self.shape();
        let shape2 = second.shape();
        let out_shape = broadcast_shapes::<R, R2, R3>(&shape1, &shape2);

        // Broadcast both tensors to the output shape
        let b1: GpuOr<R3, D, ConcreteTensor<D, R3>> = self.broadcast_as(out_shape);
        let b2: GpuOr<R3, D, ConcreteTensor<D, R3>> = second.broadcast_as(out_shape);

        // Apply the operation
        op(b1, b2).to_concrete()
    }
}

/// Calculate the broadcasted shape for two tensors.
fn broadcast_shapes<const R1: usize, const R2: usize, const R3: usize>(
    shape1: &[usize; R1],
    shape2: &[usize; R2],
) -> [usize; R3] {
    let mut result = [1usize; R3];

    // Align shapes from the right
    for i in 0..R1 {
        let idx = R3 - R1 + i;
        result[idx] = shape1[i];
    }
    for i in 0..R2 {
        let idx = R3 - R2 + i;
        if result[idx] == 1 {
            result[idx] = shape2[i];
        } else if shape2[i] != 1 && shape2[i] != result[idx] {
            panic!(
                "Cannot broadcast shapes: dimension {} has incompatible sizes {} and {}",
                idx, result[idx], shape2[i]
            );
        }
    }

    result
}

/// Macro to implement unary element-wise operations for GpuOr.
macro_rules! impl_gpuor_unary_op {
    ($method:ident, $op:ident) => {
        impl<const R: usize, D> GpuOr<R, D>
        where
            D: SimdElement + DataType + FloatDataType + Default,
            fusor_cpu::$op: fusor_cpu::SimdUnaryOp<D>,
        {
            #[doc = concat!("Element-wise ", stringify!($method), " operation.")]
            pub fn $method(&self) -> GpuOr<R, D> {
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
impl_gpuor_unary_op!(asin, AsinOp);
impl_gpuor_unary_op!(acos, AcosOp);
impl_gpuor_unary_op!(atan, AtanOp);
impl_gpuor_unary_op!(sinh, SinhOp);
impl_gpuor_unary_op!(cosh, CoshOp);
impl_gpuor_unary_op!(asinh, AsinhOp);
impl_gpuor_unary_op!(acosh, AcoshOp);
impl_gpuor_unary_op!(atanh, AtanhOp);

// Approximate exp operations (GPU-optimized, CPU falls back to standard exp)
impl<const R: usize, D> GpuOr<R, D>
where
    D: SimdElement + DataType + FloatDataType + Default,
    fusor_cpu::ExpOp: fusor_cpu::SimdUnaryOp<D>,
{
    /// Approximate exp function (faster but less accurate on GPU, exact on CPU).
    /// Uses a polynomial approximation on GPU for better performance.
    pub fn approximate_exp(&self) -> GpuOr<R, D> {
        match self {
            GpuOr::Cpu(t) => GpuOr::Cpu(t.exp()),
            GpuOr::Gpu(t) => GpuOr::Gpu(t.appoximate_exp()),
        }
    }

    /// Less approximate exp function (medium accuracy/speed tradeoff on GPU, exact on CPU).
    pub fn less_approximate_exp(&self) -> GpuOr<R, D> {
        match self {
            GpuOr::Cpu(t) => GpuOr::Cpu(t.exp()),
            GpuOr::Gpu(t) => GpuOr::Gpu(t.less_appoximate_exp()),
        }
    }
}

// Exact tanh operation
impl<const R: usize, D> GpuOr<R, D>
where
    D: SimdElement + DataType + FloatDataType + Default,
    fusor_cpu::TanhOp: fusor_cpu::SimdUnaryOp<D>,
{
    /// Exact tanh using (e^x - e^-x) / (e^x + e^-x).
    /// More accurate but potentially slower than built-in tanh on some platforms.
    pub fn tanh_exact(&self) -> GpuOr<R, D> {
        match self {
            // CPU tanh is already exact
            GpuOr::Cpu(t) => GpuOr::Cpu(t.tanh()),
            GpuOr::Gpu(t) => GpuOr::Gpu(t.tanh_exact()),
        }
    }
}

// Conditional operation (where_cond)
impl<const R: usize, D> GpuOr<R, D>
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
impl<const R: usize, D> GpuOr<R, D>
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

    /// Raise each element to a power (alias for pow_scalar for fusor-core API compatibility).
    pub fn pow_elementwise(&self, exponent: D) -> Self {
        self.pow_scalar(exponent)
    }

    /// Element-wise maximum with a scalar (alias for max_scalar for fusor-core API compatibility).
    pub fn max_elementwise(&self, element: D) -> Self {
        self.max_scalar(element)
    }

    /// Element-wise minimum with a scalar (alias for min_scalar for fusor-core API compatibility).
    pub fn min_elementwise(&self, element: D) -> Self {
        self.min_scalar(element)
    }

    /// Add a scalar to each element.
    pub fn add_scalar(&self, scalar: D) -> Self
    where
        D: std::ops::Add<Output = D>,
        AddOp: SimdBinaryOp<D>,
    {
        match self {
            GpuOr::Cpu(t) => GpuOr::Cpu(t.add_scalar(scalar).eval()),
            GpuOr::Gpu(t) => GpuOr::Gpu(t.clone() + scalar),
        }
    }

    /// Subtract a scalar from each element.
    pub fn sub_scalar(&self, scalar: D) -> Self
    where
        D: std::ops::Sub<Output = D>,
        SubOp: SimdBinaryOp<D>,
    {
        match self {
            GpuOr::Cpu(t) => GpuOr::Cpu(t.sub_scalar(scalar).eval()),
            GpuOr::Gpu(t) => GpuOr::Gpu(t.clone() - scalar),
        }
    }

    /// Multiply each element by a scalar.
    pub fn mul_scalar(&self, scalar: D) -> Self
    where
        D: std::ops::Mul<Output = D>,
        MulOp: SimdBinaryOp<D>,
    {
        match self {
            GpuOr::Cpu(t) => GpuOr::Cpu(t.mul_scalar(scalar).eval()),
            GpuOr::Gpu(t) => GpuOr::Gpu(t.clone() * scalar),
        }
    }

    /// Divide each element by a scalar.
    pub fn div_scalar(&self, scalar: D) -> Self
    where
        D: std::ops::Div<Output = D>,
        DivOp: SimdBinaryOp<D>,
    {
        match self {
            GpuOr::Cpu(t) => GpuOr::Cpu(t.div_scalar(scalar).eval()),
            GpuOr::Gpu(t) => GpuOr::Gpu(t.clone() / scalar),
        }
    }
}

// Cast operation
impl<const R: usize, D> GpuOr<R, D>
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
impl<const R: usize, D> GpuOr<R, D>
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
impl<const R: usize, D> GpuOr<R, D>
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
impl<const R: usize, D> GpuOr<R, D>
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

    /// Alias for matmul (for API compatibility with fusor-core)
    pub fn mat_mul(&self, rhs: &Self) -> Self {
        self.matmul(rhs)
    }
}

// Quantized matrix multiplication for GpuOr<R, f32>
impl<const R: usize, B> GpuOr<R, f32, B>
where
    B: TensorBacking<R, Elem = f32> + ResolveTensor<R>,
{
    /// Quantized matrix multiplication: self @ weights where weights is quantized.
    ///
    /// Computes `self @ weights` where `self` is an f32 tensor and `weights` is a
    /// quantized 2D tensor. This is optimized for the case where weights are stored
    /// in quantized format (e.g., from GGUF model files).
    ///
    /// # Arguments
    /// * `weights` - A quantized 2D weight matrix
    ///
    /// # Panics
    /// * If attempting to mix CPU and GPU tensors (self on CPU, weights on GPU or vice versa)
    /// * If R < 2 (matrix multiplication requires at least 2 dimensions)
    pub fn q_mat_mul(&self, weights: &crate::QGpuOr<2>) -> GpuOr<R, f32> {
        use crate::QGpuOr;

        match (self, weights) {
            // CPU path - dispatch based on block type
            // eval() returns Tensor<R, ConcreteTensor>, so we need .inner() to get ConcreteTensor
            (GpuOr::Cpu(lhs), QGpuOr::CpuQ4_0(rhs)) => {
                GpuOr::Cpu(fusor_cpu::Tensor::new(lhs.eval().inner().q_mat_mul(rhs)))
            }
            (GpuOr::Cpu(lhs), QGpuOr::CpuQ5_0(rhs)) => {
                GpuOr::Cpu(fusor_cpu::Tensor::new(lhs.eval().inner().q_mat_mul(rhs)))
            }
            (GpuOr::Cpu(lhs), QGpuOr::CpuQ8_0(rhs)) => {
                GpuOr::Cpu(fusor_cpu::Tensor::new(lhs.eval().inner().q_mat_mul(rhs)))
            }
            (GpuOr::Cpu(lhs), QGpuOr::CpuQ4K(rhs)) => {
                GpuOr::Cpu(fusor_cpu::Tensor::new(lhs.eval().inner().q_mat_mul(rhs)))
            }
            (GpuOr::Cpu(lhs), QGpuOr::CpuQ6K(rhs)) => {
                GpuOr::Cpu(fusor_cpu::Tensor::new(lhs.eval().inner().q_mat_mul(rhs)))
            }

            // GPU path
            (GpuOr::Gpu(lhs), QGpuOr::Gpu(rhs)) => GpuOr::Gpu(lhs.q_mat_mul(rhs)),

            // Mixed - panic
            _ => panic!("Cannot mix CPU and GPU tensors in q_mat_mul"),
        }
    }
}

// Flatten operations
impl<const R: usize, D> GpuOr<R, D>
where
    D: SimdElement + DataType + Default,
{
    /// Flatten the last FROM_END+1 dimensions into one.
    ///
    /// This follows the GPU/fusor-core semantic where FROM_END is the number of
    /// extra dimensions beyond the one being flattened into.
    /// So FROM_END=0 flattens just the last dimension (no-op),
    /// FROM_END=1 flattens the last 2 dimensions, etc.
    ///
    /// Output rank R2 = R - FROM_END.
    pub fn flatten_last_n<const FROM_END: usize, const R2: usize>(
        &self,
    ) -> GpuOr<R2, D, ConcreteTensor<D, R2>>
    where
        fusor_core::Tensor<R, D>: fusor_core::SmallerRank<FROM_END, R2, D>,
    {
        match self {
            GpuOr::Cpu(t) => {
                // CPU flatten_last_n takes N where output = input - N + 1
                // So we need CPU_N = FROM_END + 1 to match GPU semantics
                // Calculate new shape manually since we can't do const arithmetic
                let shape = t.shape();
                let new_shape: [usize; R2] = std::array::from_fn(|i| {
                    if i < R - 1 - FROM_END {
                        shape[i]
                    } else if i == R - 1 - FROM_END {
                        shape[R - 1 - FROM_END..].iter().product()
                    } else {
                        1
                    }
                });
                GpuOr::Cpu(t.reshape(new_shape))
            }
            GpuOr::Gpu(t) => GpuOr::Gpu(t.flatten_last_n::<FROM_END, R2>()),
        }
    }

    /// Flatten the first FROM_START+1 dimensions into one.
    ///
    /// This follows the GPU/fusor-core semantic where FROM_START is the number of
    /// extra dimensions beyond the one being flattened into.
    /// So FROM_START=0 flattens just the first dimension (no-op),
    /// FROM_START=1 flattens the first 2 dimensions, etc.
    ///
    /// Output rank R2 = R - FROM_START.
    pub fn flatten_first_n<const FROM_START: usize, const R2: usize>(
        &self,
    ) -> GpuOr<R2, D, ConcreteTensor<D, R2>>
    where
        fusor_core::Tensor<R, D>: fusor_core::SmallerRank<FROM_START, R2, D>,
    {
        match self {
            GpuOr::Cpu(t) => {
                // Calculate new shape: first element is product of first FROM_START+1 dims
                // remaining elements are the rest of the dimensions
                let shape = t.shape();
                let new_shape: [usize; R2] = std::array::from_fn(|i| {
                    if i == 0 {
                        shape[..=FROM_START].iter().product()
                    } else {
                        shape[i + FROM_START]
                    }
                });
                GpuOr::Cpu(t.reshape(new_shape))
            }
            GpuOr::Gpu(t) => GpuOr::Gpu(t.flatten_first_n::<FROM_START, R2>()),
        }
    }
}

// Device accessor
impl<const R: usize, D, B: TensorBacking<R, Elem = D>> GpuOr<R, D, B>
where
    D: SimdElement + DataType,
{
    /// Get the device this tensor is on.
    pub fn device(&self) -> Device {
        match self {
            GpuOr::Cpu(_) => Device::Cpu,
            GpuOr::Gpu(t) => Device::Gpu(t.device().clone()),
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
