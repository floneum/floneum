//! Unified CPU/GPU tensor abstraction
//!
//! This crate provides a unified interface over `fusor-cpu` (CPU tensors with SIMD fusion)
//! and `fusor-core` (GPU tensors with compute graph batching).
//!
//! The key design is:
//! - `Tensor<CpuT, GpuT>` is a runtime dispatch enum holding either CPU or GPU version
//! - CPU kernel fusion is preserved (expression types stay lazy)
//! - GPU laziness is preserved (compute graph batching)

pub mod cache;
mod composite;
mod device;
mod error;
pub mod layers;
pub mod quantized;
mod varbuilder;

pub use varbuilder::{ShardedVarBuilder, VarBuilder};

pub use quantized::QMatrix;

use std::ops::{Deref, Range};

pub use composite::{arange, arange_step, cat, stack, ToVec1, ToVec2, ToVec3};
pub use device::Device;
pub use error::Error;

/// Result type for fusor operations.
pub type Result<T, E = Error> = std::result::Result<T, E>;
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
pub use fusor_core::{DataType, FloatDataType, GgufReadError};

/// Runtime dispatch wrapper - holds either CPU or GPU version of an operation/tensor type.
///
/// This enum enables writing generic code that works with both CPU and GPU tensors
/// while preserving the benefits of each backend:
/// - CPU: Expression types stay lazy and fuse at resolve time
/// - GPU: Operations build a compute graph that batches at resolve time
#[derive(Clone)]
pub enum Tensor<const R: usize, D, B: TensorBacking<R, Elem = D> = fusor_cpu::ConcreteTensor<D, R>> {
    Cpu(CpuTensor<R, B>),
    Gpu(GpuTensor<R, D>),
}

impl<const R: usize, D, B> Tensor<R, D, B>
where
    B: TensorBacking<R, Elem = D>,
{
    /// Returns true if this is the CPU variant.
    #[inline]
    pub fn is_cpu(&self) -> bool {
        matches!(self, Tensor::Cpu(_))
    }

    /// Returns true if this is the GPU variant.
    #[inline]
    pub fn is_gpu(&self) -> bool {
        matches!(self, Tensor::Gpu(_))
    }

    /// Returns a reference to the CPU tensor if this is the CPU variant.
    #[inline]
    pub fn as_cpu(&self) -> Option<&CpuTensor<R, B>> {
        match self {
            Tensor::Cpu(t) => Some(t),
            _ => None,
        }
    }

    /// Returns a reference to the GPU tensor if this is the GPU variant.
    #[inline]
    pub fn as_gpu(&self) -> Option<&GpuTensor<R, D>> {
        match self {
            Tensor::Gpu(t) => Some(t),
            _ => None,
        }
    }

    /// Returns a mutable reference to the CPU tensor if this is the CPU variant.
    #[inline]
    pub fn as_cpu_mut(&mut self) -> Option<&mut CpuTensor<R, B>> {
        match self {
            Tensor::Cpu(t) => Some(t),
            _ => None,
        }
    }

    /// Returns a mutable reference to the GPU tensor if this is the GPU variant.
    #[inline]
    pub fn as_gpu_mut(&mut self) -> Option<&mut GpuTensor<R, D>> {
        match self {
            Tensor::Gpu(t) => Some(t),
            _ => None,
        }
    }

    /// Unwrap the CPU variant, panicking if this is a GPU tensor.
    #[inline]
    pub fn unwrap_cpu(self) -> CpuTensor<R, B> {
        match self {
            Tensor::Cpu(t) => t,
            Tensor::Gpu(_) => panic!("Expected CPU tensor, found GPU tensor"),
        }
    }

    /// Unwrap the GPU variant, panicking if this is a CPU tensor.
    #[inline]
    pub fn unwrap_gpu(self) -> GpuTensor<R, D> {
        match self {
            Tensor::Gpu(t) => t,
            Tensor::Cpu(_) => panic!("Expected GPU tensor, found CPU tensor"),
        }
    }

    #[inline]
    pub fn dispatch<const R2: usize, D2, B2>(
        self,
        cpu_fn: impl FnOnce(CpuTensor<R, B>) -> CpuTensor<R2, B2>,
        gpu_fn: impl FnOnce(GpuTensor<R, D>) -> GpuTensor<R2, D2>,
    ) -> Tensor<R2, D2, B2>
    where
        B2: TensorBacking<R2, Elem = D2>,
    {
        match self {
            Tensor::Cpu(t) => Tensor::Cpu(cpu_fn(t)),
            Tensor::Gpu(t) => Tensor::Gpu(gpu_fn(t)),
        }
    }

    pub async fn as_slice(self) -> Result<TensorSlice<R, D, EitherMappedBuffer>, Error>
    where
        B: ResolveTensor<R>,
        D: fusor_cpu::SimdElement + DataType,
    {
        match self {
            Tensor::Cpu(t) => Ok(t.as_slice().map_bytes(EitherMappedBuffer::Cpu)),
            Tensor::Gpu(t) => {
                let mapped = t.as_slice().await.map_err(|err| Error::Gpu(err.into()))?;
                Ok(mapped.map_bytes(EitherMappedBuffer::Gpu))
            }
        }
    }

    /// Materialize the tensor to a concrete form.
    ///
    /// For CPU tensors, this evaluates any lazy expressions.
    /// For GPU tensors, this is a no-op as GPU tensors are already concrete.
    pub fn to_concrete(&self) -> Tensor<R, D>
    where
        B: ResolveTensor<R>,
        D: SimdElement,
    {
        match self {
            Tensor::Cpu(t) => Tensor::Cpu(t.eval()),
            Tensor::Gpu(t) => Tensor::Gpu(t.clone()),
        }
    }

    /// Returns the shape of the tensor.
    pub fn shape(&self) -> [usize; R]
    where
        B: Expr<Elem = D>,
        D: SimdElement + DataType,
    {
        match self {
            Tensor::Cpu(t) => Expr::shape(t).try_into().expect("Shape length mismatch"),
            Tensor::Gpu(t) => *t.shape(),
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

/// Macro to implement pairwise operators for Tensor.
///
/// Generates all four combinations of owned/reference implementations:
/// - `Tensor op Tensor` (owned + owned)
/// - `&Tensor op &Tensor` (ref + ref)
/// - `Tensor op &Tensor` (owned + ref)
/// - `&Tensor op Tensor` (ref + owned)
macro_rules! impl_tensor_pairwise_op {
    ($trait:ident, $method:ident, $op:tt, $panic_msg:literal) => {
        // Owned + Owned
        impl<const R: usize, D, B, B2, O> std::ops::$trait<Tensor<R, D, O>> for Tensor<R, D, B>
        where
            CpuTensor<R, B>: std::ops::$trait<CpuTensor<R, O>, Output = CpuTensor<R, B2>>,
            GpuTensor<R, D>: std::ops::$trait<Output = GpuTensor<R, D>>,
            B: TensorBacking<R, Elem = D>,
            O: TensorBacking<R, Elem = D>,
            B2: TensorBacking<R, Elem = D>,
        {
            type Output = Tensor<R, D, B2>;

            fn $method(self, rhs: Tensor<R, D, O>) -> Self::Output {
                match (self, rhs) {
                    (Tensor::Cpu(lhs), Tensor::Cpu(rhs)) => Tensor::Cpu(lhs $op rhs),
                    (Tensor::Gpu(lhs), Tensor::Gpu(rhs)) => Tensor::Gpu(lhs $op rhs),
                    _ => panic!($panic_msg),
                }
            }
        }

        // Ref + Ref
        impl<'a, const R: usize, D, B, B2, O> std::ops::$trait<&'a Tensor<R, D, O>> for &'a Tensor<R, D, B>
        where
            &'a CpuTensor<R, B>: std::ops::$trait<&'a CpuTensor<R, O>, Output = CpuTensor<R, B2>>,
            &'a GpuTensor<R, D>: std::ops::$trait<Output = GpuTensor<R, D>>,
            B: TensorBacking<R, Elem = D>,
            O: TensorBacking<R, Elem = D>,
            B2: TensorBacking<R, Elem = D>,
        {
            type Output = Tensor<R, D, B2>;

            fn $method(self, rhs: &'a Tensor<R, D, O>) -> Self::Output {
                match (self, rhs) {
                    (Tensor::Cpu(lhs), Tensor::Cpu(rhs)) => Tensor::Cpu(lhs $op rhs),
                    (Tensor::Gpu(lhs), Tensor::Gpu(rhs)) => Tensor::Gpu(lhs $op rhs),
                    _ => panic!($panic_msg),
                }
            }
        }

        // Ref + Owned
        impl<'a, const R: usize, D, B, B2, O> std::ops::$trait<Tensor<R, D, O>> for &'a Tensor<R, D, B>
        where
            &'a CpuTensor<R, B>: std::ops::$trait<CpuTensor<R, O>, Output = CpuTensor<R, B2>>,
            &'a GpuTensor<R, D>: std::ops::$trait<GpuTensor<R, D>, Output = GpuTensor<R, D>>,
            B: TensorBacking<R, Elem = D>,
            O: TensorBacking<R, Elem = D>,
            B2: TensorBacking<R, Elem = D>,
        {
            type Output = Tensor<R, D, B2>;

            fn $method(self, rhs: Tensor<R, D, O>) -> Self::Output {
                match (self, rhs) {
                    (Tensor::Cpu(lhs), Tensor::Cpu(rhs)) => Tensor::Cpu(lhs $op rhs),
                    (Tensor::Gpu(lhs), Tensor::Gpu(rhs)) => Tensor::Gpu(lhs $op rhs),
                    _ => panic!($panic_msg),
                }
            }
        }

        // Owned + Ref
        impl<'a, const R: usize, D, B, B2, O> std::ops::$trait<&'a Tensor<R, D, O>> for Tensor<R, D, B>
        where
            CpuTensor<R, B>: std::ops::$trait<&'a CpuTensor<R, O>, Output = CpuTensor<R, B2>>,
            GpuTensor<R, D>: std::ops::$trait<&'a GpuTensor<R, D>, Output = GpuTensor<R, D>>,
            B: TensorBacking<R, Elem = D>,
            O: TensorBacking<R, Elem = D>,
            B2: TensorBacking<R, Elem = D>,
        {
            type Output = Tensor<R, D, B2>;

            fn $method(self, rhs: &'a Tensor<R, D, O>) -> Self::Output {
                match (self, rhs) {
                    (Tensor::Cpu(lhs), Tensor::Cpu(rhs)) => Tensor::Cpu(lhs $op rhs),
                    (Tensor::Gpu(lhs), Tensor::Gpu(rhs)) => Tensor::Gpu(lhs $op rhs),
                    _ => panic!($panic_msg),
                }
            }
        }
    };
}

impl_tensor_pairwise_op!(Add, add, +, "Cannot add CPU tensor to GPU tensor");
impl_tensor_pairwise_op!(Sub, sub, -, "Cannot subtract CPU tensor from GPU tensor");
impl_tensor_pairwise_op!(Mul, mul, *, "Cannot multiply CPU tensor with GPU tensor");
impl_tensor_pairwise_op!(Div, div, /, "Cannot divide CPU tensor by GPU tensor");
impl_tensor_pairwise_op!(Rem, rem, %, "Cannot perform remainder on CPU tensor with GPU tensor");

// Neg trait implementation for Tensor
impl<const R: usize, D, B, B2> std::ops::Neg for Tensor<R, D, B>
where
    CpuTensor<R, B>: std::ops::Neg<Output = CpuTensor<R, B2>>,
    GpuTensor<R, D>: std::ops::Neg<Output = GpuTensor<R, D>>,
    B: TensorBacking<R, Elem = D>,
    B2: TensorBacking<R, Elem = D>,
{
    type Output = Tensor<R, D, B2>;

    fn neg(self) -> Self::Output {
        match self {
            Tensor::Cpu(t) => Tensor::Cpu(-t),
            Tensor::Gpu(t) => Tensor::Gpu(-t),
        }
    }
}

impl<'a, const R: usize, D, B, B2> std::ops::Neg for &'a Tensor<R, D, B>
where
    &'a CpuTensor<R, B>: std::ops::Neg<Output = CpuTensor<R, B2>>,
    &'a GpuTensor<R, D>: std::ops::Neg<Output = GpuTensor<R, D>>,
    B: TensorBacking<R, Elem = D>,
    B2: TensorBacking<R, Elem = D>,
{
    type Output = Tensor<R, D, B2>;

    fn neg(self) -> Self::Output {
        match self {
            Tensor::Cpu(t) => Tensor::Cpu(-t),
            Tensor::Gpu(t) => Tensor::Gpu(-t),
        }
    }
}

// Broadcasting binary operations that can work with tensors of different ranks
impl<const R: usize, D> Tensor<R, D>
where
    D: SimdElement + DataType + Default,
{
    /// Broadcasting add: broadcasts both tensors to a common shape and adds them.
    pub fn add_<const R2: usize, const R3: usize>(
        &self,
        second: &Tensor<R2, D>,
    ) -> Tensor<R3, D, ConcreteTensor<D, R3>>
    where
        (fusor_core::Tensor<R, D>, fusor_core::Tensor<R2, D>): fusor_core::MaxRank<R3, D>,
        D: std::ops::Add<Output = D>,
        AddOp: SimdBinaryOp<D>,
    {
        self.broadcast_binary_op(second, |a, b| match (a, b) {
            (Tensor::Cpu(a), Tensor::Cpu(b)) => Tensor::Cpu((&a + &b).eval()),
            (Tensor::Gpu(a), Tensor::Gpu(b)) => Tensor::Gpu(&a + &b),
            _ => panic!("Cannot mix CPU and GPU tensors"),
        })
    }

    /// Broadcasting subtract: broadcasts both tensors to a common shape and subtracts them.
    pub fn sub_<const R2: usize, const R3: usize>(
        &self,
        second: &Tensor<R2, D>,
    ) -> Tensor<R3, D, ConcreteTensor<D, R3>>
    where
        (fusor_core::Tensor<R, D>, fusor_core::Tensor<R2, D>): fusor_core::MaxRank<R3, D>,
        D: std::ops::Sub<Output = D>,
        SubOp: SimdBinaryOp<D>,
    {
        self.broadcast_binary_op(second, |a, b| match (a, b) {
            (Tensor::Cpu(a), Tensor::Cpu(b)) => Tensor::Cpu((&a - &b).eval()),
            (Tensor::Gpu(a), Tensor::Gpu(b)) => Tensor::Gpu(&a - &b),
            _ => panic!("Cannot mix CPU and GPU tensors"),
        })
    }

    /// Broadcasting multiply: broadcasts both tensors to a common shape and multiplies them.
    pub fn mul_<const R2: usize, const R3: usize>(
        &self,
        second: &Tensor<R2, D>,
    ) -> Tensor<R3, D, ConcreteTensor<D, R3>>
    where
        (fusor_core::Tensor<R, D>, fusor_core::Tensor<R2, D>): fusor_core::MaxRank<R3, D>,
        D: std::ops::Mul<Output = D>,
        MulOp: SimdBinaryOp<D>,
    {
        self.broadcast_binary_op(second, |a, b| match (a, b) {
            (Tensor::Cpu(a), Tensor::Cpu(b)) => Tensor::Cpu((&a * &b).eval()),
            (Tensor::Gpu(a), Tensor::Gpu(b)) => Tensor::Gpu(&a * &b),
            _ => panic!("Cannot mix CPU and GPU tensors"),
        })
    }

    /// Broadcasting divide: broadcasts both tensors to a common shape and divides them.
    pub fn div_<const R2: usize, const R3: usize>(
        &self,
        second: &Tensor<R2, D>,
    ) -> Tensor<R3, D, ConcreteTensor<D, R3>>
    where
        (fusor_core::Tensor<R, D>, fusor_core::Tensor<R2, D>): fusor_core::MaxRank<R3, D>,
        D: std::ops::Div<Output = D>,
        DivOp: SimdBinaryOp<D>,
    {
        self.broadcast_binary_op(second, |a, b| match (a, b) {
            (Tensor::Cpu(a), Tensor::Cpu(b)) => Tensor::Cpu((&a / &b).eval()),
            (Tensor::Gpu(a), Tensor::Gpu(b)) => Tensor::Gpu(&a / &b),
            _ => panic!("Cannot mix CPU and GPU tensors"),
        })
    }

    /// Broadcasting power: broadcasts both tensors to a common shape and computes power.
    pub fn pow_<const R2: usize, const R3: usize>(
        &self,
        second: &Tensor<R2, D>,
    ) -> Tensor<R3, D, ConcreteTensor<D, R3>>
    where
        (fusor_core::Tensor<R, D>, fusor_core::Tensor<R2, D>): fusor_core::MaxRank<R3, D>,
        D: FloatDataType + FloatOps,
    {
        self.broadcast_binary_op(second, |a, b| match (&a, &b) {
            (Tensor::Cpu(_), Tensor::Cpu(_)) => {
                // Use the pow method from math module
                a.to_concrete().pow(&b.to_concrete())
            }
            (Tensor::Gpu(a), Tensor::Gpu(b)) => Tensor::Gpu(a.pow(b)),
            _ => panic!("Cannot mix CPU and GPU tensors"),
        })
    }

    /// Helper function for broadcasting binary operations.
    fn broadcast_binary_op<const R2: usize, const R3: usize>(
        &self,
        second: &Tensor<R2, D>,
        op: impl Fn(Tensor<R3, D>, Tensor<R3, D>) -> Tensor<R3, D>,
    ) -> Tensor<R3, D, ConcreteTensor<D, R3>>
    where
        (fusor_core::Tensor<R, D>, fusor_core::Tensor<R2, D>): fusor_core::MaxRank<R3, D>,
    {
        // Calculate the broadcasted shape
        let shape1 = self.shape();
        let shape2 = second.shape();
        let out_shape = broadcast_shapes::<R, R2, R3>(&shape1, &shape2);

        // Broadcast both tensors to the output shape
        let b1: Tensor<R3, D, ConcreteTensor<D, R3>> = self.broadcast_as(out_shape);
        let b2: Tensor<R3, D, ConcreteTensor<D, R3>> = second.broadcast_as(out_shape);

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

/// Macro to implement unary element-wise operations for Tensor.
macro_rules! impl_tensor_unary_op {
    ($method:ident, $op:ident) => {
        impl<const R: usize, D> Tensor<R, D>
        where
            D: SimdElement + DataType + FloatDataType + Default,
            fusor_cpu::$op: fusor_cpu::SimdUnaryOp<D>,
        {
            #[doc = concat!("Element-wise ", stringify!($method), " operation.")]
            pub fn $method(&self) -> Tensor<R, D> {
                match self {
                    Tensor::Cpu(t) => Tensor::Cpu(t.$method()),
                    Tensor::Gpu(t) => Tensor::Gpu(t.$method()),
                }
            }
        }
    };
}

impl_tensor_unary_op!(abs, AbsOp);
impl_tensor_unary_op!(sqrt, SqrtOp);
impl_tensor_unary_op!(exp, ExpOp);
impl_tensor_unary_op!(exp2, Exp2Op);
impl_tensor_unary_op!(log, LogOp);
impl_tensor_unary_op!(log2, Log2Op);
impl_tensor_unary_op!(sin, SinOp);
impl_tensor_unary_op!(cos, CosOp);
impl_tensor_unary_op!(tan, TanOp);
impl_tensor_unary_op!(tanh, TanhOp);
impl_tensor_unary_op!(asin, AsinOp);
impl_tensor_unary_op!(acos, AcosOp);
impl_tensor_unary_op!(atan, AtanOp);
impl_tensor_unary_op!(sinh, SinhOp);
impl_tensor_unary_op!(cosh, CoshOp);
impl_tensor_unary_op!(asinh, AsinhOp);
impl_tensor_unary_op!(acosh, AcoshOp);
impl_tensor_unary_op!(atanh, AtanhOp);

// Approximate exp operations (GPU-optimized, CPU falls back to standard exp)
impl<const R: usize, D> Tensor<R, D>
where
    D: SimdElement + DataType + FloatDataType + Default,
    fusor_cpu::ExpOp: fusor_cpu::SimdUnaryOp<D>,
{
    /// Approximate exp function (faster but less accurate on GPU, exact on CPU).
    /// Uses a polynomial approximation on GPU for better performance.
    pub fn approximate_exp(&self) -> Tensor<R, D> {
        match self {
            Tensor::Cpu(t) => Tensor::Cpu(t.exp()),
            Tensor::Gpu(t) => Tensor::Gpu(t.appoximate_exp()),
        }
    }

    /// Less approximate exp function (medium accuracy/speed tradeoff on GPU, exact on CPU).
    pub fn less_approximate_exp(&self) -> Tensor<R, D> {
        match self {
            Tensor::Cpu(t) => Tensor::Cpu(t.exp()),
            Tensor::Gpu(t) => Tensor::Gpu(t.less_appoximate_exp()),
        }
    }
}

// Exact tanh operation
impl<const R: usize, D> Tensor<R, D>
where
    D: SimdElement + DataType + FloatDataType + Default,
    fusor_cpu::TanhOp: fusor_cpu::SimdUnaryOp<D>,
{
    /// Exact tanh using (e^x - e^-x) / (e^x + e^-x).
    /// More accurate but potentially slower than built-in tanh on some platforms.
    pub fn tanh_exact(&self) -> Tensor<R, D> {
        match self {
            // CPU tanh is already exact
            Tensor::Cpu(t) => Tensor::Cpu(t.tanh()),
            Tensor::Gpu(t) => Tensor::Gpu(t.tanh_exact()),
        }
    }
}

// Conditional operation (where_cond)
impl<const R: usize, D> Tensor<R, D>
where
    D: SimdElement + DataType + Default + IsNonZero,
{
    /// Conditional selection: where self != 0, select on_true, else on_false.
    pub fn where_cond(&self, on_true: &Self, on_false: &Self) -> Self {
        match (self, on_true, on_false) {
            (Tensor::Cpu(c), Tensor::Cpu(t), Tensor::Cpu(f)) => Tensor::Cpu(c.where_cond(t, f)),
            (Tensor::Gpu(c), Tensor::Gpu(t), Tensor::Gpu(f)) => Tensor::Gpu(c.clone().where_cond(t, f)),
            _ => panic!("Cannot mix CPU and GPU tensors in where_cond"),
        }
    }
}

// Float operations (pow_scalar, max_scalar, min_scalar, clamp)
impl<const R: usize, D> Tensor<R, D>
where
    D: SimdElement + DataType + FloatDataType + FloatOps + Default,
{
    /// Raise each element to a power.
    pub fn pow_scalar(&self, exponent: D) -> Self {
        match self {
            Tensor::Cpu(t) => Tensor::Cpu(t.pow_scalar(exponent)),
            Tensor::Gpu(t) => Tensor::Gpu(t.pow_elementwise(exponent)),
        }
    }

    /// Element-wise maximum with a scalar.
    pub fn max_scalar(&self, scalar: D) -> Self {
        match self {
            Tensor::Cpu(t) => Tensor::Cpu(t.max_scalar(scalar)),
            Tensor::Gpu(t) => Tensor::Gpu(t.max_elementwise(scalar)),
        }
    }

    /// Element-wise minimum with a scalar.
    pub fn min_scalar(&self, scalar: D) -> Self {
        match self {
            Tensor::Cpu(t) => Tensor::Cpu(t.min_scalar(scalar)),
            Tensor::Gpu(t) => Tensor::Gpu(t.min_elementwise(scalar)),
        }
    }

    /// Clamp each element to a range [min, max].
    pub fn clamp(&self, min: D, max: D) -> Self {
        match self {
            Tensor::Cpu(t) => Tensor::Cpu(t.clamp(min, max)),
            Tensor::Gpu(t) => Tensor::Gpu(t.max_elementwise(min).min_elementwise(max)),
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
            Tensor::Cpu(t) => Tensor::Cpu(t.add_scalar(scalar).eval()),
            Tensor::Gpu(t) => Tensor::Gpu(t.clone() + scalar),
        }
    }

    /// Subtract a scalar from each element.
    pub fn sub_scalar(&self, scalar: D) -> Self
    where
        D: std::ops::Sub<Output = D>,
        SubOp: SimdBinaryOp<D>,
    {
        match self {
            Tensor::Cpu(t) => Tensor::Cpu(t.sub_scalar(scalar).eval()),
            Tensor::Gpu(t) => Tensor::Gpu(t.clone() - scalar),
        }
    }

    /// Multiply each element by a scalar.
    pub fn mul_scalar(&self, scalar: D) -> Self
    where
        D: std::ops::Mul<Output = D>,
        MulOp: SimdBinaryOp<D>,
    {
        match self {
            Tensor::Cpu(t) => Tensor::Cpu(t.mul_scalar(scalar).eval()),
            Tensor::Gpu(t) => Tensor::Gpu(t.clone() * scalar),
        }
    }

    /// Divide each element by a scalar.
    pub fn div_scalar(&self, scalar: D) -> Self
    where
        D: std::ops::Div<Output = D>,
        DivOp: SimdBinaryOp<D>,
    {
        match self {
            Tensor::Cpu(t) => Tensor::Cpu(t.div_scalar(scalar).eval()),
            Tensor::Gpu(t) => Tensor::Gpu(t.clone() / scalar),
        }
    }
}

// Cast operation
impl<const R: usize, D> Tensor<R, D>
where
    D: SimdElement + DataType + Default,
{
    /// Cast tensor to another element type.
    pub fn cast<D2>(&self) -> Tensor<R, D2, ConcreteTensor<D2, R>>
    where
        D: CastTo<D2> + fusor_core::CastTensor<D2>,
        D2: SimdElement + DataType + Default,
    {
        match self {
            Tensor::Cpu(t) => Tensor::Cpu(t.cast()),
            Tensor::Gpu(t) => Tensor::Gpu(t.cast()),
        }
    }
}

// Index select operation
impl<const R: usize, D> Tensor<R, D>
where
    D: SimdElement + DataType + Default,
{
    /// Select elements along a dimension using indices.
    pub fn index_select(
        &self,
        dimension: usize,
        indices: &Tensor<1, u32, ConcreteTensor<u32, 1>>,
    ) -> Self {
        match (self, indices) {
            (Tensor::Cpu(t), Tensor::Cpu(idx)) => Tensor::Cpu(t.index_select(dimension, idx)),
            (Tensor::Gpu(t), Tensor::Gpu(idx)) => Tensor::Gpu(t.index_select(dimension, idx)),
            _ => panic!("Cannot mix CPU and GPU tensors in index_select"),
        }
    }
}

// Slice assign operation
impl<const R: usize, D> Tensor<R, D>
where
    D: SimdElement + DataType + Default,
{
    /// Returns a new tensor with the slice region replaced by values from the value tensor.
    pub fn slice_assign(&self, slices: [Range<usize>; R], value: &Self) -> Self {
        match (self, value) {
            (Tensor::Cpu(t), Tensor::Cpu(v)) => Tensor::Cpu(t.slice_assign(slices, v)),
            (Tensor::Gpu(t), Tensor::Gpu(v)) => Tensor::Gpu(t.slice_assign(slices, v)),
            _ => panic!("Cannot mix CPU and GPU tensors in slice_assign"),
        }
    }
}

// Matrix multiplication for N-dimensional tensors (N >= 2)
impl<const R: usize, D> Tensor<R, D>
where
    D: SimdElement + DataType + FloatDataType + Default + MatmulImpl,
{
    /// Matrix multiplication (batched for rank > 2)
    /// For 2D: [M, K] @ [K, N] -> [M, N]
    /// For ND: [...batch, M, K] @ [...batch, K, N] -> [...batch, M, N]
    /// Panics if R < 2
    pub fn matmul(&self, rhs: &Self) -> Self {
        match (self, rhs) {
            (Tensor::Cpu(a), Tensor::Cpu(b)) => Tensor::Cpu(a.matmul(b)),
            (Tensor::Gpu(a), Tensor::Gpu(b)) => Tensor::Gpu(a.mat_mul(b)),
            _ => panic!("Cannot multiply CPU tensor with GPU tensor"),
        }
    }

    /// Alias for matmul (for API compatibility with fusor-core)
    pub fn mat_mul(&self, rhs: &Self) -> Self {
        self.matmul(rhs)
    }
}

// Quantized matrix multiplication for Tensor<R, f32>
impl<const R: usize, B> Tensor<R, f32, B>
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
    pub fn q_mat_mul(&self, weights: &crate::QMatrix<2>) -> Tensor<R, f32> {
        use crate::QMatrix;

        match (self, weights) {
            // CPU path - dispatch based on block type
            // eval() returns Tensor<R, ConcreteTensor>, so we need .inner() to get ConcreteTensor
            (Tensor::Cpu(lhs), QMatrix::CpuQ4_0(rhs)) => {
                Tensor::Cpu(fusor_cpu::Tensor::new(lhs.eval().inner().q_mat_mul(rhs)))
            }
            (Tensor::Cpu(lhs), QMatrix::CpuQ5_0(rhs)) => {
                Tensor::Cpu(fusor_cpu::Tensor::new(lhs.eval().inner().q_mat_mul(rhs)))
            }
            (Tensor::Cpu(lhs), QMatrix::CpuQ8_0(rhs)) => {
                Tensor::Cpu(fusor_cpu::Tensor::new(lhs.eval().inner().q_mat_mul(rhs)))
            }
            (Tensor::Cpu(lhs), QMatrix::CpuQ4K(rhs)) => {
                Tensor::Cpu(fusor_cpu::Tensor::new(lhs.eval().inner().q_mat_mul(rhs)))
            }
            (Tensor::Cpu(lhs), QMatrix::CpuQ6K(rhs)) => {
                Tensor::Cpu(fusor_cpu::Tensor::new(lhs.eval().inner().q_mat_mul(rhs)))
            }
            // F32 is not quantized, use regular matmul with transpose
            // Weight is [N, K] (out_features, in_features), we need input @ weight.T
            (Tensor::Cpu(lhs), QMatrix::CpuF32(rhs)) => {
                use fusor_cpu::ResolvedTensor;
                let rhs_shape = ResolvedTensor::shape(rhs);
                let n = rhs_shape[0]; // out_features
                let k = rhs_shape[1]; // in_features

                // Transpose weight from [N, K] to [K, N]
                let rhs_tensor = fusor_cpu::Tensor::new(rhs.clone());
                let rhs_transposed = rhs_tensor.transpose(0, 1);

                // Reshape to R dimensions: [1, 1, ..., K, N]
                let weight_shape: [usize; R] = std::array::from_fn(|i| {
                    if i < R - 2 {
                        1 // Broadcast batch dimensions
                    } else if i == R - 2 {
                        k // K dimension
                    } else {
                        n // N dimension
                    }
                });
                let rhs_broadcast = rhs_transposed.reshape(weight_shape);

                // Do regular matmul
                let lhs_eval = lhs.eval();
                let result = lhs_eval.matmul(&rhs_broadcast);
                Tensor::Cpu(result)
            }

            // GPU path
            (Tensor::Gpu(lhs), QMatrix::Gpu(rhs)) => Tensor::Gpu(lhs.q_mat_mul(rhs)),

            // Mixed - panic
            _ => panic!("Cannot mix CPU and GPU tensors in q_mat_mul"),
        }
    }
}

// Flatten operations
impl<const R: usize, D> Tensor<R, D>
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
    ) -> Tensor<R2, D, ConcreteTensor<D, R2>>
    where
        fusor_core::Tensor<R, D>: fusor_core::SmallerRank<FROM_END, R2, D>,
    {
        match self {
            Tensor::Cpu(t) => {
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
                Tensor::Cpu(t.reshape(new_shape))
            }
            Tensor::Gpu(t) => Tensor::Gpu(t.flatten_last_n::<FROM_END, R2>()),
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
    ) -> Tensor<R2, D, ConcreteTensor<D, R2>>
    where
        fusor_core::Tensor<R, D>: fusor_core::SmallerRank<FROM_START, R2, D>,
    {
        match self {
            Tensor::Cpu(t) => {
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
                Tensor::Cpu(t.reshape(new_shape))
            }
            Tensor::Gpu(t) => Tensor::Gpu(t.flatten_first_n::<FROM_START, R2>()),
        }
    }
}

// Device accessor
impl<const R: usize, D, B: TensorBacking<R, Elem = D>> Tensor<R, D, B>
where
    D: SimdElement + DataType,
{
    /// Get the device this tensor is on.
    pub fn device(&self) -> Device {
        match self {
            Tensor::Cpu(_) => Device::Cpu,
            Tensor::Gpu(t) => Device::Gpu(t.device().clone()),
        }
    }

    /// Returns the rank (number of dimensions) of the tensor.
    ///
    /// This is a const function that returns the compile-time rank R.
    #[inline]
    pub const fn rank(&self) -> usize {
        R
    }
}

// Scalar conversion
impl<const R: usize, D, B: TensorBacking<R, Elem = D>> Tensor<R, D, B>
where
    D: SimdElement + DataType + Default + Copy,
    B: ResolveTensor<R>,
{
    /// Convert a scalar tensor (or get the first element) to a scalar value.
    ///
    /// This is an async operation because GPU tensors need to be mapped to CPU memory.
    pub async fn to_scalar(&self) -> Result<D, Error> {
        match self {
            Tensor::Cpu(t) => {
                let slice = t.as_slice();
                Ok(slice.as_scalar())
            }
            Tensor::Gpu(t) => {
                let result = t.to_scalar().await.map_err(|e| Error::Gpu(e.into()))?;
                Ok(result)
            }
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

    let a_cpu_or: Tensor<1, f32> = Tensor::Cpu(a_cpu);
    let b_cpu_or: Tensor<1, f32> = Tensor::Cpu(b_cpu);
    let a_gpu_or: Tensor<1, f32> = Tensor::Gpu(a_gpu);
    let b_gpu_or: Tensor<1, f32> = Tensor::Gpu(b_gpu);

    let c_cpu_or = (&a_cpu_or + &b_cpu_or) * &b_cpu_or;
    println!("c_cpu_or: {:?}", c_cpu_or.as_slice().await.unwrap());
    let c_gpu_or = (&a_gpu_or + &b_gpu_or) * &b_gpu_or;
    println!("c_gpu_or: {:?}", c_gpu_or.as_slice().await.unwrap());
}

#[cfg(test)]
#[tokio::test]
async fn test_matmul_cpu_vs_gpu() {
    // Create random-ish data for matmul test
    // Simulating attention: Q @ K^T with shape [batch, heads, seq_len, head_dim]
    let a_data: Vec<f32> = (0..1*8*100*64).map(|i| (i as f32 * 0.001).sin()).collect();
    let b_data: Vec<f32> = (0..1*8*64*100).map(|i| (i as f32 * 0.001).cos()).collect();

    // CPU version: [1, 8, 100, 64] @ [1, 8, 64, 100] -> [1, 8, 100, 100]
    let cpu_a: Tensor<4, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 8, 100, 64], &a_data));
    let cpu_b: Tensor<4, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 8, 64, 100], &b_data));
    let cpu_result = cpu_a.matmul(&cpu_b);
    let cpu_slice = cpu_result.as_slice().await.unwrap();

    // GPU version
    let gpu_device = Device::new().await.expect("GPU required for this test");
    let gpu_a: Tensor<4, f32> = Tensor::from_slice(&gpu_device, [1, 8, 100, 64], &a_data);
    let gpu_b: Tensor<4, f32> = Tensor::from_slice(&gpu_device, [1, 8, 64, 100], &b_data);
    let gpu_result = gpu_a.matmul(&gpu_b);
    let gpu_slice = gpu_result.as_slice().await.unwrap();

    // Compare
    assert_eq!(cpu_slice.shape(), gpu_slice.shape());
    assert_eq!(cpu_slice.shape(), &[1, 8, 100, 100]);

    let mut max_diff = 0.0f32;
    let mut sum_diff = 0.0f32;
    let mut count = 0;
    for i in 0..cpu_slice.shape()[0] {
        for j in 0..cpu_slice.shape()[1] {
            for k in 0..cpu_slice.shape()[2].min(50) {
                for l in 0..cpu_slice.shape()[3].min(50) {
                    let cpu_val: f32 = cpu_slice[[i, j, k, l]].into();
                    let gpu_val: f32 = gpu_slice[[i, j, k, l]].into();
                    let diff = (cpu_val - gpu_val).abs();
                    max_diff = max_diff.max(diff);
                    sum_diff += diff;
                    count += 1;
                }
            }
        }
    }

    eprintln!("Matmul CPU vs GPU: max_diff={}, mean_diff={}", max_diff, sum_diff / count as f32);
    eprintln!("CPU[0,0,0,0..5]: {:?}", (0..5).map(|i| cpu_slice[[0, 0, 0, i]]).collect::<Vec<f32>>());
    eprintln!("GPU[0,0,0,0..5]: {:?}", (0..5).map(|i| gpu_slice[[0, 0, 0, i]]).collect::<Vec<f32>>());

    assert!(max_diff < 0.001, "Matmul CPU and GPU outputs differ too much: max_diff={}", max_diff);
}
