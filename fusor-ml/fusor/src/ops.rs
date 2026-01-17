//! Operations on GpuOr tensors
//!
//! This module provides arithmetic and element-wise operations on GpuOr tensors
//! that dispatch to either CPU or GPU implementations at runtime.

use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::gpu::{
    GpuAbs, GpuAdd, GpuCos, GpuDiv, GpuExp, GpuExp2, GpuLog, GpuLog2, GpuMul, GpuNeg, GpuSin,
    GpuSqrt, GpuSub, GpuTan, GpuTanh, GpuTensor, GpuTensorLike,
};
use crate::GpuOr;

// ============================================================================
// Binary Operations for concrete tensor types: Add, Sub, Mul, Div
// ============================================================================

// Add: &GpuOr<ConcreteTensor, GpuTensor> + &GpuOr<ConcreteTensor, GpuTensor>
impl<const R: usize, T> Add<&GpuOr<fusor_cpu::ConcreteTensor<T, R>, GpuTensor<R, T>>>
    for &GpuOr<fusor_cpu::ConcreteTensor<T, R>, GpuTensor<R, T>>
where
    T: fusor_cpu::SimdElement
        + fusor_core::DataType
        + Default
        + std::ops::Add<Output = T>
        + Send
        + Sync,
    fusor_cpu::AddOp: fusor_cpu::SimdBinaryOp<T>,
{
    type Output = GpuOr<
        fusor_cpu::Add<T, R, fusor_cpu::ConcreteTensor<T, R>, fusor_cpu::ConcreteTensor<T, R>>,
        GpuAdd<R, T, GpuTensor<R, T>, GpuTensor<R, T>>,
    >;

    fn add(
        self,
        rhs: &GpuOr<fusor_cpu::ConcreteTensor<T, R>, GpuTensor<R, T>>,
    ) -> Self::Output {
        match (self, rhs) {
            (GpuOr::Cpu(l), GpuOr::Cpu(r)) => {
                GpuOr::Cpu(fusor_cpu::Add::new(l.clone(), r.clone()))
            }
            (GpuOr::Gpu(l), GpuOr::Gpu(r)) => GpuOr::Gpu(GpuAdd::new(l, r)),
            _ => panic!("Cannot add tensors from different devices"),
        }
    }
}

// Sub: &GpuOr - &GpuOr
impl<const R: usize, T> Sub<&GpuOr<fusor_cpu::ConcreteTensor<T, R>, GpuTensor<R, T>>>
    for &GpuOr<fusor_cpu::ConcreteTensor<T, R>, GpuTensor<R, T>>
where
    T: fusor_cpu::SimdElement
        + fusor_core::DataType
        + Default
        + std::ops::Sub<Output = T>
        + Send
        + Sync,
    fusor_cpu::SubOp: fusor_cpu::SimdBinaryOp<T>,
{
    type Output = GpuOr<
        fusor_cpu::Sub<T, R, fusor_cpu::ConcreteTensor<T, R>, fusor_cpu::ConcreteTensor<T, R>>,
        GpuSub<R, T, GpuTensor<R, T>, GpuTensor<R, T>>,
    >;

    fn sub(
        self,
        rhs: &GpuOr<fusor_cpu::ConcreteTensor<T, R>, GpuTensor<R, T>>,
    ) -> Self::Output {
        match (self, rhs) {
            (GpuOr::Cpu(l), GpuOr::Cpu(r)) => {
                GpuOr::Cpu(fusor_cpu::Sub::new(l.clone(), r.clone()))
            }
            (GpuOr::Gpu(l), GpuOr::Gpu(r)) => GpuOr::Gpu(GpuSub::new(l, r)),
            _ => panic!("Cannot subtract tensors from different devices"),
        }
    }
}

// Mul: &GpuOr * &GpuOr
impl<const R: usize, T> Mul<&GpuOr<fusor_cpu::ConcreteTensor<T, R>, GpuTensor<R, T>>>
    for &GpuOr<fusor_cpu::ConcreteTensor<T, R>, GpuTensor<R, T>>
where
    T: fusor_cpu::SimdElement
        + fusor_core::DataType
        + Default
        + std::ops::Mul<Output = T>
        + Send
        + Sync,
    fusor_cpu::MulOp: fusor_cpu::SimdBinaryOp<T>,
{
    type Output = GpuOr<
        fusor_cpu::Mul<T, R, fusor_cpu::ConcreteTensor<T, R>, fusor_cpu::ConcreteTensor<T, R>>,
        GpuMul<R, T, GpuTensor<R, T>, GpuTensor<R, T>>,
    >;

    fn mul(
        self,
        rhs: &GpuOr<fusor_cpu::ConcreteTensor<T, R>, GpuTensor<R, T>>,
    ) -> Self::Output {
        match (self, rhs) {
            (GpuOr::Cpu(l), GpuOr::Cpu(r)) => {
                GpuOr::Cpu(fusor_cpu::Mul::new(l.clone(), r.clone()))
            }
            (GpuOr::Gpu(l), GpuOr::Gpu(r)) => GpuOr::Gpu(GpuMul::new(l, r)),
            _ => panic!("Cannot multiply tensors from different devices"),
        }
    }
}

// Div: &GpuOr / &GpuOr
impl<const R: usize, T> Div<&GpuOr<fusor_cpu::ConcreteTensor<T, R>, GpuTensor<R, T>>>
    for &GpuOr<fusor_cpu::ConcreteTensor<T, R>, GpuTensor<R, T>>
where
    T: fusor_cpu::SimdElement
        + fusor_core::DataType
        + Default
        + std::ops::Div<Output = T>
        + Send
        + Sync,
    fusor_cpu::DivOp: fusor_cpu::SimdBinaryOp<T>,
{
    type Output = GpuOr<
        fusor_cpu::Div<T, R, fusor_cpu::ConcreteTensor<T, R>, fusor_cpu::ConcreteTensor<T, R>>,
        GpuDiv<R, T, GpuTensor<R, T>, GpuTensor<R, T>>,
    >;

    fn div(
        self,
        rhs: &GpuOr<fusor_cpu::ConcreteTensor<T, R>, GpuTensor<R, T>>,
    ) -> Self::Output {
        match (self, rhs) {
            (GpuOr::Cpu(l), GpuOr::Cpu(r)) => {
                GpuOr::Cpu(fusor_cpu::Div::new(l.clone(), r.clone()))
            }
            (GpuOr::Gpu(l), GpuOr::Gpu(r)) => GpuOr::Gpu(GpuDiv::new(l, r)),
            _ => panic!("Cannot divide tensors from different devices"),
        }
    }
}

// ============================================================================
// Unary Operations: Neg
// ============================================================================

// Neg: -&GpuOr<ConcreteTensor, GpuTensor>
impl<const R: usize, T> Neg for &GpuOr<fusor_cpu::ConcreteTensor<T, R>, GpuTensor<R, T>>
where
    T: fusor_cpu::SimdElement
        + fusor_core::DataType
        + Default
        + std::ops::Neg<Output = T>
        + Send
        + Sync,
    fusor_cpu::NegOp: fusor_cpu::SimdUnaryOp<T>,
{
    type Output = GpuOr<
        fusor_cpu::Neg<T, R, fusor_cpu::ConcreteTensor<T, R>>,
        GpuNeg<R, T, GpuTensor<R, T>>,
    >;

    fn neg(self) -> Self::Output {
        match self {
            GpuOr::Cpu(t) => GpuOr::Cpu(fusor_cpu::Neg::new(t.clone())),
            GpuOr::Gpu(t) => GpuOr::Gpu(GpuNeg::new(t)),
        }
    }
}

// ============================================================================
// Element-wise Operations for concrete tensor types
// ============================================================================

impl<const R: usize, T> GpuOr<fusor_cpu::ConcreteTensor<T, R>, GpuTensor<R, T>>
where
    T: fusor_cpu::SimdElement + fusor_core::FloatDataType + Default + Send + Sync,
{
    /// Element-wise absolute value.
    pub fn abs(
        &self,
    ) -> GpuOr<fusor_cpu::Abs<T, R, fusor_cpu::ConcreteTensor<T, R>>, GpuAbs<R, T, GpuTensor<R, T>>>
    where
        fusor_cpu::AbsOp: fusor_cpu::SimdUnaryOp<T>,
    {
        match self {
            GpuOr::Cpu(t) => GpuOr::Cpu(fusor_cpu::Abs::new(t.clone())),
            GpuOr::Gpu(t) => GpuOr::Gpu(GpuAbs::new(t)),
        }
    }

    /// Element-wise square root.
    pub fn sqrt(
        &self,
    ) -> GpuOr<fusor_cpu::Sqrt<T, R, fusor_cpu::ConcreteTensor<T, R>>, GpuSqrt<R, T, GpuTensor<R, T>>>
    where
        fusor_cpu::SqrtOp: fusor_cpu::SimdUnaryOp<T>,
    {
        match self {
            GpuOr::Cpu(t) => GpuOr::Cpu(fusor_cpu::Sqrt::new(t.clone())),
            GpuOr::Gpu(t) => GpuOr::Gpu(GpuSqrt::new(t)),
        }
    }

    /// Element-wise exponential (e^x).
    pub fn exp(
        &self,
    ) -> GpuOr<fusor_cpu::Exp<T, R, fusor_cpu::ConcreteTensor<T, R>>, GpuExp<R, T, GpuTensor<R, T>>>
    where
        fusor_cpu::ExpOp: fusor_cpu::SimdUnaryOp<T>,
    {
        match self {
            GpuOr::Cpu(t) => GpuOr::Cpu(fusor_cpu::Exp::new(t.clone())),
            GpuOr::Gpu(t) => GpuOr::Gpu(GpuExp::new(t)),
        }
    }

    /// Element-wise base-2 exponential (2^x).
    pub fn exp2(
        &self,
    ) -> GpuOr<fusor_cpu::Exp2<T, R, fusor_cpu::ConcreteTensor<T, R>>, GpuExp2<R, T, GpuTensor<R, T>>>
    where
        fusor_cpu::Exp2Op: fusor_cpu::SimdUnaryOp<T>,
    {
        match self {
            GpuOr::Cpu(t) => GpuOr::Cpu(fusor_cpu::Exp2::new(t.clone())),
            GpuOr::Gpu(t) => GpuOr::Gpu(GpuExp2::new(t)),
        }
    }

    /// Element-wise natural logarithm.
    pub fn log(
        &self,
    ) -> GpuOr<fusor_cpu::Log<T, R, fusor_cpu::ConcreteTensor<T, R>>, GpuLog<R, T, GpuTensor<R, T>>>
    where
        fusor_cpu::LogOp: fusor_cpu::SimdUnaryOp<T>,
    {
        match self {
            GpuOr::Cpu(t) => GpuOr::Cpu(fusor_cpu::Log::new(t.clone())),
            GpuOr::Gpu(t) => GpuOr::Gpu(GpuLog::new(t)),
        }
    }

    /// Element-wise base-2 logarithm.
    pub fn log2(
        &self,
    ) -> GpuOr<fusor_cpu::Log2<T, R, fusor_cpu::ConcreteTensor<T, R>>, GpuLog2<R, T, GpuTensor<R, T>>>
    where
        fusor_cpu::Log2Op: fusor_cpu::SimdUnaryOp<T>,
    {
        match self {
            GpuOr::Cpu(t) => GpuOr::Cpu(fusor_cpu::Log2::new(t.clone())),
            GpuOr::Gpu(t) => GpuOr::Gpu(GpuLog2::new(t)),
        }
    }

    /// Element-wise sine.
    pub fn sin(
        &self,
    ) -> GpuOr<fusor_cpu::Sin<T, R, fusor_cpu::ConcreteTensor<T, R>>, GpuSin<R, T, GpuTensor<R, T>>>
    where
        fusor_cpu::SinOp: fusor_cpu::SimdUnaryOp<T>,
    {
        match self {
            GpuOr::Cpu(t) => GpuOr::Cpu(fusor_cpu::Sin::new(t.clone())),
            GpuOr::Gpu(t) => GpuOr::Gpu(GpuSin::new(t)),
        }
    }

    /// Element-wise cosine.
    pub fn cos(
        &self,
    ) -> GpuOr<fusor_cpu::Cos<T, R, fusor_cpu::ConcreteTensor<T, R>>, GpuCos<R, T, GpuTensor<R, T>>>
    where
        fusor_cpu::CosOp: fusor_cpu::SimdUnaryOp<T>,
    {
        match self {
            GpuOr::Cpu(t) => GpuOr::Cpu(fusor_cpu::Cos::new(t.clone())),
            GpuOr::Gpu(t) => GpuOr::Gpu(GpuCos::new(t)),
        }
    }

    /// Element-wise tangent.
    pub fn tan(
        &self,
    ) -> GpuOr<fusor_cpu::Tan<T, R, fusor_cpu::ConcreteTensor<T, R>>, GpuTan<R, T, GpuTensor<R, T>>>
    where
        fusor_cpu::TanOp: fusor_cpu::SimdUnaryOp<T>,
    {
        match self {
            GpuOr::Cpu(t) => GpuOr::Cpu(fusor_cpu::Tan::new(t.clone())),
            GpuOr::Gpu(t) => GpuOr::Gpu(GpuTan::new(t)),
        }
    }

    /// Element-wise hyperbolic tangent.
    pub fn tanh(
        &self,
    ) -> GpuOr<fusor_cpu::Tanh<T, R, fusor_cpu::ConcreteTensor<T, R>>, GpuTanh<R, T, GpuTensor<R, T>>>
    where
        fusor_cpu::TanhOp: fusor_cpu::SimdUnaryOp<T>,
    {
        match self {
            GpuOr::Cpu(t) => GpuOr::Cpu(fusor_cpu::Tanh::new(t.clone())),
            GpuOr::Gpu(t) => GpuOr::Gpu(GpuTanh::new(t)),
        }
    }

    /// Resolve the tensor synchronously, executing any pending operations.
    ///
    /// For CPU tensors, this just clones since ConcreteTensor is already resolved.
    /// For GPU tensors, this returns the tensor itself.
    pub fn resolve(&self) -> GpuOr<fusor_cpu::ConcreteTensor<T, R>, GpuTensor<R, T>> {
        self.clone()
    }

    /// Get the data as a Vec asynchronously.
    pub async fn to_vec(&self) -> Vec<T> {
        match self {
            GpuOr::Cpu(t) => fusor_cpu::ResolvedTensor::data(t).to_vec(),
            GpuOr::Gpu(t) => t.to_vec().await,
        }
    }

    /// Get the data as a Vec synchronously, blocking for GPU tensors.
    pub fn to_vec_blocking(&self) -> Vec<T> {
        pollster::block_on(self.to_vec())
    }

    /// Returns the shape of the tensor.
    pub fn shape(&self) -> &[usize] {
        match self {
            GpuOr::Cpu(t) => fusor_cpu::ResolvedTensor::shape(t),
            GpuOr::Gpu(t) => t.shape(),
        }
    }
}

// ============================================================================
// Resolve / to_vec for expression types (Add, Sub, etc.)
// ============================================================================

// Helper macro to add resolve/to_vec for binary expression GpuOr types
macro_rules! impl_binary_expr_resolve {
    ($cpu_type:ident, $gpu_type:ident) => {
        impl<const R: usize, T>
            GpuOr<
                fusor_cpu::$cpu_type<
                    T,
                    R,
                    fusor_cpu::ConcreteTensor<T, R>,
                    fusor_cpu::ConcreteTensor<T, R>,
                >,
                $gpu_type<R, T, GpuTensor<R, T>, GpuTensor<R, T>>,
            >
        where
            T: fusor_cpu::SimdElement + fusor_core::DataType + Default + Send + Sync,
        {
            /// Resolve the expression to a concrete tensor.
            pub fn resolve(
                &self,
            ) -> GpuOr<fusor_cpu::ConcreteTensor<T, R>, GpuTensor<R, T>>
            where
                fusor_cpu::$cpu_type<
                    T,
                    R,
                    fusor_cpu::ConcreteTensor<T, R>,
                    fusor_cpu::ConcreteTensor<T, R>,
                >: fusor_cpu::ResolveTensor<Concrete = fusor_cpu::ConcreteTensor<T, R>>,
            {
                match self {
                    GpuOr::Cpu(t) => {
                        GpuOr::Cpu(fusor_cpu::ResolveTensor::to_concrete(t))
                    }
                    GpuOr::Gpu(t) => {
                        GpuOr::Gpu(GpuTensor::new(t.as_core().clone()))
                    }
                }
            }

            /// Get the data as a Vec asynchronously.
            pub async fn to_vec(&self) -> Vec<T>
            where
                fusor_cpu::$cpu_type<
                    T,
                    R,
                    fusor_cpu::ConcreteTensor<T, R>,
                    fusor_cpu::ConcreteTensor<T, R>,
                >: fusor_cpu::ResolveTensor<Concrete = fusor_cpu::ConcreteTensor<T, R>>,
            {
                match self {
                    GpuOr::Cpu(t) => {
                        let concrete = fusor_cpu::ResolveTensor::to_concrete(t);
                        fusor_cpu::ResolvedTensor::data(&concrete).to_vec()
                    }
                    GpuOr::Gpu(t) => t.to_vec().await,
                }
            }

            /// Get the data as a Vec synchronously.
            pub fn to_vec_blocking(&self) -> Vec<T>
            where
                fusor_cpu::$cpu_type<
                    T,
                    R,
                    fusor_cpu::ConcreteTensor<T, R>,
                    fusor_cpu::ConcreteTensor<T, R>,
                >: fusor_cpu::ResolveTensor<Concrete = fusor_cpu::ConcreteTensor<T, R>>,
            {
                pollster::block_on(self.to_vec())
            }
        }
    };
}

impl_binary_expr_resolve!(Add, GpuAdd);
impl_binary_expr_resolve!(Sub, GpuSub);
impl_binary_expr_resolve!(Mul, GpuMul);
impl_binary_expr_resolve!(Div, GpuDiv);

// Helper macro to add resolve/to_vec for unary expression GpuOr types
macro_rules! impl_unary_expr_resolve {
    ($cpu_type:ident, $gpu_type:ident) => {
        impl<const R: usize, T>
            GpuOr<
                fusor_cpu::$cpu_type<T, R, fusor_cpu::ConcreteTensor<T, R>>,
                $gpu_type<R, T, GpuTensor<R, T>>,
            >
        where
            T: fusor_cpu::SimdElement + fusor_core::DataType + Default + Send + Sync,
        {
            /// Resolve the expression to a concrete tensor.
            pub fn resolve(&self) -> GpuOr<fusor_cpu::ConcreteTensor<T, R>, GpuTensor<R, T>>
            where
                fusor_cpu::$cpu_type<T, R, fusor_cpu::ConcreteTensor<T, R>>:
                    fusor_cpu::ResolveTensor<Concrete = fusor_cpu::ConcreteTensor<T, R>>,
            {
                match self {
                    GpuOr::Cpu(t) => GpuOr::Cpu(fusor_cpu::ResolveTensor::to_concrete(t)),
                    GpuOr::Gpu(t) => GpuOr::Gpu(GpuTensor::new(t.as_core().clone())),
                }
            }

            /// Get the data as a Vec asynchronously.
            pub async fn to_vec(&self) -> Vec<T>
            where
                fusor_cpu::$cpu_type<T, R, fusor_cpu::ConcreteTensor<T, R>>:
                    fusor_cpu::ResolveTensor<Concrete = fusor_cpu::ConcreteTensor<T, R>>,
            {
                match self {
                    GpuOr::Cpu(t) => {
                        let concrete = fusor_cpu::ResolveTensor::to_concrete(t);
                        fusor_cpu::ResolvedTensor::data(&concrete).to_vec()
                    }
                    GpuOr::Gpu(t) => t.to_vec().await,
                }
            }

            /// Get the data as a Vec synchronously.
            pub fn to_vec_blocking(&self) -> Vec<T>
            where
                fusor_cpu::$cpu_type<T, R, fusor_cpu::ConcreteTensor<T, R>>:
                    fusor_cpu::ResolveTensor<Concrete = fusor_cpu::ConcreteTensor<T, R>>,
            {
                pollster::block_on(self.to_vec())
            }
        }
    };
}

impl_unary_expr_resolve!(Neg, GpuNeg);
impl_unary_expr_resolve!(Abs, GpuAbs);
impl_unary_expr_resolve!(Sqrt, GpuSqrt);
impl_unary_expr_resolve!(Exp, GpuExp);
impl_unary_expr_resolve!(Exp2, GpuExp2);
impl_unary_expr_resolve!(Log, GpuLog);
impl_unary_expr_resolve!(Log2, GpuLog2);
impl_unary_expr_resolve!(Sin, GpuSin);
impl_unary_expr_resolve!(Cos, GpuCos);
impl_unary_expr_resolve!(Tan, GpuTan);
impl_unary_expr_resolve!(Tanh, GpuTanh);
