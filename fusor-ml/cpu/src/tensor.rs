//! Tensor - the unified interface over different tensor backends

use std::ops::{Add as StdAdd, Div as StdDiv, Mul as StdMul, Neg as StdNeg, Range, Sub as StdSub};

use pulp::Simd;

use crate::cast::{cast_tensor, CastTo};
use crate::comparison::{comparison_scalar_op_ref, comparison_tensor_op_ref};
use crate::comparison::{EqOp, GteOp, GtOp, LteOp, LtOp, NeOp, SimdComparisonOp};
use crate::conditional::{where_cond_ref, IsNonZero};
use crate::elementwise::{unary_tensor_op_ref, AbsOp, CosOp, ExpOp, LogOp, NegOp, SimdUnaryOp, SinOp, SqrtOp, TanhOp};
use crate::expr::Expr;
use crate::index::index_select_ref;
use crate::slice_assign::slice_assign_ref;
use crate::matmul::MatmulImpl;
use crate::pairwise::{AddOp, DivOp, MulOp, SimdBinaryOp, SubOp};
use crate::reduce::{reduce_tensor_axis, reduce_tensor_op, MaxOp, MinOp, ProdOp, SimdReduceOp, SumOp};
use crate::{elementwise, pairwise, ConcreteTensor, CpuMappedBuffer, LastRank, ResolveTensor, ResolvedTensor, SimdElement, TensorBacking, TensorSlice};

/// A tensor wrapper that provides a unified interface over different tensor backends.
#[derive(Clone)]
pub struct Tensor<const R: usize, T: TensorBacking<R>> {
    inner: T,
}

impl<const R: usize, T: TensorBacking<R>> Tensor<R, T> {
    /// Create a new tensor from an inner backing type.
    pub fn new(inner: T) -> Self {
        Self { inner }
    }

    /// Get a reference to the inner backing type.
    pub fn inner(&self) -> &T {
        &self.inner
    }

    /// Get a mutable reference to the inner backing type.
    pub fn inner_mut(&mut self) -> &mut T {
        &mut self.inner
    }

    /// Consume the tensor and return the inner backing type.
    pub fn into_inner(self) -> T {
        self.inner
    }
}

// Constructors for Tensor that create ConcreteTensor backing
impl<const R: usize, E: SimdElement> Tensor<R, ConcreteTensor<E, R>> {
    /// Create a new tensor filled with zeros
    pub fn zeros(shape: [usize; R]) -> Self
    where
        E: Default,
    {
        Self::new(ConcreteTensor::zeros(shape))
    }

    /// Create a new tensor from existing data
    pub fn from_slice(shape: [usize; R], data: &[E]) -> Self {
        Self::new(ConcreteTensor::from_slice(shape, data))
    }

    /// Get element at logical indices
    pub fn get(&self, indices: [usize; R]) -> E {
        self.inner.get(indices)
    }

    /// Set element at logical indices
    pub fn set(&mut self, indices: [usize; R], value: E) {
        self.inner.set(indices, value)
    }
}

// Methods available on any Tensor with ResolveTensor inner
impl<const R: usize, E, T> Tensor<R, T>
where
    E: SimdElement,
    T: TensorBacking<R, Elem = E> + ResolveTensor<R, Elem = E>,
{
    /// Materialize the tensor to a ConcreteTensor
    pub fn eval(&self) -> Tensor<R, ConcreteTensor<E, R>> {
        Tensor::new(self.inner.to_concrete())
    }

    /// Absolute value element-wise
    #[inline]
    pub fn abs(&self) -> Tensor<R, ConcreteTensor<E, R>>
    where
        E: Default,
        AbsOp: SimdUnaryOp<E>,
    {
        Tensor::new(unary_tensor_op_ref::<E, R, AbsOp>(&self.inner.to_concrete()))
    }

    /// Square root element-wise
    #[inline]
    pub fn sqrt(&self) -> Tensor<R, ConcreteTensor<E, R>>
    where
        E: Default,
        SqrtOp: SimdUnaryOp<E>,
    {
        Tensor::new(unary_tensor_op_ref::<E, R, SqrtOp>(&self.inner.to_concrete()))
    }

    /// Exponential (e^x) element-wise
    #[inline]
    pub fn exp(&self) -> Tensor<R, ConcreteTensor<E, R>>
    where
        E: Default,
        ExpOp: SimdUnaryOp<E>,
    {
        Tensor::new(unary_tensor_op_ref::<E, R, ExpOp>(&self.inner.to_concrete()))
    }

    /// Natural logarithm element-wise
    #[inline]
    pub fn log(&self) -> Tensor<R, ConcreteTensor<E, R>>
    where
        E: Default,
        LogOp: SimdUnaryOp<E>,
    {
        Tensor::new(unary_tensor_op_ref::<E, R, LogOp>(&self.inner.to_concrete()))
    }

    /// Sine element-wise
    #[inline]
    pub fn sin(&self) -> Tensor<R, ConcreteTensor<E, R>>
    where
        E: Default,
        SinOp: SimdUnaryOp<E>,
    {
        Tensor::new(unary_tensor_op_ref::<E, R, SinOp>(&self.inner.to_concrete()))
    }

    /// Cosine element-wise
    #[inline]
    pub fn cos(&self) -> Tensor<R, ConcreteTensor<E, R>>
    where
        E: Default,
        CosOp: SimdUnaryOp<E>,
    {
        Tensor::new(unary_tensor_op_ref::<E, R, CosOp>(&self.inner.to_concrete()))
    }

    /// Hyperbolic tangent element-wise
    #[inline]
    pub fn tanh(&self) -> Tensor<R, ConcreteTensor<E, R>>
    where
        E: Default,
        TanhOp: SimdUnaryOp<E>,
    {
        Tensor::new(unary_tensor_op_ref::<E, R, TanhOp>(&self.inner.to_concrete()))
    }

    /// Sum all elements in the tensor
    #[inline]
    pub fn sum(&self) -> E
    where
        SumOp: SimdReduceOp<E>,
    {
        reduce_tensor_op::<E, R, SumOp>(&self.inner.to_concrete())
    }

    /// Find the maximum element in the tensor
    #[inline]
    pub fn max(&self) -> E
    where
        MaxOp: SimdReduceOp<E>,
    {
        reduce_tensor_op::<E, R, MaxOp>(&self.inner.to_concrete())
    }

    /// Find the minimum element in the tensor
    #[inline]
    pub fn min(&self) -> E
    where
        MinOp: SimdReduceOp<E>,
    {
        reduce_tensor_op::<E, R, MinOp>(&self.inner.to_concrete())
    }

    /// Multiply all elements in the tensor
    #[inline]
    pub fn prod(&self) -> E
    where
        ProdOp: SimdReduceOp<E>,
    {
        reduce_tensor_op::<E, R, ProdOp>(&self.inner.to_concrete())
    }

    /// Element-wise equality comparison
    #[inline]
    pub fn eq(&self, rhs: &Tensor<R, impl TensorBacking<R, Elem = E> + ResolveTensor<R, Elem = E>>) -> Tensor<R, ConcreteTensor<E, R>>
    where
        E: Default,
        EqOp: SimdComparisonOp<E>,
    {
        Tensor::new(comparison_tensor_op_ref::<E, R, EqOp>(&self.inner.to_concrete(), &rhs.inner.to_concrete()))
    }

    /// Element-wise less than comparison
    #[inline]
    pub fn lt(&self, rhs: &Tensor<R, impl TensorBacking<R, Elem = E> + ResolveTensor<R, Elem = E>>) -> Tensor<R, ConcreteTensor<E, R>>
    where
        E: Default,
        LtOp: SimdComparisonOp<E>,
    {
        Tensor::new(comparison_tensor_op_ref::<E, R, LtOp>(&self.inner.to_concrete(), &rhs.inner.to_concrete()))
    }

    /// Element-wise greater than comparison
    #[inline]
    pub fn gt(&self, rhs: &Tensor<R, impl TensorBacking<R, Elem = E> + ResolveTensor<R, Elem = E>>) -> Tensor<R, ConcreteTensor<E, R>>
    where
        E: Default,
        GtOp: SimdComparisonOp<E>,
    {
        Tensor::new(comparison_tensor_op_ref::<E, R, GtOp>(&self.inner.to_concrete(), &rhs.inner.to_concrete()))
    }

    /// Element-wise not equal comparison
    #[inline]
    pub fn ne(&self, rhs: &Tensor<R, impl TensorBacking<R, Elem = E> + ResolveTensor<R, Elem = E>>) -> Tensor<R, ConcreteTensor<E, R>>
    where
        E: Default,
        NeOp: SimdComparisonOp<E>,
    {
        Tensor::new(comparison_tensor_op_ref::<E, R, NeOp>(&self.inner.to_concrete(), &rhs.inner.to_concrete()))
    }

    /// Element-wise less than or equal comparison
    #[inline]
    pub fn lte(&self, rhs: &Tensor<R, impl TensorBacking<R, Elem = E> + ResolveTensor<R, Elem = E>>) -> Tensor<R, ConcreteTensor<E, R>>
    where
        E: Default,
        LteOp: SimdComparisonOp<E>,
    {
        Tensor::new(comparison_tensor_op_ref::<E, R, LteOp>(&self.inner.to_concrete(), &rhs.inner.to_concrete()))
    }

    /// Element-wise greater than or equal comparison
    #[inline]
    pub fn gte(&self, rhs: &Tensor<R, impl TensorBacking<R, Elem = E> + ResolveTensor<R, Elem = E>>) -> Tensor<R, ConcreteTensor<E, R>>
    where
        E: Default,
        GteOp: SimdComparisonOp<E>,
    {
        Tensor::new(comparison_tensor_op_ref::<E, R, GteOp>(&self.inner.to_concrete(), &rhs.inner.to_concrete()))
    }

    /// Compare tensor elements with scalar for equality
    #[inline]
    pub fn eq_scalar(&self, scalar: E) -> Tensor<R, ConcreteTensor<E, R>>
    where
        EqOp: SimdComparisonOp<E>,
    {
        Tensor::new(comparison_scalar_op_ref::<E, R, EqOp>(&self.inner.to_concrete(), scalar))
    }

    /// Compare tensor elements with scalar for inequality
    #[inline]
    pub fn ne_scalar(&self, scalar: E) -> Tensor<R, ConcreteTensor<E, R>>
    where
        NeOp: SimdComparisonOp<E>,
    {
        Tensor::new(comparison_scalar_op_ref::<E, R, NeOp>(&self.inner.to_concrete(), scalar))
    }

    /// Compare tensor elements with scalar for less than
    #[inline]
    pub fn lt_scalar(&self, scalar: E) -> Tensor<R, ConcreteTensor<E, R>>
    where
        LtOp: SimdComparisonOp<E>,
    {
        Tensor::new(comparison_scalar_op_ref::<E, R, LtOp>(&self.inner.to_concrete(), scalar))
    }

    /// Compare tensor elements with scalar for less than or equal
    #[inline]
    pub fn lte_scalar(&self, scalar: E) -> Tensor<R, ConcreteTensor<E, R>>
    where
        LteOp: SimdComparisonOp<E>,
    {
        Tensor::new(comparison_scalar_op_ref::<E, R, LteOp>(&self.inner.to_concrete(), scalar))
    }

    /// Compare tensor elements with scalar for greater than
    #[inline]
    pub fn gt_scalar(&self, scalar: E) -> Tensor<R, ConcreteTensor<E, R>>
    where
        GtOp: SimdComparisonOp<E>,
    {
        Tensor::new(comparison_scalar_op_ref::<E, R, GtOp>(&self.inner.to_concrete(), scalar))
    }

    /// Compare tensor elements with scalar for greater than or equal
    #[inline]
    pub fn gte_scalar(&self, scalar: E) -> Tensor<R, ConcreteTensor<E, R>>
    where
        GteOp: SimdComparisonOp<E>,
    {
        Tensor::new(comparison_scalar_op_ref::<E, R, GteOp>(&self.inner.to_concrete(), scalar))
    }

    /// Conditional selection: where self != 0, select on_true, else on_false
    #[inline]
    pub fn where_cond(
        &self,
        on_true: &Tensor<R, impl TensorBacking<R, Elem = E> + ResolveTensor<R, Elem = E>>,
        on_false: &Tensor<R, impl TensorBacking<R, Elem = E> + ResolveTensor<R, Elem = E>>,
    ) -> Tensor<R, ConcreteTensor<E, R>>
    where
        E: Default + IsNonZero,
    {
        Tensor::new(where_cond_ref(&self.inner.to_concrete(), &on_true.inner.to_concrete(), &on_false.inner.to_concrete()))
    }

    /// Cast tensor to another element type
    #[inline]
    pub fn cast<E2>(&self) -> Tensor<R, ConcreteTensor<E2, R>>
    where
        E: CastTo<E2>,
        E2: SimdElement,
    {
        Tensor::new(cast_tensor(&self.inner.to_concrete()))
    }

    /// Get the tensor data as a TensorSlice for reading.
    ///
    /// This is the CPU equivalent of fusor-core's `as_slice()` method for GPU tensors.
    /// It materializes the tensor (if lazy) and returns a slice view of the data.
    pub fn as_slice(&self) -> TensorSlice<R, E, CpuMappedBuffer> {
        let concrete = self.inner.to_concrete();
        let layout = concrete.layout().clone();
        // Convert the tensor data to raw bytes
        let bytes: Box<[u8]> = bytemuck::cast_slice(concrete.data().as_ref()).into();
        TensorSlice::new(CpuMappedBuffer::new(bytes), layout)
    }

    /// Select elements along a dimension using indices
    #[inline]
    pub fn index_select(&self, dimension: usize, indices: &Tensor<1, ConcreteTensor<u32, 1>>) -> Tensor<R, ConcreteTensor<E, R>> {
        Tensor::new(index_select_ref(&self.inner.to_concrete(), dimension, &indices.inner))
    }

    /// Returns a new tensor with the slice region replaced by values from the value tensor
    ///
    /// # Arguments
    /// * `slices` - Array of ranges specifying the slice region in each dimension
    /// * `value` - Tensor containing values to assign into the slice region
    ///
    /// # Panics
    /// * If slice bounds exceed input tensor dimensions
    /// * If value tensor shape doesn't match slice dimensions
    ///
    /// # Example
    /// ```ignore
    /// let tensor = Tensor::from_slice([3, 3], &[1.0; 9]);
    /// let value = Tensor::from_slice([2, 2], &[10.0; 4]);
    /// let result = tensor.slice_assign([0..2, 0..2], &value);
    /// // result[0..2, 0..2] = value, rest copied from tensor
    /// ```
    #[inline]
    pub fn slice_assign(
        &self,
        slices: [Range<usize>; R],
        value: &Tensor<R, impl TensorBacking<R, Elem = E> + ResolveTensor<R, Elem = E>>,
    ) -> Tensor<R, ConcreteTensor<E, R>> {
        Tensor::new(slice_assign_ref(
            &self.inner.to_concrete(),
            slices,
            &value.inner.to_concrete(),
        ))
    }

    /// Sum along a specific axis, reducing the tensor rank by 1
    #[inline]
    pub fn sum_axis<const OUT_RANK: usize, const AXIS: usize>(&self) -> Tensor<OUT_RANK, ConcreteTensor<E, OUT_RANK>>
    where
        E: Default,
        ConcreteTensor<E, R>: LastRank<OUT_RANK, E>,
        SumOp: SimdReduceOp<E>,
    {
        Tensor::new(reduce_tensor_axis::<E, R, OUT_RANK, AXIS, SumOp>(&self.inner.to_concrete()))
    }

    /// Maximum along a specific axis, reducing the tensor rank by 1
    #[inline]
    pub fn max_axis<const OUT_RANK: usize, const AXIS: usize>(&self) -> Tensor<OUT_RANK, ConcreteTensor<E, OUT_RANK>>
    where
        E: Default,
        ConcreteTensor<E, R>: LastRank<OUT_RANK, E>,
        MaxOp: SimdReduceOp<E>,
    {
        Tensor::new(reduce_tensor_axis::<E, R, OUT_RANK, AXIS, MaxOp>(&self.inner.to_concrete()))
    }

    /// Minimum along a specific axis, reducing the tensor rank by 1
    #[inline]
    pub fn min_axis<const OUT_RANK: usize, const AXIS: usize>(&self) -> Tensor<OUT_RANK, ConcreteTensor<E, OUT_RANK>>
    where
        E: Default,
        ConcreteTensor<E, R>: LastRank<OUT_RANK, E>,
        MinOp: SimdReduceOp<E>,
    {
        Tensor::new(reduce_tensor_axis::<E, R, OUT_RANK, AXIS, MinOp>(&self.inner.to_concrete()))
    }

    /// Product along a specific axis, reducing the tensor rank by 1
    #[inline]
    pub fn prod_axis<const OUT_RANK: usize, const AXIS: usize>(&self) -> Tensor<OUT_RANK, ConcreteTensor<E, OUT_RANK>>
    where
        E: Default,
        ConcreteTensor<E, R>: LastRank<OUT_RANK, E>,
        ProdOp: SimdReduceOp<E>,
    {
        Tensor::new(reduce_tensor_axis::<E, R, OUT_RANK, AXIS, ProdOp>(&self.inner.to_concrete()))
    }
}

/// Trait for float types that support power, min, max, and clamp operations
pub trait FloatOps: SimdElement + PartialOrd {
    fn powf(self, exp: Self) -> Self;
    fn float_max(self, other: Self) -> Self;
    fn float_min(self, other: Self) -> Self;
}

impl FloatOps for f32 {
    #[inline(always)]
    fn powf(self, exp: Self) -> Self { self.powf(exp) }
    #[inline(always)]
    fn float_max(self, other: Self) -> Self { self.max(other) }
    #[inline(always)]
    fn float_min(self, other: Self) -> Self { self.min(other) }
}

impl FloatOps for f64 {
    #[inline(always)]
    fn powf(self, exp: Self) -> Self { self.powf(exp) }
    #[inline(always)]
    fn float_max(self, other: Self) -> Self { self.max(other) }
    #[inline(always)]
    fn float_min(self, other: Self) -> Self { self.min(other) }
}

// Specialized methods for float tensors (f32 and f64)
impl<const R: usize, E, T> Tensor<R, T>
where
    E: FloatOps,
    T: TensorBacking<R, Elem = E> + ResolveTensor<R, Elem = E>,
{
    /// Raise each element to a power
    #[inline]
    pub fn pow_scalar(&self, exponent: E) -> Tensor<R, ConcreteTensor<E, R>> {
        let concrete = self.inner.to_concrete();
        let shape: [usize; R] = ResolvedTensor::shape(&concrete)
            .try_into()
            .expect("Shape length mismatch");
        let mut output = ConcreteTensor::<E, R>::uninit_unchecked(shape);
        for (i, &val) in concrete.data().iter().enumerate() {
            output.data_mut()[i] = val.powf(exponent);
        }
        Tensor::new(output)
    }

    /// Element-wise maximum with a scalar
    #[inline]
    pub fn max_scalar(&self, scalar: E) -> Tensor<R, ConcreteTensor<E, R>> {
        let concrete = self.inner.to_concrete();
        let shape: [usize; R] = ResolvedTensor::shape(&concrete)
            .try_into()
            .expect("Shape length mismatch");
        let mut output = ConcreteTensor::<E, R>::uninit_unchecked(shape);
        for (i, &val) in concrete.data().iter().enumerate() {
            output.data_mut()[i] = val.float_max(scalar);
        }
        Tensor::new(output)
    }

    /// Element-wise minimum with a scalar
    #[inline]
    pub fn min_scalar(&self, scalar: E) -> Tensor<R, ConcreteTensor<E, R>> {
        let concrete = self.inner.to_concrete();
        let shape: [usize; R] = ResolvedTensor::shape(&concrete)
            .try_into()
            .expect("Shape length mismatch");
        let mut output = ConcreteTensor::<E, R>::uninit_unchecked(shape);
        for (i, &val) in concrete.data().iter().enumerate() {
            output.data_mut()[i] = val.float_min(scalar);
        }
        Tensor::new(output)
    }

    /// Clamp each element to a range [min, max]
    #[inline]
    pub fn clamp(&self, min: E, max: E) -> Tensor<R, ConcreteTensor<E, R>> {
        let concrete = self.inner.to_concrete();
        let shape: [usize; R] = ResolvedTensor::shape(&concrete)
            .try_into()
            .expect("Shape length mismatch");
        let mut output = ConcreteTensor::<E, R>::uninit_unchecked(shape);
        for (i, &val) in concrete.data().iter().enumerate() {
            output.data_mut()[i] = val.float_max(min).float_min(max);
        }
        Tensor::new(output)
    }
}

// Specialized methods for 2D tensors (matrix operations)
impl<E, T> Tensor<2, T>
where
    E: SimdElement + MatmulImpl,
    T: TensorBacking<2, Elem = E> + ResolveTensor<2, Elem = E>,
{
    /// Matrix multiplication: self (M x K) @ rhs (K x N) -> (M x N)
    #[inline]
    pub fn matmul(&self, rhs: &Tensor<2, impl TensorBacking<2, Elem = E> + ResolveTensor<2, Elem = E>>) -> Tensor<2, ConcreteTensor<E, 2>> {
        Tensor::new(self.inner.to_concrete().matmul_ref(&rhs.inner.to_concrete()))
    }
}

impl<const R: usize, T: TensorBacking<R>> TensorBacking<R> for Tensor<R, T> {
    type Elem = T::Elem;
}

/// Macro to implement pairwise operators for CPU Tensor.
///
/// Generates all four combinations of owned/reference implementations:
/// - `Tensor op Tensor` (owned + owned)
/// - `&Tensor op &Tensor` (ref + ref)
/// - `Tensor op &Tensor` (owned + ref)
/// - `&Tensor op Tensor` (ref + owned)
macro_rules! impl_cpu_pairwise_op {
    ($std_trait:ident, $method:ident, $pairwise_ty:ident, $simd_op:ident) => {
        // Owned + Owned
        impl<const R: usize, T1, T2> $std_trait<Tensor<R, T2>> for Tensor<R, T1>
        where
            T1: TensorBacking<R>,
            T2: TensorBacking<R, Elem = T1::Elem>,
            T1::Elem: SimdElement + $std_trait<Output = T1::Elem> + Default,
            $simd_op: SimdBinaryOp<T1::Elem>,
        {
            type Output = Tensor<R, pairwise::$pairwise_ty<T1::Elem, R, T1, T2>>;

            fn $method(self, rhs: Tensor<R, T2>) -> Self::Output {
                Tensor::new(pairwise::$pairwise_ty::new(self.inner, rhs.inner))
            }
        }

        // Ref + Ref
        impl<'a, const R: usize, T1, T2> $std_trait<&'a Tensor<R, T2>> for &'a Tensor<R, T1>
        where
            T1: TensorBacking<R>,
            T2: TensorBacking<R, Elem = T1::Elem>,
            T1::Elem: SimdElement + $std_trait<Output = T1::Elem> + Default,
            $simd_op: SimdBinaryOp<T1::Elem>,
        {
            type Output = Tensor<R, pairwise::$pairwise_ty<T1::Elem, R, &'a T1, &'a T2>>;

            fn $method(self, rhs: &'a Tensor<R, T2>) -> Self::Output {
                Tensor::new(pairwise::$pairwise_ty::new(&self.inner, &rhs.inner))
            }
        }

        // Owned + Ref
        impl<'a, const R: usize, T1, T2> $std_trait<&'a Tensor<R, T2>> for Tensor<R, T1>
        where
            T1: TensorBacking<R>,
            T2: TensorBacking<R, Elem = T1::Elem>,
            T1::Elem: SimdElement + $std_trait<Output = T1::Elem> + Default,
            $simd_op: SimdBinaryOp<T1::Elem>,
        {
            type Output = Tensor<R, pairwise::$pairwise_ty<T1::Elem, R, T1, &'a T2>>;

            fn $method(self, rhs: &'a Tensor<R, T2>) -> Self::Output {
                Tensor::new(pairwise::$pairwise_ty::new(self.inner, &rhs.inner))
            }
        }

        // Ref + Owned
        impl<'a, const R: usize, T1, T2> $std_trait<Tensor<R, T2>> for &'a Tensor<R, T1>
        where
            T1: TensorBacking<R>,
            T2: TensorBacking<R, Elem = T1::Elem>,
            T1::Elem: SimdElement + $std_trait<Output = T1::Elem> + Default,
            $simd_op: SimdBinaryOp<T1::Elem>,
        {
            type Output = Tensor<R, pairwise::$pairwise_ty<T1::Elem, R, &'a T1, T2>>;

            fn $method(self, rhs: Tensor<R, T2>) -> Self::Output {
                Tensor::new(pairwise::$pairwise_ty::new(&self.inner, rhs.inner))
            }
        }
    };
}

impl_cpu_pairwise_op!(StdAdd, add, Add, AddOp);
impl_cpu_pairwise_op!(StdSub, sub, Sub, SubOp);
impl_cpu_pairwise_op!(StdMul, mul, Mul, MulOp);
impl_cpu_pairwise_op!(StdDiv, div, Div, DivOp);

// Neg is unary, so handle separately
impl<const R: usize, T> StdNeg for Tensor<R, T>
where
    T: TensorBacking<R>,
    T::Elem: SimdElement + StdNeg<Output = T::Elem> + Default,
    NegOp: SimdUnaryOp<T::Elem>,
{
    type Output = Tensor<R, elementwise::Neg<T::Elem, R, T>>;

    fn neg(self) -> Self::Output {
        Tensor::new(elementwise::Neg::new(self.inner))
    }
}

impl<'a, const R: usize, T> StdNeg for &'a Tensor<R, T>
where
    T: TensorBacking<R>,
    T::Elem: SimdElement + StdNeg<Output = T::Elem> + Default,
    NegOp: SimdUnaryOp<T::Elem>,
{
    type Output = Tensor<R, elementwise::Neg<T::Elem, R, &'a T>>;

    fn neg(self) -> Self::Output {
        Tensor::new(elementwise::Neg::new(&self.inner))
    }
}

// Implement Expr for Tensor to enable evaluation
impl<const R: usize, E, T> Expr for Tensor<R, T>
where
    E: SimdElement,
    T: TensorBacking<R, Elem = E> + Expr<Elem = E>,
{
    type Elem = E;

    #[inline(always)]
    fn eval_scalar(&self, idx: usize) -> Self::Elem {
        self.inner.eval_scalar(idx)
    }

    #[inline(always)]
    fn eval_simd<S: Simd>(&self, simd: S, base_idx: usize) -> E::Simd<S> {
        self.inner.eval_simd(simd, base_idx)
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn shape(&self) -> &[usize] {
        self.inner.shape()
    }

    fn is_contiguous(&self) -> bool {
        self.inner.is_contiguous()
    }
}

// Implement ResolveTensor for Tensor to enable materialization
impl<const R: usize, E, T> ResolveTensor<R> for Tensor<R, T>
where
    E: SimdElement,
    T: TensorBacking<R, Elem = E> + ResolveTensor<R, Elem = E>,
{
    fn to_concrete(&self) -> ConcreteTensor<E, R> {
        self.inner.to_concrete()
    }
}
