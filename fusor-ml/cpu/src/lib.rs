//! CPU tensor operations with SIMD acceleration

use std::mem::MaybeUninit;
use std::ops::{Add as StdAdd, Div as StdDiv, Mul as StdMul, Neg as StdNeg, Sub as StdSub};

use aligned_vec::{ABox, AVec};
use pulp::bytemuck::Pod;
use pulp::Simd;

// Module declarations
mod cast;
mod comparison;
mod conditional;
mod elementwise;
mod expr;
mod index;
mod matmul;
mod pairwise;
mod reduce;

/// Maximum number of SIMD lanes supported for strided tensor gather operations.
/// This covers AVX-512 with 64 x i8 lanes. Current architectures don't exceed this,
/// but this constant provides a clear point for future updates if needed.
pub(crate) const MAX_SIMD_LANES: usize = 64;


// Re-export public types
pub use elementwise::{Abs, Cos, Exp, Exp2, Log, Log2, Neg, Sin, Sqrt, Tan, Tanh};
pub use expr::{materialize_expr, Expr};
pub use pairwise::{Add, Div, Mul, Sub};

// Re-export operation traits and markers for public bounds
pub use elementwise::{
    AbsOp, CosOp, Exp2Op, ExpOp, Log2Op, LogOp, NegOp, SimdUnaryOp, SinOp, SqrtOp, TanOp, TanhOp,
};
pub use pairwise::{AddOp, DivOp, MulOp, SimdBinaryOp, SubOp};
pub use reduce::{MaxOp, MinOp, ProdOp, SimdReduceOp, SumOp};
pub use comparison::{EqOp, NeOp, LtOp, LteOp, GtOp, GteOp, SimdComparisonOp};
pub use conditional::IsNonZero;
pub use cast::CastTo;

// Internal imports from modules
use cast::cast_tensor;
use comparison::{comparison_tensor_op_ref, comparison_scalar_op_ref};
use conditional::where_cond_ref;
use elementwise::unary_tensor_op_ref;
use index::index_select_ref;
use pairwise::binary_tensor_op_ref;
use reduce::{reduce_tensor_axis, reduce_tensor_op};

// Trait for mapping tensor to its one-rank-smaller type (for axis reductions)
pub trait LastRankInner {
    type LastRank;
}

pub trait LastRank<const R: usize, T: SimdElement>:
    LastRankInner<LastRank = ConcreteTensor<T, R>>
{
}

impl<const R: usize, T: SimdElement, X> LastRank<R, T> for X where
    X: LastRankInner<LastRank = ConcreteTensor<T, R>>
{
}

// Macro to generate LastRankInner implementations for each rank
macro_rules! impl_last_rank {
    ($($R:literal),*) => {
        $(
            impl<T: SimdElement> LastRankInner for ConcreteTensor<T, $R> {
                type LastRank = ConcreteTensor<T, { $R - 1 }>;
            }
        )*
    };
}

// Generate for ranks 1-10
impl_last_rank!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);

pub trait TensorBacking<const R: usize> {
    type Elem: SimdElement;
}

pub trait ResolveTensor<const R: usize, M = ()>: TensorBacking<R> {
    fn to_concrete(&self) -> ConcreteTensor<Self::Elem, R>;
}

pub trait ResolvedTensor<const R: usize>: TensorBacking<R> {
    fn shape(&self) -> &[usize];
    fn strides(&self) -> &[usize];
    fn offset(&self) -> usize;
    fn data(&self) -> &ABox<[Self::Elem]>;
    fn data_mut(&mut self) -> &mut ABox<[Self::Elem]>;
}

/// A tensor wrapper that provides a unified interface over different tensor backends.
#[derive(Clone)]
pub struct Tensor<T: TensorBacking<R>, const R: usize> {
    inner: T,
}

impl<T: TensorBacking<R>, const R: usize> Tensor<T, R> {
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
impl<E: SimdElement, const R: usize> Tensor<ConcreteTensor<E, R>, R> {
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
impl<E, T, const R: usize> Tensor<T, R>
where
    E: SimdElement,
    T: TensorBacking<R, Elem = E> + ResolveTensor<R, Elem = E>,
{
    /// Materialize the tensor to a ConcreteTensor
    pub fn eval(&self) -> Tensor<ConcreteTensor<E, R>, R> {
        Tensor::new(self.inner.to_concrete())
    }

    /// Add two tensors element-wise
    #[inline]
    pub fn add_ref(&self, rhs: &Tensor<impl TensorBacking<R, Elem = E> + ResolveTensor<R, Elem = E>, R>) -> Tensor<ConcreteTensor<E, R>, R>
    where
        E: Default + StdAdd<Output = E>,
        AddOp: SimdBinaryOp<E>,
    {
        Tensor::new(binary_tensor_op_ref::<E, R, AddOp>(&self.inner.to_concrete(), &rhs.inner.to_concrete()))
    }

    /// Subtract two tensors element-wise
    #[inline]
    pub fn sub_ref(&self, rhs: &Tensor<impl TensorBacking<R, Elem = E> + ResolveTensor<R, Elem = E>, R>) -> Tensor<ConcreteTensor<E, R>, R>
    where
        E: Default + StdSub<Output = E>,
        SubOp: SimdBinaryOp<E>,
    {
        Tensor::new(binary_tensor_op_ref::<E, R, SubOp>(&self.inner.to_concrete(), &rhs.inner.to_concrete()))
    }

    /// Multiply two tensors element-wise
    #[inline]
    pub fn mul_ref(&self, rhs: &Tensor<impl TensorBacking<R, Elem = E> + ResolveTensor<R, Elem = E>, R>) -> Tensor<ConcreteTensor<E, R>, R>
    where
        E: Default + StdMul<Output = E>,
        MulOp: SimdBinaryOp<E>,
    {
        Tensor::new(binary_tensor_op_ref::<E, R, MulOp>(&self.inner.to_concrete(), &rhs.inner.to_concrete()))
    }

    /// Divide two tensors element-wise
    #[inline]
    pub fn div_ref(&self, rhs: &Tensor<impl TensorBacking<R, Elem = E> + ResolveTensor<R, Elem = E>, R>) -> Tensor<ConcreteTensor<E, R>, R>
    where
        E: Default + StdDiv<Output = E>,
        DivOp: SimdBinaryOp<E>,
    {
        Tensor::new(binary_tensor_op_ref::<E, R, DivOp>(&self.inner.to_concrete(), &rhs.inner.to_concrete()))
    }

    /// Negate tensor element-wise
    #[inline]
    pub fn neg_ref(&self) -> Tensor<ConcreteTensor<E, R>, R>
    where
        E: Default + StdNeg<Output = E>,
        NegOp: SimdUnaryOp<E>,
    {
        Tensor::new(unary_tensor_op_ref::<E, R, NegOp>(&self.inner.to_concrete()))
    }

    /// Absolute value element-wise
    #[inline]
    pub fn abs_ref(&self) -> Tensor<ConcreteTensor<E, R>, R>
    where
        E: Default,
        AbsOp: SimdUnaryOp<E>,
    {
        Tensor::new(unary_tensor_op_ref::<E, R, AbsOp>(&self.inner.to_concrete()))
    }

    /// Square root element-wise
    #[inline]
    pub fn sqrt_ref(&self) -> Tensor<ConcreteTensor<E, R>, R>
    where
        E: Default,
        SqrtOp: SimdUnaryOp<E>,
    {
        Tensor::new(unary_tensor_op_ref::<E, R, SqrtOp>(&self.inner.to_concrete()))
    }

    /// Exponential (e^x) element-wise
    #[inline]
    pub fn exp_ref(&self) -> Tensor<ConcreteTensor<E, R>, R>
    where
        E: Default,
        ExpOp: SimdUnaryOp<E>,
    {
        Tensor::new(unary_tensor_op_ref::<E, R, ExpOp>(&self.inner.to_concrete()))
    }

    /// Natural logarithm element-wise
    #[inline]
    pub fn log_ref(&self) -> Tensor<ConcreteTensor<E, R>, R>
    where
        E: Default,
        LogOp: SimdUnaryOp<E>,
    {
        Tensor::new(unary_tensor_op_ref::<E, R, LogOp>(&self.inner.to_concrete()))
    }

    /// Sine element-wise
    #[inline]
    pub fn sin_ref(&self) -> Tensor<ConcreteTensor<E, R>, R>
    where
        E: Default,
        SinOp: SimdUnaryOp<E>,
    {
        Tensor::new(unary_tensor_op_ref::<E, R, SinOp>(&self.inner.to_concrete()))
    }

    /// Cosine element-wise
    #[inline]
    pub fn cos_ref(&self) -> Tensor<ConcreteTensor<E, R>, R>
    where
        E: Default,
        CosOp: SimdUnaryOp<E>,
    {
        Tensor::new(unary_tensor_op_ref::<E, R, CosOp>(&self.inner.to_concrete()))
    }

    /// Hyperbolic tangent element-wise
    #[inline]
    pub fn tanh_ref(&self) -> Tensor<ConcreteTensor<E, R>, R>
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
    pub fn eq_ref(&self, rhs: &Tensor<impl TensorBacking<R, Elem = E> + ResolveTensor<R, Elem = E>, R>) -> Tensor<ConcreteTensor<E, R>, R>
    where
        E: Default,
        EqOp: SimdComparisonOp<E>,
    {
        Tensor::new(comparison_tensor_op_ref::<E, R, EqOp>(&self.inner.to_concrete(), &rhs.inner.to_concrete()))
    }

    /// Element-wise less than comparison
    #[inline]
    pub fn lt_ref(&self, rhs: &Tensor<impl TensorBacking<R, Elem = E> + ResolveTensor<R, Elem = E>, R>) -> Tensor<ConcreteTensor<E, R>, R>
    where
        E: Default,
        LtOp: SimdComparisonOp<E>,
    {
        Tensor::new(comparison_tensor_op_ref::<E, R, LtOp>(&self.inner.to_concrete(), &rhs.inner.to_concrete()))
    }

    /// Element-wise greater than comparison
    #[inline]
    pub fn gt_ref(&self, rhs: &Tensor<impl TensorBacking<R, Elem = E> + ResolveTensor<R, Elem = E>, R>) -> Tensor<ConcreteTensor<E, R>, R>
    where
        E: Default,
        GtOp: SimdComparisonOp<E>,
    {
        Tensor::new(comparison_tensor_op_ref::<E, R, GtOp>(&self.inner.to_concrete(), &rhs.inner.to_concrete()))
    }

    /// Conditional selection: where self != 0, select on_true, else on_false
    #[inline]
    pub fn where_cond(
        &self,
        on_true: &Tensor<impl TensorBacking<R, Elem = E> + ResolveTensor<R, Elem = E>, R>,
        on_false: &Tensor<impl TensorBacking<R, Elem = E> + ResolveTensor<R, Elem = E>, R>,
    ) -> Tensor<ConcreteTensor<E, R>, R>
    where
        E: Default + IsNonZero,
    {
        Tensor::new(where_cond_ref(&self.inner.to_concrete(), &on_true.inner.to_concrete(), &on_false.inner.to_concrete()))
    }

    /// Cast tensor to another element type
    #[inline]
    pub fn cast<E2>(&self) -> Tensor<ConcreteTensor<E2, R>, R>
    where
        E: CastTo<E2>,
        E2: SimdElement,
    {
        Tensor::new(cast_tensor(&self.inner.to_concrete()))
    }

    /// Select elements along a dimension using indices
    #[inline]
    pub fn index_select(&self, dimension: usize, indices: &Tensor<ConcreteTensor<u32, 1>, 1>) -> Tensor<ConcreteTensor<E, R>, R> {
        Tensor::new(index_select_ref(&self.inner.to_concrete(), dimension, &indices.inner))
    }

    /// Sum along a specific axis, reducing the tensor rank by 1
    #[inline]
    pub fn sum_axis<const OUT_RANK: usize, const AXIS: usize>(&self) -> Tensor<ConcreteTensor<E, OUT_RANK>, OUT_RANK>
    where
        E: Default,
        ConcreteTensor<E, R>: LastRank<OUT_RANK, E>,
        SumOp: SimdReduceOp<E>,
    {
        Tensor::new(reduce_tensor_axis::<E, R, OUT_RANK, AXIS, SumOp>(&self.inner.to_concrete()))
    }

    /// Maximum along a specific axis, reducing the tensor rank by 1
    #[inline]
    pub fn max_axis<const OUT_RANK: usize, const AXIS: usize>(&self) -> Tensor<ConcreteTensor<E, OUT_RANK>, OUT_RANK>
    where
        E: Default,
        ConcreteTensor<E, R>: LastRank<OUT_RANK, E>,
        MaxOp: SimdReduceOp<E>,
    {
        Tensor::new(reduce_tensor_axis::<E, R, OUT_RANK, AXIS, MaxOp>(&self.inner.to_concrete()))
    }

    /// Minimum along a specific axis, reducing the tensor rank by 1
    #[inline]
    pub fn min_axis<const OUT_RANK: usize, const AXIS: usize>(&self) -> Tensor<ConcreteTensor<E, OUT_RANK>, OUT_RANK>
    where
        E: Default,
        ConcreteTensor<E, R>: LastRank<OUT_RANK, E>,
        MinOp: SimdReduceOp<E>,
    {
        Tensor::new(reduce_tensor_axis::<E, R, OUT_RANK, AXIS, MinOp>(&self.inner.to_concrete()))
    }
}

impl<T: TensorBacking<R>, const R: usize> TensorBacking<R> for Tensor<T, R> {
    type Elem = T::Elem;
}

// Implement std::ops traits for Tensor wrapper to enable operator syntax

impl<T1, T2, const R: usize> StdAdd<Tensor<T2, R>> for Tensor<T1, R>
where
    T1: TensorBacking<R>,
    T2: TensorBacking<R, Elem = T1::Elem>,
    T1::Elem: SimdElement + StdAdd<Output = T1::Elem> + Default,
    AddOp: SimdBinaryOp<T1::Elem>,
{
    type Output = Tensor<pairwise::Add<T1::Elem, R, T1, T2>, R>;

    fn add(self, rhs: Tensor<T2, R>) -> Self::Output {
        Tensor::new(pairwise::Add::new(self.inner, rhs.inner))
    }
}

impl<T1, T2, const R: usize> StdSub<Tensor<T2, R>> for Tensor<T1, R>
where
    T1: TensorBacking<R>,
    T2: TensorBacking<R, Elem = T1::Elem>,
    T1::Elem: SimdElement + StdSub<Output = T1::Elem> + Default,
    SubOp: SimdBinaryOp<T1::Elem>,
{
    type Output = Tensor<pairwise::Sub<T1::Elem, R, T1, T2>, R>;

    fn sub(self, rhs: Tensor<T2, R>) -> Self::Output {
        Tensor::new(pairwise::Sub::new(self.inner, rhs.inner))
    }
}

impl<T1, T2, const R: usize> StdMul<Tensor<T2, R>> for Tensor<T1, R>
where
    T1: TensorBacking<R>,
    T2: TensorBacking<R, Elem = T1::Elem>,
    T1::Elem: SimdElement + StdMul<Output = T1::Elem> + Default,
    MulOp: SimdBinaryOp<T1::Elem>,
{
    type Output = Tensor<pairwise::Mul<T1::Elem, R, T1, T2>, R>;

    fn mul(self, rhs: Tensor<T2, R>) -> Self::Output {
        Tensor::new(pairwise::Mul::new(self.inner, rhs.inner))
    }
}

impl<T1, T2, const R: usize> StdDiv<Tensor<T2, R>> for Tensor<T1, R>
where
    T1: TensorBacking<R>,
    T2: TensorBacking<R, Elem = T1::Elem>,
    T1::Elem: SimdElement + StdDiv<Output = T1::Elem> + Default,
    DivOp: SimdBinaryOp<T1::Elem>,
{
    type Output = Tensor<pairwise::Div<T1::Elem, R, T1, T2>, R>;

    fn div(self, rhs: Tensor<T2, R>) -> Self::Output {
        Tensor::new(pairwise::Div::new(self.inner, rhs.inner))
    }
}

impl<T, const R: usize> StdNeg for Tensor<T, R>
where
    T: TensorBacking<R>,
    T::Elem: SimdElement + StdNeg<Output = T::Elem> + Default,
    NegOp: SimdUnaryOp<T::Elem>,
{
    type Output = Tensor<elementwise::Neg<T::Elem, R, T>, R>;

    fn neg(self) -> Self::Output {
        Tensor::new(elementwise::Neg::new(self.inner))
    }
}

// Implement Expr for Tensor to enable evaluation
impl<E, T, const R: usize> Expr for Tensor<T, R>
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
impl<E, T, const R: usize> ResolveTensor<R> for Tensor<T, R>
where
    E: SimdElement,
    T: TensorBacking<R, Elem = E> + ResolveTensor<R, Elem = E>,
{
    fn to_concrete(&self) -> ConcreteTensor<E, R> {
        self.inner.to_concrete()
    }
}

/// Helper to iterate over indices of a tensor with given shape
pub(crate) struct IndexIterator {
    shape: Box<[usize]>,
    indices: Vec<usize>,
    done: bool,
}

impl IndexIterator {
    pub(crate) fn new(shape: &[usize]) -> Self {
        let done = shape.iter().any(|&s| s == 0);
        Self {
            shape: shape.into(),
            indices: vec![0; shape.len()],
            done,
        }
    }
}

impl Iterator for IndexIterator {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let result = self.indices.clone();

        // Increment indices (row-major order)
        for i in (0..self.shape.len()).rev() {
            self.indices[i] += 1;
            if self.indices[i] < self.shape[i] {
                break;
            }
            self.indices[i] = 0;
            if i == 0 {
                self.done = true;
            }
        }

        Some(result)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Layout {
    pub(crate) offset: usize,
    pub(crate) shape: Box<[usize]>,
    pub(crate) strides: Box<[usize]>,
    /// Cached contiguity flag - computed once at construction
    is_contiguous: bool,
}

impl Layout {
    /// Create a new layout with explicit offset, shape, and strides.
    /// Contiguity is computed automatically based on offset and strides.
    #[allow(dead_code)] // Available for view/slice operations
    pub(crate) fn new(offset: usize, shape: Box<[usize]>, strides: Box<[usize]>) -> Self {
        let is_contiguous = offset == 0 && strides == Self::contiguous_strides(&shape);
        Self {
            offset,
            shape,
            strides,
            is_contiguous,
        }
    }

    /// Create a contiguous layout for the given shape.
    pub(crate) fn contiguous(shape: &[usize]) -> Self {
        let strides = Self::contiguous_strides(shape);
        Self {
            offset: 0,
            shape: shape.into(),
            strides,
            is_contiguous: true, // Contiguous layout is always contiguous
        }
    }

    fn contiguous_strides(shape: &[usize]) -> Box<[usize]> {
        let mut acc = 1;
        let mut strides = vec![0; shape.len()].into_boxed_slice();
        for i in (0..shape.len()).rev() {
            strides[i] = acc;
            acc *= shape[i];
        }
        strides
    }

    /// Check if the tensor has contiguous memory layout (cached, O(1))
    #[inline(always)]
    pub(crate) fn is_contiguous(&self) -> bool {
        self.is_contiguous
    }

    pub(crate) fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Calculate the linear index for a given set of logical indices
    pub(crate) fn linear_index(&self, indices: &[usize]) -> usize {
        debug_assert_eq!(indices.len(), self.shape.len());
        self.offset
            + indices
                .iter()
                .zip(self.strides.iter())
                .map(|(idx, stride)| idx * stride)
                .sum::<usize>()
    }
}

#[derive(Clone)]
pub struct ConcreteTensor<T: SimdElement, const R: usize> {
    layout: Layout,
    backing: ABox<[T]>,
}

impl<T, const R: usize> TensorBacking<R> for ConcreteTensor<T, R>
where
    T: SimdElement,
{
    type Elem = T;
}

impl<T, const R: usize> ResolveTensor<R> for ConcreteTensor<T, R>
where
    T: SimdElement,
{
    fn to_concrete(&self) -> ConcreteTensor<T, R> {
        self.clone()
    }
}

impl<T, const R: usize> ResolvedTensor<R> for ConcreteTensor<T, R>
where
    T: SimdElement,
{
    fn shape(&self) -> &[usize] {
        self.layout.shape.as_ref()
    }
    fn strides(&self) -> &[usize] {
        self.layout.strides.as_ref()
    }
    fn offset(&self) -> usize {
        self.layout.offset
    }
    fn data(&self) -> &ABox<[Self::Elem]> {
        &self.backing
    }
    fn data_mut(&mut self) -> &mut ABox<[Self::Elem]> {
        &mut self.backing
    }
}

impl<T: SimdElement, const R: usize> Expr for ConcreteTensor<T, R> {
    type Elem = T;

    #[inline(always)]
    fn eval_scalar(&self, idx: usize) -> T {
        if self.layout.is_contiguous() {
            self.backing[idx]
        } else {
            // Convert linear index to logical indices for strided access
            let indices = expr::linear_to_indices::<R>(idx, &self.layout.shape);
            let phys_idx = self.layout.linear_index(&indices);
            self.backing[phys_idx]
        }
    }

    #[inline(always)]
    fn eval_simd<S: Simd>(&self, _simd: S, base_idx: usize) -> T::Simd<S> {
        if self.layout.is_contiguous() {
            // Fast path: direct SIMD load from contiguous, aligned data
            // - base_idx is always aligned to SIMD width (caller guarantees this)
            // - backing is allocated with 64-byte alignment via aligned-vec
            // - base_idx + lane_count <= len (caller guarantees this)
            let (simd_slice, _) = T::as_simd::<S>(&self.backing[base_idx..]);
            simd_slice[0]
        } else {
            // Slow path: gather elements one by one
            // For strided tensors, we fall back to scalar evaluation
            // and construct a SIMD vector manually
            let lane_count = std::mem::size_of::<T::Simd<S>>() / std::mem::size_of::<T>();
            let mut temp = [T::default(); MAX_SIMD_LANES];
            for i in 0..lane_count {
                temp[i] = self.eval_scalar(base_idx + i);
            }
            let (simd_vec, _) = T::as_simd::<S>(&temp[..lane_count]);
            simd_vec[0]
        }
    }

    fn len(&self) -> usize {
        self.layout.num_elements()
    }

    fn shape(&self) -> &[usize] {
        &self.layout.shape
    }

    fn is_contiguous(&self) -> bool {
        self.layout.is_contiguous()
    }
}

// Implement TensorBacking for references so they can be used in expression trees
impl<T: SimdElement, const R: usize> TensorBacking<R> for &ConcreteTensor<T, R> {
    type Elem = T;
}

// Implement ResolveTensor for references
impl<T: SimdElement, const R: usize> ResolveTensor<R> for &ConcreteTensor<T, R> {
    fn to_concrete(&self) -> ConcreteTensor<T, R> {
        (*self).clone()
    }
}

impl<T: SimdElement, const R: usize> ConcreteTensor<T, R> {
    /// Create a new tensor with contiguous layout from shape, filled with zeros
    pub fn zeros(shape: [usize; R]) -> Self
    where
        T: Default,
    {
        let layout = Layout::contiguous(&shape);
        let num_elements = layout.num_elements();
        let mut vec: AVec<T> = AVec::with_capacity(64, num_elements);
        vec.resize(num_elements, T::default());
        let backing = vec.into_boxed_slice();
        Self { layout, backing }
    }

    /// Create a new tensor with uninitialized memory (for internal use only)
    #[inline]
    pub(crate) fn uninit_unchecked(shape: [usize; R]) -> Self {
        let layout = Layout::contiguous(&shape);
        let num_elements = layout.num_elements();
        // Transmute the MaybeUninit vec to T vec - this is safe because we will
        // write to all elements before reading, and MaybeUninit<T> has same layout as T
        // SAFETY: MaybeUninit<T> has same layout as T, and T is Pod so any memory layout is valid
        let backing: ABox<[T]> = unsafe {
            // Allocate the aligned pointer
            let mut vec: AVec<MaybeUninit<T>> = AVec::with_capacity(64, num_elements);
            vec.set_len(num_elements);
            std::mem::transmute::<ABox<[MaybeUninit<T>]>, ABox<[T]>>(vec.into_boxed_slice())
        };
        Self { layout, backing }
    }

    /// Create a new tensor from existing data with contiguous layout
    pub fn from_slice(shape: [usize; R], data: &[T]) -> Self {
        let layout = Layout::contiguous(&shape);
        assert_eq!(layout.num_elements(), data.len());
        let mut vec: AVec<T> = AVec::with_capacity(64, data.len());
        vec.extend_from_slice(data);
        let backing = vec.into_boxed_slice();
        Self { layout, backing }
    }

    /// Get a reference to the layout
    pub(crate) fn layout(&self) -> &Layout {
        &self.layout
    }

    /// Calculate the linear index for given logical indices
    fn linear_index(&self, indices: &[usize; R]) -> usize {
        self.layout.linear_index(indices)
    }

    /// Get element at logical indices
    pub fn get(&self, indices: [usize; R]) -> T {
        let idx = self.linear_index(&indices);
        self.backing[idx]
    }

    /// Set element at logical indices
    pub fn set(&mut self, indices: [usize; R], value: T) {
        let idx = self.linear_index(&indices);
        self.backing[idx] = value;
    }

    /// Add two tensors element-wise (reference-based, no cloning)
    #[inline]
    pub fn add_ref(&self, rhs: &Self) -> Self
    where
        T: Default + StdAdd<Output = T>,
        AddOp: SimdBinaryOp<T>,
    {
        binary_tensor_op_ref::<T, R, AddOp>(self, rhs)
    }

    /// Subtract two tensors element-wise (reference-based, no cloning)
    #[inline]
    pub fn sub_ref(&self, rhs: &Self) -> Self
    where
        T: Default + StdSub<Output = T>,
        SubOp: SimdBinaryOp<T>,
    {
        binary_tensor_op_ref::<T, R, SubOp>(self, rhs)
    }

    /// Multiply two tensors element-wise (reference-based, no cloning)
    #[inline]
    pub fn mul_ref(&self, rhs: &Self) -> Self
    where
        T: Default + StdMul<Output = T>,
        MulOp: SimdBinaryOp<T>,
    {
        binary_tensor_op_ref::<T, R, MulOp>(self, rhs)
    }

    /// Divide two tensors element-wise (reference-based, no cloning)
    #[inline]
    pub fn div_ref(&self, rhs: &Self) -> Self
    where
        T: Default + StdDiv<Output = T>,
        DivOp: SimdBinaryOp<T>,
    {
        binary_tensor_op_ref::<T, R, DivOp>(self, rhs)
    }

    /// Negate tensor element-wise (reference-based, no cloning)
    #[inline]
    pub fn neg_ref(&self) -> Self
    where
        T: Default + StdNeg<Output = T>,
        NegOp: SimdUnaryOp<T>,
    {
        unary_tensor_op_ref::<T, R, NegOp>(self)
    }

    /// Absolute value element-wise (reference-based, no cloning)
    #[inline]
    pub fn abs_ref(&self) -> Self
    where
        T: Default,
        AbsOp: SimdUnaryOp<T>,
    {
        unary_tensor_op_ref::<T, R, AbsOp>(self)
    }

    /// Square root element-wise (reference-based, no cloning)
    #[inline]
    pub fn sqrt_ref(&self) -> Self
    where
        T: Default,
        SqrtOp: SimdUnaryOp<T>,
    {
        unary_tensor_op_ref::<T, R, SqrtOp>(self)
    }

    /// Exponential (e^x) element-wise (reference-based, no cloning)
    #[inline]
    pub fn exp_ref(&self) -> Self
    where
        T: Default,
        ExpOp: SimdUnaryOp<T>,
    {
        unary_tensor_op_ref::<T, R, ExpOp>(self)
    }

    /// Base-2 exponential (2^x) element-wise (reference-based, no cloning)
    #[inline]
    pub fn exp2_ref(&self) -> Self
    where
        T: Default,
        Exp2Op: SimdUnaryOp<T>,
    {
        unary_tensor_op_ref::<T, R, Exp2Op>(self)
    }

    /// Natural logarithm (ln) element-wise (reference-based, no cloning)
    #[inline]
    pub fn log_ref(&self) -> Self
    where
        T: Default,
        LogOp: SimdUnaryOp<T>,
    {
        unary_tensor_op_ref::<T, R, LogOp>(self)
    }

    /// Base-2 logarithm element-wise (reference-based, no cloning)
    #[inline]
    pub fn log2_ref(&self) -> Self
    where
        T: Default,
        Log2Op: SimdUnaryOp<T>,
    {
        unary_tensor_op_ref::<T, R, Log2Op>(self)
    }

    /// Sine element-wise (reference-based, no cloning)
    #[inline]
    pub fn sin_ref(&self) -> Self
    where
        T: Default,
        SinOp: SimdUnaryOp<T>,
    {
        unary_tensor_op_ref::<T, R, SinOp>(self)
    }

    /// Cosine element-wise (reference-based, no cloning)
    #[inline]
    pub fn cos_ref(&self) -> Self
    where
        T: Default,
        CosOp: SimdUnaryOp<T>,
    {
        unary_tensor_op_ref::<T, R, CosOp>(self)
    }

    /// Tangent element-wise (reference-based, no cloning)
    #[inline]
    pub fn tan_ref(&self) -> Self
    where
        T: Default,
        TanOp: SimdUnaryOp<T>,
    {
        unary_tensor_op_ref::<T, R, TanOp>(self)
    }

    /// Hyperbolic tangent element-wise (reference-based, no cloning)
    #[inline]
    pub fn tanh_ref(&self) -> Self
    where
        T: Default,
        TanhOp: SimdUnaryOp<T>,
    {
        unary_tensor_op_ref::<T, R, TanhOp>(self)
    }

    /// Element-wise equality comparison (tensor vs tensor)
    /// Returns 1.0/1 where equal, 0.0/0 where not equal
    #[inline]
    pub fn eq_ref(&self, rhs: &Self) -> Self
    where
        T: Default,
        EqOp: SimdComparisonOp<T>,
    {
        comparison_tensor_op_ref::<T, R, EqOp>(self, rhs)
    }

    /// Element-wise inequality comparison (tensor vs tensor)
    #[inline]
    pub fn ne_ref(&self, rhs: &Self) -> Self
    where
        T: Default,
        NeOp: SimdComparisonOp<T>,
    {
        comparison_tensor_op_ref::<T, R, NeOp>(self, rhs)
    }

    /// Element-wise less than comparison (tensor vs tensor)
    #[inline]
    pub fn lt_ref(&self, rhs: &Self) -> Self
    where
        T: Default,
        LtOp: SimdComparisonOp<T>,
    {
        comparison_tensor_op_ref::<T, R, LtOp>(self, rhs)
    }

    /// Element-wise less than or equal comparison (tensor vs tensor)
    #[inline]
    pub fn lte_ref(&self, rhs: &Self) -> Self
    where
        T: Default,
        LteOp: SimdComparisonOp<T>,
    {
        comparison_tensor_op_ref::<T, R, LteOp>(self, rhs)
    }

    /// Element-wise greater than comparison (tensor vs tensor)
    #[inline]
    pub fn gt_ref(&self, rhs: &Self) -> Self
    where
        T: Default,
        GtOp: SimdComparisonOp<T>,
    {
        comparison_tensor_op_ref::<T, R, GtOp>(self, rhs)
    }

    /// Element-wise greater than or equal comparison (tensor vs tensor)
    #[inline]
    pub fn gte_ref(&self, rhs: &Self) -> Self
    where
        T: Default,
        GteOp: SimdComparisonOp<T>,
    {
        comparison_tensor_op_ref::<T, R, GteOp>(self, rhs)
    }

    /// Element-wise equality comparison (tensor vs scalar)
    #[inline]
    pub fn eq_scalar(&self, scalar: T) -> Self
    where
        T: Default,
        EqOp: SimdComparisonOp<T>,
    {
        comparison_scalar_op_ref::<T, R, EqOp>(self, scalar)
    }

    /// Element-wise inequality comparison (tensor vs scalar)
    #[inline]
    pub fn ne_scalar(&self, scalar: T) -> Self
    where
        T: Default,
        NeOp: SimdComparisonOp<T>,
    {
        comparison_scalar_op_ref::<T, R, NeOp>(self, scalar)
    }

    /// Element-wise less than comparison (tensor vs scalar)
    #[inline]
    pub fn lt_scalar(&self, scalar: T) -> Self
    where
        T: Default,
        LtOp: SimdComparisonOp<T>,
    {
        comparison_scalar_op_ref::<T, R, LtOp>(self, scalar)
    }

    /// Element-wise less than or equal comparison (tensor vs scalar)
    #[inline]
    pub fn lte_scalar(&self, scalar: T) -> Self
    where
        T: Default,
        LteOp: SimdComparisonOp<T>,
    {
        comparison_scalar_op_ref::<T, R, LteOp>(self, scalar)
    }

    /// Element-wise greater than comparison (tensor vs scalar)
    #[inline]
    pub fn gt_scalar(&self, scalar: T) -> Self
    where
        T: Default,
        GtOp: SimdComparisonOp<T>,
    {
        comparison_scalar_op_ref::<T, R, GtOp>(self, scalar)
    }

    /// Element-wise greater than or equal comparison (tensor vs scalar)
    #[inline]
    pub fn gte_scalar(&self, scalar: T) -> Self
    where
        T: Default,
        GteOp: SimdComparisonOp<T>,
    {
        comparison_scalar_op_ref::<T, R, GteOp>(self, scalar)
    }

    /// Conditional selection: where self != 0, select on_true, else on_false
    #[inline]
    pub fn where_cond(&self, on_true: &Self, on_false: &Self) -> Self
    where
        T: Default + IsNonZero,
    {
        where_cond_ref(self, on_true, on_false)
    }

    /// Cast tensor to another element type
    #[inline]
    pub fn cast<T2>(&self) -> ConcreteTensor<T2, R>
    where
        T: CastTo<T2>,
        T2: SimdElement,
    {
        cast_tensor(self)
    }

    /// Select elements along a dimension using indices
    ///
    /// For a 2D tensor with shape [M, N] and indices [I]:
    /// - index_select(0, indices) -> shape [I, N], selecting rows
    /// - index_select(1, indices) -> shape [M, I], selecting columns
    #[inline]
    pub fn index_select(&self, dimension: usize, indices: &ConcreteTensor<u32, 1>) -> Self {
        index_select_ref(self, dimension, indices)
    }

    /// Sum all elements in the tensor
    #[inline]
    pub fn sum(&self) -> T
    where
        SumOp: SimdReduceOp<T>,
    {
        reduce_tensor_op::<T, R, SumOp>(self)
    }

    /// Find the maximum element in the tensor
    #[inline]
    pub fn max(&self) -> T
    where
        MaxOp: SimdReduceOp<T>,
    {
        reduce_tensor_op::<T, R, MaxOp>(self)
    }

    /// Find the minimum element in the tensor
    #[inline]
    pub fn min(&self) -> T
    where
        MinOp: SimdReduceOp<T>,
    {
        reduce_tensor_op::<T, R, MinOp>(self)
    }

    /// Multiply all elements in the tensor
    #[inline]
    pub fn prod(&self) -> T
    where
        ProdOp: SimdReduceOp<T>,
    {
        reduce_tensor_op::<T, R, ProdOp>(self)
    }

    /// Sum along a specific axis, reducing the tensor rank by 1
    #[inline]
    pub fn sum_axis<const OUT_RANK: usize, const AXIS: usize>(&self) -> ConcreteTensor<T, OUT_RANK>
    where
        T: Default,
        Self: LastRank<OUT_RANK, T>,
        SumOp: SimdReduceOp<T>,
    {
        reduce_tensor_axis::<T, R, OUT_RANK, AXIS, SumOp>(self)
    }

    /// Maximum along a specific axis, reducing the tensor rank by 1
    #[inline]
    pub fn max_axis<const OUT_RANK: usize, const AXIS: usize>(&self) -> ConcreteTensor<T, OUT_RANK>
    where
        T: Default,
        Self: LastRank<OUT_RANK, T>,
        MaxOp: SimdReduceOp<T>,
    {
        reduce_tensor_axis::<T, R, OUT_RANK, AXIS, MaxOp>(self)
    }

    /// Minimum along a specific axis, reducing the tensor rank by 1
    #[inline]
    pub fn min_axis<const OUT_RANK: usize, const AXIS: usize>(&self) -> ConcreteTensor<T, OUT_RANK>
    where
        T: Default,
        Self: LastRank<OUT_RANK, T>,
        MinOp: SimdReduceOp<T>,
    {
        reduce_tensor_axis::<T, R, OUT_RANK, AXIS, MinOp>(self)
    }

    /// Product along a specific axis, reducing the tensor rank by 1
    #[inline]
    pub fn prod_axis<const OUT_RANK: usize, const AXIS: usize>(&self) -> ConcreteTensor<T, OUT_RANK>
    where
        T: Default,
        Self: LastRank<OUT_RANK, T>,
        ProdOp: SimdReduceOp<T>,
    {
        reduce_tensor_axis::<T, R, OUT_RANK, AXIS, ProdOp>(self)
    }
}

// Specialized impl for f32 (pow_scalar, min_scalar, max_scalar)
impl<const R: usize> ConcreteTensor<f32, R> {
    /// Raise each element to a scalar power (x^exp)
    #[inline]
    pub fn pow_scalar(&self, exp: f32) -> Self {
        let shape: [usize; R] = ResolvedTensor::shape(self)
            .try_into()
            .expect("Shape length mismatch");
        let mut output = ConcreteTensor::<f32, R>::uninit_unchecked(shape);

        for (i, &val) in self.data().iter().enumerate() {
            output.data_mut()[i] = val.powf(exp);
        }
        output
    }

    /// Element-wise maximum with a scalar: max(x, scalar)
    #[inline]
    pub fn max_scalar(&self, scalar: f32) -> Self {
        let shape: [usize; R] = ResolvedTensor::shape(self)
            .try_into()
            .expect("Shape length mismatch");
        let mut output = ConcreteTensor::<f32, R>::uninit_unchecked(shape);

        for (i, &val) in self.data().iter().enumerate() {
            output.data_mut()[i] = val.max(scalar);
        }
        output
    }

    /// Element-wise minimum with a scalar: min(x, scalar)
    #[inline]
    pub fn min_scalar(&self, scalar: f32) -> Self {
        let shape: [usize; R] = ResolvedTensor::shape(self)
            .try_into()
            .expect("Shape length mismatch");
        let mut output = ConcreteTensor::<f32, R>::uninit_unchecked(shape);

        for (i, &val) in self.data().iter().enumerate() {
            output.data_mut()[i] = val.min(scalar);
        }
        output
    }

    /// Clamp all elements to be within [min, max]
    #[inline]
    pub fn clamp(&self, min: f32, max: f32) -> Self {
        let shape: [usize; R] = ResolvedTensor::shape(self)
            .try_into()
            .expect("Shape length mismatch");
        let mut output = ConcreteTensor::<f32, R>::uninit_unchecked(shape);

        for (i, &val) in self.data().iter().enumerate() {
            output.data_mut()[i] = val.clamp(min, max);
        }
        output
    }
}

// Specialized impl for f64 (pow_scalar, min_scalar, max_scalar)
impl<const R: usize> ConcreteTensor<f64, R> {
    /// Raise each element to a scalar power (x^exp)
    #[inline]
    pub fn pow_scalar(&self, exp: f64) -> Self {
        let shape: [usize; R] = ResolvedTensor::shape(self)
            .try_into()
            .expect("Shape length mismatch");
        let mut output = ConcreteTensor::<f64, R>::uninit_unchecked(shape);

        for (i, &val) in self.data().iter().enumerate() {
            output.data_mut()[i] = val.powf(exp);
        }
        output
    }

    /// Element-wise maximum with a scalar: max(x, scalar)
    #[inline]
    pub fn max_scalar(&self, scalar: f64) -> Self {
        let shape: [usize; R] = ResolvedTensor::shape(self)
            .try_into()
            .expect("Shape length mismatch");
        let mut output = ConcreteTensor::<f64, R>::uninit_unchecked(shape);

        for (i, &val) in self.data().iter().enumerate() {
            output.data_mut()[i] = val.max(scalar);
        }
        output
    }

    /// Element-wise minimum with a scalar: min(x, scalar)
    #[inline]
    pub fn min_scalar(&self, scalar: f64) -> Self {
        let shape: [usize; R] = ResolvedTensor::shape(self)
            .try_into()
            .expect("Shape length mismatch");
        let mut output = ConcreteTensor::<f64, R>::uninit_unchecked(shape);

        for (i, &val) in self.data().iter().enumerate() {
            output.data_mut()[i] = val.min(scalar);
        }
        output
    }

    /// Clamp all elements to be within [min, max]
    #[inline]
    pub fn clamp(&self, min: f64, max: f64) -> Self {
        let shape: [usize; R] = ResolvedTensor::shape(self)
            .try_into()
            .expect("Shape length mismatch");
        let mut output = ConcreteTensor::<f64, R>::uninit_unchecked(shape);

        for (i, &val) in self.data().iter().enumerate() {
            output.data_mut()[i] = val.clamp(min, max);
        }
        output
    }
}

/// Trait for SIMD element types with associated SIMD vector type
pub trait SimdElement: Sized + Copy + Default + Pod {
    /// The SIMD vector type for this element (GAT)
    type Simd<S: Simd>: Copy;

    /// Convert slice to SIMD vectors + remainder
    fn as_simd<S: Simd>(slice: &[Self]) -> (&[Self::Simd<S>], &[Self]);
    fn as_mut_simd<S: Simd>(slice: &mut [Self]) -> (&mut [Self::Simd<S>], &mut [Self]);

    /// Broadcast a scalar value to all lanes of a SIMD vector
    fn splat<S: Simd>(simd: S, value: Self) -> Self::Simd<S>;
}

macro_rules! impl_simd_element {
    ($elem:ty, $simd_ty:ident, $as_simd:ident, $as_mut_simd:ident, $splat:ident) => {
        impl SimdElement for $elem {
            type Simd<S: Simd> = S::$simd_ty;

            #[inline(always)]
            fn as_simd<S: Simd>(slice: &[Self]) -> (&[S::$simd_ty], &[Self]) {
                S::$as_simd(slice)
            }

            #[inline(always)]
            fn as_mut_simd<S: Simd>(slice: &mut [Self]) -> (&mut [S::$simd_ty], &mut [Self]) {
                S::$as_mut_simd(slice)
            }

            #[inline(always)]
            fn splat<S: Simd>(simd: S, value: Self) -> S::$simd_ty {
                simd.$splat(value)
            }
        }
    };
}

impl_simd_element!(f32, f32s, as_simd_f32s, as_mut_simd_f32s, splat_f32s);
impl_simd_element!(f64, f64s, as_simd_f64s, as_mut_simd_f64s, splat_f64s);
impl_simd_element!(i8, i8s, as_simd_i8s, as_mut_simd_i8s, splat_i8s);
impl_simd_element!(i16, i16s, as_simd_i16s, as_mut_simd_i16s, splat_i16s);
impl_simd_element!(i32, i32s, as_simd_i32s, as_mut_simd_i32s, splat_i32s);
impl_simd_element!(i64, i64s, as_simd_i64s, as_mut_simd_i64s, splat_i64s);
impl_simd_element!(u8, u8s, as_simd_u8s, as_mut_simd_u8s, splat_u8s);
impl_simd_element!(u16, u16s, as_simd_u16s, as_mut_simd_u16s, splat_u16s);
impl_simd_element!(u32, u32s, as_simd_u32s, as_mut_simd_u32s, splat_u32s);
impl_simd_element!(u64, u64s, as_simd_u64s, as_mut_simd_u64s, splat_u64s);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_add_operator() {
        // Use Tensor::from_slice directly - cleaner API
        let a: Tensor<ConcreteTensor<f32, 1>, 1> = Tensor::from_slice([4], &[1.0, 2.0, 3.0, 4.0]);
        let b: Tensor<ConcreteTensor<f32, 1>, 1> = Tensor::from_slice([4], &[10.0, 20.0, 30.0, 40.0]);

        // Use + operator and eval() to get result
        let result = (a + b).eval();

        assert_eq!(result.get([0]), 11.0);
        assert_eq!(result.get([1]), 22.0);
        assert_eq!(result.get([2]), 33.0);
        assert_eq!(result.get([3]), 44.0);
    }

    #[test]
    fn test_tensor_sub_operator() {
        let a: Tensor<ConcreteTensor<f32, 1>, 1> = Tensor::from_slice([4], &[10.0, 20.0, 30.0, 40.0]);
        let b: Tensor<ConcreteTensor<f32, 1>, 1> = Tensor::from_slice([4], &[1.0, 2.0, 3.0, 4.0]);

        let result = (a - b).eval();

        assert_eq!(result.get([0]), 9.0);
        assert_eq!(result.get([1]), 18.0);
        assert_eq!(result.get([2]), 27.0);
        assert_eq!(result.get([3]), 36.0);
    }

    #[test]
    fn test_tensor_mul_operator() {
        let a: Tensor<ConcreteTensor<f32, 1>, 1> = Tensor::from_slice([4], &[1.0, 2.0, 3.0, 4.0]);
        let b: Tensor<ConcreteTensor<f32, 1>, 1> = Tensor::from_slice([4], &[2.0, 3.0, 4.0, 5.0]);

        let result = (a * b).eval();

        assert_eq!(result.get([0]), 2.0);
        assert_eq!(result.get([1]), 6.0);
        assert_eq!(result.get([2]), 12.0);
        assert_eq!(result.get([3]), 20.0);
    }

    #[test]
    fn test_tensor_div_operator() {
        let a: Tensor<ConcreteTensor<f32, 1>, 1> = Tensor::from_slice([4], &[10.0, 20.0, 30.0, 40.0]);
        let b: Tensor<ConcreteTensor<f32, 1>, 1> = Tensor::from_slice([4], &[2.0, 4.0, 5.0, 8.0]);

        let result = (a / b).eval();

        assert_eq!(result.get([0]), 5.0);
        assert_eq!(result.get([1]), 5.0);
        assert_eq!(result.get([2]), 6.0);
        assert_eq!(result.get([3]), 5.0);
    }

    #[test]
    fn test_tensor_neg_operator() {
        let a: Tensor<ConcreteTensor<f32, 1>, 1> = Tensor::from_slice([4], &[1.0, -2.0, 3.0, -4.0]);

        let result = (-a).eval();

        assert_eq!(result.get([0]), -1.0);
        assert_eq!(result.get([1]), 2.0);
        assert_eq!(result.get([2]), -3.0);
        assert_eq!(result.get([3]), 4.0);
    }

    #[test]
    fn test_tensor_chained_operators() {
        // Test (a + b) * c with lazy evaluation
        let a: Tensor<ConcreteTensor<f32, 1>, 1> = Tensor::from_slice([4], &[1.0, 2.0, 3.0, 4.0]);
        let b: Tensor<ConcreteTensor<f32, 1>, 1> = Tensor::from_slice([4], &[1.0, 1.0, 1.0, 1.0]);
        let c: Tensor<ConcreteTensor<f32, 1>, 1> = Tensor::from_slice([4], &[2.0, 2.0, 2.0, 2.0]);

        let result = ((a + b) * c).eval();

        // (1+1)*2=4, (2+1)*2=6, (3+1)*2=8, (4+1)*2=10
        assert_eq!(result.get([0]), 4.0);
        assert_eq!(result.get([1]), 6.0);
        assert_eq!(result.get([2]), 8.0);
        assert_eq!(result.get([3]), 10.0);
    }

    #[test]
    fn test_tensor_2d_operators() {
        let a: Tensor<ConcreteTensor<f32, 2>, 2> = Tensor::from_slice([2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b: Tensor<ConcreteTensor<f32, 2>, 2> = Tensor::from_slice([2, 3], &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);

        let result = (a + b).eval();

        assert_eq!(result.get([0, 0]), 11.0);
        assert_eq!(result.get([0, 2]), 33.0);
        assert_eq!(result.get([1, 0]), 44.0);
        assert_eq!(result.get([1, 2]), 66.0);
    }

    #[test]
    fn test_tensor_methods() {
        // Test the new methods on Tensor
        let a: Tensor<ConcreteTensor<f32, 1>, 1> = Tensor::from_slice([4], &[1.0, 2.0, 3.0, 4.0]);
        let b: Tensor<ConcreteTensor<f32, 1>, 1> = Tensor::from_slice([4], &[10.0, 20.0, 30.0, 40.0]);

        // Test add_ref method
        let result = a.add_ref(&b);
        assert_eq!(result.get([0]), 11.0);

        // Test sum reduction
        let c: Tensor<ConcreteTensor<f32, 1>, 1> = Tensor::from_slice([4], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(c.sum(), 10.0);

        // Test max/min
        assert_eq!(c.max(), 4.0);
        assert_eq!(c.min(), 1.0);
    }

    #[test]
    fn test_tensor_zeros() {
        let t: Tensor<ConcreteTensor<f32, 2>, 2> = Tensor::zeros([3, 4]);
        assert_eq!(t.get([0, 0]), 0.0);
        assert_eq!(t.get([2, 3]), 0.0);
    }
}
