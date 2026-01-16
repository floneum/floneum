//! CPU tensor operations with SIMD acceleration

use std::mem::MaybeUninit;
use std::ops::{Add as StdAdd, Div as StdDiv, Mul as StdMul, Neg as StdNeg, Sub as StdSub};

use aligned_vec::{ABox, AVec};
use pulp::bytemuck::Pod;
use pulp::Simd;

// Module declarations
mod elementwise;
mod expr;
mod matmul;
mod pairwise;
mod reduce;

/// Maximum number of SIMD lanes supported for strided tensor gather operations.
/// This covers AVX-512 with 64 x i8 lanes. Current architectures don't exceed this,
/// but this constant provides a clear point for future updates if needed.
pub(crate) const MAX_SIMD_LANES: usize = 64;


// Re-export public types
pub use elementwise::{Abs, Neg, Sqrt};
pub use expr::{materialize_expr, Expr};
pub use pairwise::{Add, Div, Mul, Sub};

// Re-export operation traits and markers for public bounds
pub use elementwise::{AbsOp, NegOp, SimdUnaryOp, SqrtOp};
pub use pairwise::{AddOp, DivOp, MulOp, SimdBinaryOp, SubOp};
pub use reduce::{MaxOp, MinOp, ProdOp, SimdReduceOp, SumOp};

// Internal imports from modules
use elementwise::unary_tensor_op_ref;
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

pub trait Tensor {
    type Elem: SimdElement;
    const RANK: usize;
    type Concrete: ResolvedTensor<Elem = Self::Elem>;
    const ASSERT: () = {
        assert!(
            Self::Concrete::RANK == Self::RANK,
            "Tensor rank mismatch in ConcreteTensor"
        );
    };
}

pub trait ResolveTensor<M = ()>: Tensor {
    fn to_concrete(&self) -> Self::Concrete;
}

pub trait ResolvedTensor: Tensor {
    fn shape(&self) -> &[usize];
    fn strides(&self) -> &[usize];
    fn offset(&self) -> usize;
    fn data(&self) -> &ABox<[Self::Elem]>;
    fn data_mut(&mut self) -> &mut ABox<[Self::Elem]>;
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
pub struct ConcreteTensor<T: SimdElement, const RANK: usize> {
    layout: Layout,
    backing: ABox<[T]>,
}

impl<T, const RANK: usize> Tensor for ConcreteTensor<T, RANK>
where
    T: SimdElement,
{
    type Elem = T;
    const RANK: usize = RANK;
    type Concrete = Self;
}

impl<T, const RANK: usize> ResolveTensor for ConcreteTensor<T, RANK>
where
    T: SimdElement,
{
    fn to_concrete(&self) -> Self::Concrete {
        self.clone()
    }
}

impl<T, const RANK: usize> ResolvedTensor for ConcreteTensor<T, RANK>
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

impl<T: SimdElement, const RANK: usize> Expr for ConcreteTensor<T, RANK> {
    type Elem = T;

    #[inline(always)]
    fn eval_scalar(&self, idx: usize) -> T {
        if self.layout.is_contiguous() {
            self.backing[idx]
        } else {
            // Convert linear index to logical indices for strided access
            let indices = expr::linear_to_indices::<RANK>(idx, &self.layout.shape);
            let phys_idx = self.layout.linear_index(&indices);
            self.backing[phys_idx]
        }
    }

    #[inline(always)]
    fn eval_simd<S: Simd>(&self, _simd: S, base_idx: usize) -> T::Simd<S> {
        if self.layout.is_contiguous() {
            // Fast path: direct SIMD load from contiguous, aligned data
            // SAFETY:
            // - base_idx is always aligned to SIMD width (caller guarantees this)
            // - backing is allocated with 64-byte alignment via aligned-vec
            // - base_idx + lane_count <= len (caller guarantees this)
            unsafe {
                let ptr = self.backing.as_ptr().add(base_idx) as *const T::Simd<S>;
                ptr.read_unaligned()
            }
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

// Implement Tensor for references so they can be used in expression trees
impl<T: SimdElement, const RANK: usize> Tensor for &ConcreteTensor<T, RANK> {
    type Elem = T;
    const RANK: usize = RANK;
    type Concrete = ConcreteTensor<T, RANK>;
}

// Implement ResolveTensor for references
impl<T: SimdElement, const RANK: usize> ResolveTensor for &ConcreteTensor<T, RANK> {
    fn to_concrete(&self) -> Self::Concrete {
        (*self).clone()
    }
}

impl<T: SimdElement, const RANK: usize> ConcreteTensor<T, RANK> {
    /// Create a new tensor with contiguous layout from shape, filled with zeros
    pub fn zeros(shape: [usize; RANK]) -> Self
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
    pub(crate) fn uninit_unchecked(shape: [usize; RANK]) -> Self {
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
    pub fn from_slice(shape: [usize; RANK], data: &[T]) -> Self {
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
    fn linear_index(&self, indices: &[usize; RANK]) -> usize {
        self.layout.linear_index(indices)
    }

    /// Get element at logical indices
    pub fn get(&self, indices: [usize; RANK]) -> T {
        let idx = self.linear_index(&indices);
        self.backing[idx]
    }

    /// Set element at logical indices
    pub fn set(&mut self, indices: [usize; RANK], value: T) {
        let idx = self.linear_index(&indices);
        self.backing[idx] = value;
    }

    // ========== Binary Operations (Pairwise) ==========

    /// Add two tensors element-wise (reference-based, no cloning)
    #[inline]
    pub fn add_ref(&self, rhs: &Self) -> Self
    where
        T: Default + StdAdd<Output = T>,
        AddOp: SimdBinaryOp<T>,
    {
        binary_tensor_op_ref::<T, RANK, AddOp>(self, rhs)
    }

    /// Subtract two tensors element-wise (reference-based, no cloning)
    #[inline]
    pub fn sub_ref(&self, rhs: &Self) -> Self
    where
        T: Default + StdSub<Output = T>,
        SubOp: SimdBinaryOp<T>,
    {
        binary_tensor_op_ref::<T, RANK, SubOp>(self, rhs)
    }

    /// Multiply two tensors element-wise (reference-based, no cloning)
    #[inline]
    pub fn mul_ref(&self, rhs: &Self) -> Self
    where
        T: Default + StdMul<Output = T>,
        MulOp: SimdBinaryOp<T>,
    {
        binary_tensor_op_ref::<T, RANK, MulOp>(self, rhs)
    }

    /// Divide two tensors element-wise (reference-based, no cloning)
    #[inline]
    pub fn div_ref(&self, rhs: &Self) -> Self
    where
        T: Default + StdDiv<Output = T>,
        DivOp: SimdBinaryOp<T>,
    {
        binary_tensor_op_ref::<T, RANK, DivOp>(self, rhs)
    }

    // ========== Unary Operations (Elementwise) ==========

    /// Negate tensor element-wise (reference-based, no cloning)
    #[inline]
    pub fn neg_ref(&self) -> Self
    where
        T: Default + StdNeg<Output = T>,
        NegOp: SimdUnaryOp<T>,
    {
        unary_tensor_op_ref::<T, RANK, NegOp>(self)
    }

    /// Absolute value element-wise (reference-based, no cloning)
    #[inline]
    pub fn abs_ref(&self) -> Self
    where
        T: Default,
        AbsOp: SimdUnaryOp<T>,
    {
        unary_tensor_op_ref::<T, RANK, AbsOp>(self)
    }

    /// Square root element-wise (reference-based, no cloning)
    #[inline]
    pub fn sqrt_ref(&self) -> Self
    where
        T: Default,
        SqrtOp: SimdUnaryOp<T>,
    {
        unary_tensor_op_ref::<T, RANK, SqrtOp>(self)
    }

    // ========== Reduce Operations ==========

    /// Sum all elements in the tensor
    #[inline]
    pub fn sum(&self) -> T
    where
        SumOp: SimdReduceOp<T>,
    {
        reduce_tensor_op::<T, RANK, SumOp>(self)
    }

    /// Find the maximum element in the tensor
    #[inline]
    pub fn max(&self) -> T
    where
        MaxOp: SimdReduceOp<T>,
    {
        reduce_tensor_op::<T, RANK, MaxOp>(self)
    }

    /// Find the minimum element in the tensor
    #[inline]
    pub fn min(&self) -> T
    where
        MinOp: SimdReduceOp<T>,
    {
        reduce_tensor_op::<T, RANK, MinOp>(self)
    }

    /// Multiply all elements in the tensor
    #[inline]
    pub fn prod(&self) -> T
    where
        ProdOp: SimdReduceOp<T>,
    {
        reduce_tensor_op::<T, RANK, ProdOp>(self)
    }

    // ========== Axis Reduce Operations ==========

    /// Sum along a specific axis, reducing the tensor rank by 1
    #[inline]
    pub fn sum_axis<const OUT_RANK: usize, const AXIS: usize>(&self) -> ConcreteTensor<T, OUT_RANK>
    where
        T: Default,
        Self: LastRank<OUT_RANK, T>,
        SumOp: SimdReduceOp<T>,
    {
        reduce_tensor_axis::<T, RANK, OUT_RANK, AXIS, SumOp>(self)
    }

    /// Maximum along a specific axis, reducing the tensor rank by 1
    #[inline]
    pub fn max_axis<const OUT_RANK: usize, const AXIS: usize>(&self) -> ConcreteTensor<T, OUT_RANK>
    where
        T: Default,
        Self: LastRank<OUT_RANK, T>,
        MaxOp: SimdReduceOp<T>,
    {
        reduce_tensor_axis::<T, RANK, OUT_RANK, AXIS, MaxOp>(self)
    }

    /// Minimum along a specific axis, reducing the tensor rank by 1
    #[inline]
    pub fn min_axis<const OUT_RANK: usize, const AXIS: usize>(&self) -> ConcreteTensor<T, OUT_RANK>
    where
        T: Default,
        Self: LastRank<OUT_RANK, T>,
        MinOp: SimdReduceOp<T>,
    {
        reduce_tensor_axis::<T, RANK, OUT_RANK, AXIS, MinOp>(self)
    }

    /// Product along a specific axis, reducing the tensor rank by 1
    #[inline]
    pub fn prod_axis<const OUT_RANK: usize, const AXIS: usize>(&self) -> ConcreteTensor<T, OUT_RANK>
    where
        T: Default,
        Self: LastRank<OUT_RANK, T>,
        ProdOp: SimdReduceOp<T>,
    {
        reduce_tensor_axis::<T, RANK, OUT_RANK, AXIS, ProdOp>(self)
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
