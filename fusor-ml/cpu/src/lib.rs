use std::mem::MaybeUninit;
use std::ops::{Add as StdAdd, Div as StdDiv, Mul as StdMul, Neg as StdNeg, Sub as StdSub};

use aligned_vec::{ABox, AVec};
use generativity::Id;
use pulp::bytemuck::Pod;
use pulp::{Arch, Simd, WithSimd};

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

struct Dim<'a> {
    id: Id<'a>,
}

fn with_simd_dim<F, R>(f: F) -> R
where
    F: for<'a> FnOnce(Dim<'a>) -> R,
{
    generativity::make_guard!(id);
    f(Dim { id: id.into() })
}

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
struct IndexIterator {
    shape: Box<[usize]>,
    indices: Vec<usize>,
    done: bool,
}

impl IndexIterator {
    fn new(shape: &[usize]) -> Self {
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
struct Layout {
    offset: usize,
    shape: Box<[usize]>,
    strides: Box<[usize]>,
}

impl Layout {
    fn contiguous(shape: &[usize]) -> Self {
        let strides = Self::contiguous_strides(shape);
        Self {
            offset: 0,
            shape: shape.into(),
            strides,
        }
    }

    fn from_parts(offset: usize, shape: Box<[usize]>, strides: Box<[usize]>) -> Self {
        Self {
            offset,
            shape,
            strides,
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

    fn is_contiguous(&self) -> bool {
        self.offset == 0 && self.strides == Self::contiguous_strides(&self.shape)
    }

    fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Calculate the linear index for a given set of logical indices
    fn linear_index(&self, indices: &[usize]) -> usize {
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
    fn uninit_unchecked(shape: [usize; RANK]) -> Self {
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
    fn layout(&self) -> &Layout {
        &self.layout
    }

    /// Calculate the linear index for given logical indices
    fn linear_index(&self, indices: &[usize; RANK]) -> usize {
        self.layout.linear_index(indices)
    }

    /// Get element at logical indices
    fn get(&self, indices: [usize; RANK]) -> T {
        let idx = self.linear_index(&indices);
        self.backing[idx]
    }

    /// Set element at logical indices
    fn set(&mut self, indices: [usize; RANK], value: T) {
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

/// Matrix multiplication specific implementation for 2D tensors
impl<T: SimdElement + MatmulImpl> ConcreteTensor<T, 2> {
    /// Matrix multiplication: self (M x K) @ rhs (K x N) -> (M x N)
    /// Uses optimized gemm for f32/f64, naive fallback for other types
    #[inline]
    pub fn matmul_ref(&self, rhs: &Self) -> Self {
        let m = self.shape()[0];
        let k = self.shape()[1];
        let k2 = rhs.shape()[0];
        let n = rhs.shape()[1];

        assert_eq!(
            k, k2,
            "Matrix dimension mismatch: lhs columns ({}) != rhs rows ({})",
            k, k2
        );

        let mut output = ConcreteTensor::<T, 2>::uninit_unchecked([m, n]);

        // Both inputs should be contiguous for best performance
        let lhs_contiguous = self.layout.is_contiguous();
        let rhs_contiguous = rhs.layout.is_contiguous();

        if lhs_contiguous && rhs_contiguous {
            T::matmul_contiguous(&self.backing, &rhs.backing, &mut output.backing, m, k, n);
        } else {
            // Slow path for non-contiguous tensors
            matmul_strided(self, rhs, &mut output);
        }

        output
    }
}

/// Optimized matmul for contiguous tensors using gemm crate
#[inline(always)]
fn matmul_contiguous<T: 'static>(
    lhs: &[T],
    rhs: &[T],
    out: &mut [T],
    m: usize,
    k: usize,
    n: usize,
    zero: T,
    one: T,
) {
    // gemm computes: dst := alpha×dst + beta×lhs×rhs
    // We want: out = lhs × rhs
    // So: read_dst = false (ignore existing dst), beta = 1.0
    // Note: gemm expects (column_stride, row_stride) order
    // For row-major: col_stride = 1, row_stride = num_cols
    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            out.as_mut_ptr(),
            1,          // dst_cs: col stride (row-major = 1)
            n as isize, // dst_rs: row stride (row-major = num_cols)
            false,      // read_dst: false = overwrite, don't accumulate
            lhs.as_ptr(),
            1,          // lhs_cs: col stride
            k as isize, // lhs_rs: row stride (num cols of lhs = k)
            rhs.as_ptr(),
            1,                           // rhs_cs: col stride
            n as isize,                  // rhs_rs: row stride (num cols of rhs = n)
            zero,                        // alpha (ignored when read_dst = false)
            one,                         // beta
            false,                       // conj_dst
            false,                       // conj_lhs
            false,                       // conj_rhs
            gemm::Parallelism::Rayon(0), // Use all available threads
        );
    }
}

/// Naive matmul implementation for types without gemm support
#[inline]
fn matmul_naive<T>(lhs: &[T], rhs: &[T], out: &mut [T], m: usize, k: usize, n: usize)
where
    T: Copy + Default + StdAdd<Output = T> + StdMul<Output = T>,
{
    for i in 0..m {
        for j in 0..n {
            let mut sum = T::default();
            for l in 0..k {
                sum = sum + lhs[i * k + l] * rhs[l * n + j];
            }
            out[i * n + j] = sum;
        }
    }
}

/// Trait for dispatching matmul to the appropriate implementation
trait MatmulImpl: SimdElement + Default + StdAdd<Output = Self> + StdMul<Output = Self> {
    fn matmul_contiguous(
        lhs: &[Self],
        rhs: &[Self],
        out: &mut [Self],
        m: usize,
        k: usize,
        n: usize,
    );
}

impl MatmulImpl for f32 {
    #[inline(always)]
    fn matmul_contiguous(
        lhs: &[Self],
        rhs: &[Self],
        out: &mut [Self],
        m: usize,
        k: usize,
        n: usize,
    ) {
        matmul_contiguous(lhs, rhs, out, m, k, n, 0.0, 1.0);
    }
}

impl MatmulImpl for f64 {
    #[inline(always)]
    fn matmul_contiguous(
        lhs: &[Self],
        rhs: &[Self],
        out: &mut [Self],
        m: usize,
        k: usize,
        n: usize,
    ) {
        matmul_contiguous(lhs, rhs, out, m, k, n, 0.0, 1.0);
    }
}

// Fallback implementation for other types
macro_rules! impl_matmul_naive {
    ($($t:ty),*) => {
        $(
            impl MatmulImpl for $t {
                #[inline]
                fn matmul_contiguous(lhs: &[Self], rhs: &[Self], out: &mut [Self], m: usize, k: usize, n: usize) {
                    matmul_naive(lhs, rhs, out, m, k, n);
                }
            }
        )*
    };
}

impl_matmul_naive!(i8, i16, i32, i64, u8, u16, u32, u64);

/// Strided matmul for non-contiguous tensors (slower path)
fn matmul_strided<T: SimdElement + Default + StdAdd<Output = T> + StdMul<Output = T>>(
    lhs: &ConcreteTensor<T, 2>,
    rhs: &ConcreteTensor<T, 2>,
    out: &mut ConcreteTensor<T, 2>,
) {
    let m = lhs.shape()[0];
    let k = lhs.shape()[1];
    let n = rhs.shape()[1];

    for i in 0..m {
        for j in 0..n {
            let mut sum = T::default();
            for l in 0..k {
                let lhs_idx = lhs.layout.linear_index(&[i, l]);
                let rhs_idx = rhs.layout.linear_index(&[l, j]);
                sum = sum + lhs.backing[lhs_idx] * rhs.backing[rhs_idx];
            }
            let out_idx = out.layout.linear_index(&[i, j]);
            out.backing[out_idx] = sum;
        }
    }
}

/// Generic helper for binary tensor operations
fn binary_tensor_op<E, const RANK: usize, T1, T2, Op>(lhs: &T1, rhs: &T2) -> ConcreteTensor<E, RANK>
where
    E: SimdElement + Default,
    Op: SimdBinaryOp<E>,
    T1: ResolveTensor<Elem = E, Concrete = ConcreteTensor<E, RANK>>,
    T2: ResolveTensor<Elem = E, Concrete = ConcreteTensor<E, RANK>>,
{
    let lhs_concrete = lhs.to_concrete();
    let rhs_concrete = rhs.to_concrete();

    let shape: [usize; RANK] = lhs_concrete
        .shape()
        .try_into()
        .expect("Shape length mismatch");
    let mut output = ConcreteTensor::<E, RANK>::zeros(shape);

    let lhs_layout = lhs_concrete.layout().clone();
    let rhs_layout = rhs_concrete.layout().clone();
    let out_layout = output.layout().clone();
    let tensor_shape: Box<[usize]> = lhs_concrete.shape().into();

    let all_contiguous =
        lhs_layout.is_contiguous() && rhs_layout.is_contiguous() && out_layout.is_contiguous();

    if all_contiguous {
        binary_op_contiguous::<E, Op>(
            &lhs_concrete.backing,
            &rhs_concrete.backing,
            &mut output.backing,
        );
    } else {
        let lhs_data = lhs_concrete.data();
        let rhs_data = rhs_concrete.data();
        let out_data = output.data_mut();

        for indices in IndexIterator::new(&tensor_shape) {
            let lhs_idx = lhs_layout.linear_index(&indices);
            let rhs_idx = rhs_layout.linear_index(&indices);
            let out_idx = out_layout.linear_index(&indices);
            out_data[out_idx] = Op::apply_scalar(lhs_data[lhs_idx], rhs_data[rhs_idx]);
        }
    }

    output
}

/// Generic helper for unary tensor operations
fn unary_tensor_op<E, const RANK: usize, T, Op>(input: &T) -> ConcreteTensor<E, RANK>
where
    E: SimdElement + Default,
    Op: SimdUnaryOp<E>,
    T: ResolveTensor<Elem = E, Concrete = ConcreteTensor<E, RANK>>,
{
    let input_concrete = input.to_concrete();

    let shape: [usize; RANK] = input_concrete
        .shape()
        .try_into()
        .expect("Shape length mismatch");
    let mut output = ConcreteTensor::<E, RANK>::zeros(shape);

    let in_layout = input_concrete.layout().clone();
    let out_layout = output.layout().clone();
    let tensor_shape: Box<[usize]> = input_concrete.shape().into();

    let all_contiguous = in_layout.is_contiguous() && out_layout.is_contiguous();

    if all_contiguous {
        unary_op_contiguous::<E, Op>(&input_concrete.backing, &mut output.backing);
    } else {
        let in_data = input_concrete.data();
        let out_data = output.data_mut();

        for indices in IndexIterator::new(&tensor_shape) {
            let in_idx = in_layout.linear_index(&indices);
            let out_idx = out_layout.linear_index(&indices);
            out_data[out_idx] = Op::apply_scalar(in_data[in_idx]);
        }
    }

    output
}

/// Optimized binary tensor operation that works directly with ConcreteTensor references
/// Avoids cloning by working with references directly
#[inline(always)]
fn binary_tensor_op_ref<E, const RANK: usize, Op>(
    lhs: &ConcreteTensor<E, RANK>,
    rhs: &ConcreteTensor<E, RANK>,
) -> ConcreteTensor<E, RANK>
where
    E: SimdElement,
    Op: SimdBinaryOp<E>,
{
    let shape: [usize; RANK] = lhs.shape().try_into().expect("Shape length mismatch");
    // SAFETY: We write to all elements before returning
    let mut output = ConcreteTensor::<E, RANK>::uninit_unchecked(shape);

    // Fast path: all contiguous (common case)
    // Output is always contiguous since we just created it
    let all_contiguous = lhs.layout.is_contiguous() && rhs.layout.is_contiguous();

    if all_contiguous {
        binary_op_contiguous::<E, Op>(&lhs.backing, &rhs.backing, &mut output.backing);
    } else {
        let tensor_shape = &lhs.layout.shape;
        for indices in IndexIterator::new(tensor_shape) {
            let lhs_idx = lhs.layout.linear_index(&indices);
            let rhs_idx = rhs.layout.linear_index(&indices);
            let out_idx = output.layout.linear_index(&indices);
            output.backing[out_idx] = Op::apply_scalar(lhs.backing[lhs_idx], rhs.backing[rhs_idx]);
        }
    }

    output
}

/// Optimized unary tensor operation that works directly with ConcreteTensor references
#[inline(always)]
fn unary_tensor_op_ref<E, const RANK: usize, Op>(
    input: &ConcreteTensor<E, RANK>,
) -> ConcreteTensor<E, RANK>
where
    E: SimdElement,
    Op: SimdUnaryOp<E>,
{
    let shape: [usize; RANK] = input.shape().try_into().expect("Shape length mismatch");
    // SAFETY: We write to all elements before returning
    let mut output = ConcreteTensor::<E, RANK>::uninit_unchecked(shape);

    // Output is always contiguous since we just created it
    let all_contiguous = input.layout.is_contiguous();

    if all_contiguous {
        unary_op_contiguous::<E, Op>(&input.backing, &mut output.backing);
    } else {
        let tensor_shape = &input.layout.shape;
        for indices in IndexIterator::new(tensor_shape) {
            let in_idx = input.layout.linear_index(&indices);
            let out_idx = output.layout.linear_index(&indices);
            output.backing[out_idx] = Op::apply_scalar(input.backing[in_idx]);
        }
    }

    output
}

/// Macro to define binary tensor operations (Add, Sub, Mul, Div)
macro_rules! define_binary_tensor_op {
    ($name:ident, $std_trait:ident, $simd_op:ty, $error_msg:literal) => {
        pub struct $name<
            E: SimdElement,
            const RANK: usize,
            T1: Tensor<Elem = E>,
            T2: Tensor<Elem = E>,
        > {
            lhs: T1,
            rhs: T2,
            _marker: std::marker::PhantomData<E>,
        }

        impl<E, const RANK: usize, T1, T2> $name<E, RANK, T1, T2>
        where
            E: SimdElement,
            T1: Tensor<Elem = E>,
            T2: Tensor<Elem = E>,
        {
            pub fn new(lhs: T1, rhs: T2) -> Self {
                Self {
                    lhs,
                    rhs,
                    _marker: std::marker::PhantomData,
                }
            }
        }

        impl<E, const RANK: usize, T1, T2> Tensor for $name<E, RANK, T1, T2>
        where
            E: SimdElement + $std_trait<Output = E> + Default,
            $simd_op: SimdBinaryOp<E>,
            T1: Tensor<Elem = E, Concrete = ConcreteTensor<E, RANK>>,
            T2: Tensor<Elem = E, Concrete = ConcreteTensor<E, RANK>>,
        {
            type Elem = E;
            const RANK: usize = {
                assert!(T2::RANK == T1::RANK, $error_msg);
                T1::RANK
            };
            type Concrete = ConcreteTensor<Self::Elem, RANK>;
        }

        impl<E, const RANK: usize, T1, T2> ResolveTensor for $name<E, RANK, T1, T2>
        where
            E: SimdElement + $std_trait<Output = E> + Default,
            $simd_op: SimdBinaryOp<E>,
            T1: ResolveTensor<Elem = E, Concrete = ConcreteTensor<E, RANK>>,
            T2: ResolveTensor<Elem = E, Concrete = ConcreteTensor<E, RANK>>,
        {
            fn to_concrete(&self) -> Self::Concrete {
                binary_tensor_op::<E, RANK, T1, T2, $simd_op>(&self.lhs, &self.rhs)
            }
        }
    };
}

/// Macro to define unary tensor operations (Neg, Abs, Sqrt)
macro_rules! define_unary_tensor_op {
    ($name:ident, $simd_op:ty) => {
        pub struct $name<E: SimdElement, const RANK: usize, T: Tensor<Elem = E>> {
            input: T,
            _marker: std::marker::PhantomData<E>,
        }

        impl<E, const RANK: usize, T> $name<E, RANK, T>
        where
            E: SimdElement,
            T: Tensor<Elem = E>,
        {
            pub fn new(input: T) -> Self {
                Self {
                    input,
                    _marker: std::marker::PhantomData,
                }
            }
        }

        impl<E, const RANK: usize, T> Tensor for $name<E, RANK, T>
        where
            E: SimdElement + Default,
            $simd_op: SimdUnaryOp<E>,
            T: Tensor<Elem = E, Concrete = ConcreteTensor<E, RANK>>,
        {
            type Elem = E;
            const RANK: usize = T::RANK;
            type Concrete = ConcreteTensor<Self::Elem, RANK>;
        }

        impl<E, const RANK: usize, T> ResolveTensor for $name<E, RANK, T>
        where
            E: SimdElement + Default,
            $simd_op: SimdUnaryOp<E>,
            T: ResolveTensor<Elem = E, Concrete = ConcreteTensor<E, RANK>>,
        {
            fn to_concrete(&self) -> Self::Concrete {
                unary_tensor_op::<E, RANK, T, $simd_op>(&self.input)
            }
        }
    };
    ($name:ident, $simd_op:ty, $std_trait:ident) => {
        pub struct $name<E: SimdElement, const RANK: usize, T: Tensor<Elem = E>> {
            input: T,
            _marker: std::marker::PhantomData<E>,
        }

        impl<E, const RANK: usize, T> $name<E, RANK, T>
        where
            E: SimdElement,
            T: Tensor<Elem = E>,
        {
            pub fn new(input: T) -> Self {
                Self {
                    input,
                    _marker: std::marker::PhantomData,
                }
            }
        }

        impl<E, const RANK: usize, T> Tensor for $name<E, RANK, T>
        where
            E: SimdElement + $std_trait<Output = E> + Default,
            $simd_op: SimdUnaryOp<E>,
            T: Tensor<Elem = E, Concrete = ConcreteTensor<E, RANK>>,
        {
            type Elem = E;
            const RANK: usize = T::RANK;
            type Concrete = ConcreteTensor<Self::Elem, RANK>;
        }

        impl<E, const RANK: usize, T> ResolveTensor for $name<E, RANK, T>
        where
            E: SimdElement + $std_trait<Output = E> + Default,
            $simd_op: SimdUnaryOp<E>,
            T: ResolveTensor<Elem = E, Concrete = ConcreteTensor<E, RANK>>,
        {
            fn to_concrete(&self) -> Self::Concrete {
                unary_tensor_op::<E, RANK, T, $simd_op>(&self.input)
            }
        }
    };
}

// Binary tensor operations
define_binary_tensor_op!(Add, StdAdd, AddOp, "Tensor rank mismatch in Add");
define_binary_tensor_op!(Sub, StdSub, SubOp, "Tensor rank mismatch in Sub");
define_binary_tensor_op!(Mul, StdMul, MulOp, "Tensor rank mismatch in Mul");
define_binary_tensor_op!(Div, StdDiv, DivOp, "Tensor rank mismatch in Div");

// Unary tensor operations
define_unary_tensor_op!(Neg, NegOp, StdNeg);
define_unary_tensor_op!(Abs, AbsOp);
define_unary_tensor_op!(Sqrt, SqrtOp);

/// Trait for binary operations that have SIMD support
trait SimdBinaryOp<E: SimdElement>: Copy {
    /// Apply operation to SIMD vectors
    fn apply_simd_vec<S: Simd>(simd: S, a: E::Simd<S>, b: E::Simd<S>) -> E::Simd<S>;

    /// Apply operation to scalars
    fn apply_scalar(a: E, b: E) -> E;
}

// Operation marker macro and definitions
macro_rules! define_op_marker {
    ($($name:ident),* $(,)?) => {
        $(
            #[derive(Copy, Clone)]
            struct $name;
        )*
    };
}
define_op_marker!(AddOp, SubOp, MulOp, DivOp);

// Macro to implement binary operations for numeric types
macro_rules! impl_binary_op {
    ($op:ty, $scalar_op:tt, $simd_method:ident, $elem:ty) => {
        impl SimdBinaryOp<$elem> for $op {
            #[inline(always)]
            fn apply_simd_vec<S: Simd>(simd: S, a: <$elem as SimdElement>::Simd<S>, b: <$elem as SimdElement>::Simd<S>) -> <$elem as SimdElement>::Simd<S> {
                simd.$simd_method(a, b)
            }

            #[inline(always)]
            fn apply_scalar(a: $elem, b: $elem) -> $elem {
                a $scalar_op b
            }
        }
    };
}

// Implement AddOp for all types
impl_binary_op!(AddOp, +, add_f32s, f32);
impl_binary_op!(AddOp, +, add_f64s, f64);
impl_binary_op!(AddOp, +, add_i8s, i8);
impl_binary_op!(AddOp, +, add_i16s, i16);
impl_binary_op!(AddOp, +, add_i32s, i32);
impl_binary_op!(AddOp, +, add_i64s, i64);
impl_binary_op!(AddOp, +, add_u8s, u8);
impl_binary_op!(AddOp, +, add_u16s, u16);
impl_binary_op!(AddOp, +, add_u32s, u32);
impl_binary_op!(AddOp, +, add_u64s, u64);

// Implement SubOp for all types
impl_binary_op!(SubOp, -, sub_f32s, f32);
impl_binary_op!(SubOp, -, sub_f64s, f64);
impl_binary_op!(SubOp, -, sub_i8s, i8);
impl_binary_op!(SubOp, -, sub_i16s, i16);
impl_binary_op!(SubOp, -, sub_i32s, i32);
impl_binary_op!(SubOp, -, sub_i64s, i64);
impl_binary_op!(SubOp, -, sub_u8s, u8);
impl_binary_op!(SubOp, -, sub_u16s, u16);
impl_binary_op!(SubOp, -, sub_u32s, u32);
impl_binary_op!(SubOp, -, sub_u64s, u64);

// Implement MulOp for types with SIMD multiply
impl_binary_op!(MulOp, *, mul_f32s, f32);
impl_binary_op!(MulOp, *, mul_f64s, f64);
impl_binary_op!(MulOp, *, mul_i16s, i16);
impl_binary_op!(MulOp, *, mul_i32s, i32);
impl_binary_op!(MulOp, *, mul_u16s, u16);
impl_binary_op!(MulOp, *, mul_u32s, u32);

// Implement DivOp for float types only
impl_binary_op!(DivOp, /, div_f32s, f32);
impl_binary_op!(DivOp, /, div_f64s, f64);

/// Trait for unary operations that have SIMD support
trait SimdUnaryOp<E: SimdElement>: Copy {
    /// Apply operation to SIMD vector
    fn apply_simd_vec<S: Simd>(simd: S, a: E::Simd<S>) -> E::Simd<S>;

    /// Apply operation to scalar
    fn apply_scalar(val: E) -> E;
}

// Unary operation markers
define_op_marker!(NegOp, AbsOp, SqrtOp);

// Macro for unary ops with SIMD support
macro_rules! impl_unary_op {
    ($op:ty, $scalar_fn:expr, $simd_method:ident, $elem:ty) => {
        impl SimdUnaryOp<$elem> for $op {
            #[inline(always)]
            fn apply_simd_vec<S: Simd>(
                simd: S,
                a: <$elem as SimdElement>::Simd<S>,
            ) -> <$elem as SimdElement>::Simd<S> {
                simd.$simd_method(a)
            }

            #[inline(always)]
            fn apply_scalar(val: $elem) -> $elem {
                let f: fn($elem) -> $elem = $scalar_fn;
                f(val)
            }
        }
    };
}

// NegOp implementations
impl_unary_op!(NegOp, |x: f32| -x, neg_f32s, f32);
impl_unary_op!(NegOp, |x: f64| -x, neg_f64s, f64);

// NegOp for integer types using subtraction from zero
macro_rules! impl_neg_int_op {
    ($elem:ty, $splat:ident, $sub:ident) => {
        impl SimdUnaryOp<$elem> for NegOp {
            #[inline(always)]
            fn apply_simd_vec<S: Simd>(
                simd: S,
                a: <$elem as SimdElement>::Simd<S>,
            ) -> <$elem as SimdElement>::Simd<S> {
                simd.$sub(simd.$splat(0), a)
            }

            #[inline(always)]
            fn apply_scalar(val: $elem) -> $elem {
                val.wrapping_neg()
            }
        }
    };
}

impl_neg_int_op!(i8, splat_i8s, sub_i8s);
impl_neg_int_op!(i16, splat_i16s, sub_i16s);
impl_neg_int_op!(i32, splat_i32s, sub_i32s);
impl_neg_int_op!(i64, splat_i64s, sub_i64s);

// AbsOp for floats (native SIMD support)
impl_unary_op!(AbsOp, |x: f32| x.abs(), abs_f32s, f32);
impl_unary_op!(AbsOp, |x: f64| x.abs(), abs_f64s, f64);

// AbsOp for integers using max(x, -x)
macro_rules! impl_abs_int_op {
    ($elem:ty, $splat:ident, $sub:ident, $max:ident) => {
        impl SimdUnaryOp<$elem> for AbsOp {
            #[inline(always)]
            fn apply_simd_vec<S: Simd>(
                simd: S,
                a: <$elem as SimdElement>::Simd<S>,
            ) -> <$elem as SimdElement>::Simd<S> {
                let zero = simd.$splat(0);
                let neg_a = simd.$sub(zero, a);
                simd.$max(a, neg_a)
            }

            #[inline(always)]
            fn apply_scalar(val: $elem) -> $elem {
                val.wrapping_abs()
            }
        }
    };
}

impl_abs_int_op!(i8, splat_i8s, sub_i8s, max_i8s);
impl_abs_int_op!(i16, splat_i16s, sub_i16s, max_i16s);
impl_abs_int_op!(i32, splat_i32s, sub_i32s, max_i32s);
impl_abs_int_op!(i64, splat_i64s, sub_i64s, max_i64s);

// Sqrt for floats
impl_unary_op!(SqrtOp, |x: f32| x.sqrt(), sqrt_f32s, f32);
impl_unary_op!(SqrtOp, |x: f64| x.sqrt(), sqrt_f64s, f64);

/// Trait for reduction operations that have SIMD support
trait SimdReduceOp<E: SimdElement>: Copy {
    /// Identity element for the reduction (e.g., 0 for sum, MIN for max)
    fn identity() -> E;

    /// Create a SIMD vector filled with the identity value
    fn splat_identity<S: Simd>(simd: S) -> E::Simd<S>;

    /// Combine two SIMD vectors element-wise (e.g., add for sum, max for max)
    fn combine_simd_vec<S: Simd>(simd: S, a: E::Simd<S>, b: E::Simd<S>) -> E::Simd<S>;

    /// Combine two scalar values
    fn combine_scalar(a: E, b: E) -> E;

    /// Reduce a SIMD vector to a scalar (horizontal reduction)
    fn reduce_simd_vec<S: Simd>(simd: S, v: E::Simd<S>) -> E;
}

// Reduce operation markers
define_op_marker!(SumOp, MaxOp, MinOp, ProdOp);

// Trait for scalar combine operations used by both SIMD and horizontal reduction
trait ScalarCombine<T>: Copy {
    fn combine(a: T, b: T) -> T;
}

impl ScalarCombine<f32> for SumOp {
    #[inline(always)]
    fn combine(a: f32, b: f32) -> f32 {
        a + b
    }
}
impl ScalarCombine<f64> for SumOp {
    #[inline(always)]
    fn combine(a: f64, b: f64) -> f64 {
        a + b
    }
}
impl ScalarCombine<f32> for MaxOp {
    #[inline(always)]
    fn combine(a: f32, b: f32) -> f32 {
        a.max(b)
    }
}
impl ScalarCombine<f64> for MaxOp {
    #[inline(always)]
    fn combine(a: f64, b: f64) -> f64 {
        a.max(b)
    }
}
impl ScalarCombine<f32> for MinOp {
    #[inline(always)]
    fn combine(a: f32, b: f32) -> f32 {
        a.min(b)
    }
}
impl ScalarCombine<f64> for MinOp {
    #[inline(always)]
    fn combine(a: f64, b: f64) -> f64 {
        a.min(b)
    }
}
impl ScalarCombine<f32> for ProdOp {
    #[inline(always)]
    fn combine(a: f32, b: f32) -> f32 {
        a * b
    }
}
impl ScalarCombine<f64> for ProdOp {
    #[inline(always)]
    fn combine(a: f64, b: f64) -> f64 {
        a * b
    }
}

macro_rules! impl_scalar_combine_int {
    ($op:ty, $elem:ty, $method:ident) => {
        impl ScalarCombine<$elem> for $op {
            #[inline(always)]
            fn combine(a: $elem, b: $elem) -> $elem {
                a.$method(b)
            }
        }
    };
}

// SumOp for integers (wrapping add)
impl_scalar_combine_int!(SumOp, i8, wrapping_add);
impl_scalar_combine_int!(SumOp, i16, wrapping_add);
impl_scalar_combine_int!(SumOp, i32, wrapping_add);
impl_scalar_combine_int!(SumOp, i64, wrapping_add);
impl_scalar_combine_int!(SumOp, u8, wrapping_add);
impl_scalar_combine_int!(SumOp, u16, wrapping_add);
impl_scalar_combine_int!(SumOp, u32, wrapping_add);
impl_scalar_combine_int!(SumOp, u64, wrapping_add);

// MaxOp for integers
impl_scalar_combine_int!(MaxOp, i8, max);
impl_scalar_combine_int!(MaxOp, i16, max);
impl_scalar_combine_int!(MaxOp, i32, max);
impl_scalar_combine_int!(MaxOp, i64, max);
impl_scalar_combine_int!(MaxOp, u8, max);
impl_scalar_combine_int!(MaxOp, u16, max);
impl_scalar_combine_int!(MaxOp, u32, max);
impl_scalar_combine_int!(MaxOp, u64, max);

// MinOp for integers
impl_scalar_combine_int!(MinOp, i8, min);
impl_scalar_combine_int!(MinOp, i16, min);
impl_scalar_combine_int!(MinOp, i32, min);
impl_scalar_combine_int!(MinOp, i64, min);
impl_scalar_combine_int!(MinOp, u8, min);
impl_scalar_combine_int!(MinOp, u16, min);
impl_scalar_combine_int!(MinOp, u32, min);
impl_scalar_combine_int!(MinOp, u64, min);

// ProdOp for integers (wrapping mul)
impl_scalar_combine_int!(ProdOp, i16, wrapping_mul);
impl_scalar_combine_int!(ProdOp, i32, wrapping_mul);
impl_scalar_combine_int!(ProdOp, u16, wrapping_mul);
impl_scalar_combine_int!(ProdOp, u32, wrapping_mul);

// Macro for reduce implementations with SIMD support
macro_rules! impl_reduce_op {
    ($op:ty, $elem:ty, $identity:expr, $splat:ident, $simd_combine:ident) => {
        impl SimdReduceOp<$elem> for $op {
            #[inline(always)]
            fn identity() -> $elem {
                $identity
            }

            #[inline(always)]
            fn splat_identity<S: Simd>(simd: S) -> <$elem as SimdElement>::Simd<S> {
                simd.$splat($identity)
            }

            #[inline(always)]
            fn combine_simd_vec<S: Simd>(
                simd: S,
                a: <$elem as SimdElement>::Simd<S>,
                b: <$elem as SimdElement>::Simd<S>,
            ) -> <$elem as SimdElement>::Simd<S> {
                simd.$simd_combine(a, b)
            }

            #[inline(always)]
            fn combine_scalar(a: $elem, b: $elem) -> $elem {
                <$op as ScalarCombine<$elem>>::combine(a, b)
            }

            #[inline(always)]
            fn reduce_simd_vec<S: Simd>(_simd: S, v: <$elem as SimdElement>::Simd<S>) -> $elem {
                // Convert SIMD to slice via pointer cast (safe for Pod types)
                let ptr = &v as *const _ as *const $elem;
                let len = std::mem::size_of_val(&v) / std::mem::size_of::<$elem>();
                let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
                slice.iter().copied().fold($identity, |acc, x| <$op as ScalarCombine<$elem>>::combine(acc, x))
            }
        }
    };
}

// SumOp for floats
impl_reduce_op!(SumOp, f32, 0.0, splat_f32s, add_f32s);
impl_reduce_op!(SumOp, f64, 0.0, splat_f64s, add_f64s);

// MaxOp for floats
impl_reduce_op!(MaxOp, f32, f32::NEG_INFINITY, splat_f32s, max_f32s);
impl_reduce_op!(MaxOp, f64, f64::NEG_INFINITY, splat_f64s, max_f64s);

// MinOp for floats
impl_reduce_op!(MinOp, f32, f32::INFINITY, splat_f32s, min_f32s);
impl_reduce_op!(MinOp, f64, f64::INFINITY, splat_f64s, min_f64s);

// ProdOp for floats
impl_reduce_op!(ProdOp, f32, 1.0, splat_f32s, mul_f32s);
impl_reduce_op!(ProdOp, f64, 1.0, splat_f64s, mul_f64s);

// SumOp for integers
impl_reduce_op!(SumOp, i8, 0, splat_i8s, add_i8s);
impl_reduce_op!(SumOp, i16, 0, splat_i16s, add_i16s);
impl_reduce_op!(SumOp, i32, 0, splat_i32s, add_i32s);
impl_reduce_op!(SumOp, i64, 0, splat_i64s, add_i64s);
impl_reduce_op!(SumOp, u8, 0, splat_u8s, add_u8s);
impl_reduce_op!(SumOp, u16, 0, splat_u16s, add_u16s);
impl_reduce_op!(SumOp, u32, 0, splat_u32s, add_u32s);
impl_reduce_op!(SumOp, u64, 0, splat_u64s, add_u64s);

// MaxOp for integers
impl_reduce_op!(MaxOp, i8, i8::MIN, splat_i8s, max_i8s);
impl_reduce_op!(MaxOp, i16, i16::MIN, splat_i16s, max_i16s);
impl_reduce_op!(MaxOp, i32, i32::MIN, splat_i32s, max_i32s);
impl_reduce_op!(MaxOp, i64, i64::MIN, splat_i64s, max_i64s);
impl_reduce_op!(MaxOp, u8, u8::MIN, splat_u8s, max_u8s);
impl_reduce_op!(MaxOp, u16, u16::MIN, splat_u16s, max_u16s);
impl_reduce_op!(MaxOp, u32, u32::MIN, splat_u32s, max_u32s);
impl_reduce_op!(MaxOp, u64, u64::MIN, splat_u64s, max_u64s);

// MinOp for integers
impl_reduce_op!(MinOp, i8, i8::MAX, splat_i8s, min_i8s);
impl_reduce_op!(MinOp, i16, i16::MAX, splat_i16s, min_i16s);
impl_reduce_op!(MinOp, i32, i32::MAX, splat_i32s, min_i32s);
impl_reduce_op!(MinOp, i64, i64::MAX, splat_i64s, min_i64s);
impl_reduce_op!(MinOp, u8, u8::MAX, splat_u8s, min_u8s);
impl_reduce_op!(MinOp, u16, u16::MAX, splat_u16s, min_u16s);
impl_reduce_op!(MinOp, u32, u32::MAX, splat_u32s, min_u32s);
impl_reduce_op!(MinOp, u64, u64::MAX, splat_u64s, min_u64s);

// ProdOp for integers that have SIMD multiply (i16, i32, u16, u32)
impl_reduce_op!(ProdOp, i16, 1, splat_i16s, mul_i16s);
impl_reduce_op!(ProdOp, i32, 1, splat_i32s, mul_i32s);
impl_reduce_op!(ProdOp, u16, 1, splat_u16s, mul_u16s);
impl_reduce_op!(ProdOp, u32, 1, splat_u32s, mul_u32s);

/// Helper struct for dispatching reduce operations via Arch::dispatch
struct ReduceOpDispatch<'a, E: SimdElement, Op: SimdReduceOp<E>> {
    input: &'a [E],
    _op: std::marker::PhantomData<Op>,
}

impl<E: SimdElement, Op: SimdReduceOp<E>> WithSimd for ReduceOpDispatch<'_, E, Op> {
    type Output = E;

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> E {
        let (in_simd, in_tail) = E::as_simd::<S>(self.input);

        // Use 4 accumulators for better instruction-level parallelism
        let mut acc0 = Op::splat_identity(simd);
        let mut acc1 = Op::splat_identity(simd);
        let mut acc2 = Op::splat_identity(simd);
        let mut acc3 = Op::splat_identity(simd);

        // Process 4 SIMD vectors at a time for better ILP
        let chunks = in_simd.chunks_exact(4);
        let remainder = chunks.remainder();

        for chunk in chunks {
            acc0 = Op::combine_simd_vec(simd, acc0, chunk[0]);
            acc1 = Op::combine_simd_vec(simd, acc1, chunk[1]);
            acc2 = Op::combine_simd_vec(simd, acc2, chunk[2]);
            acc3 = Op::combine_simd_vec(simd, acc3, chunk[3]);
        }

        // Handle remaining SIMD vectors
        for v in remainder {
            acc0 = Op::combine_simd_vec(simd, acc0, *v);
        }

        // Combine accumulators
        acc0 = Op::combine_simd_vec(simd, acc0, acc1);
        acc2 = Op::combine_simd_vec(simd, acc2, acc3);
        acc0 = Op::combine_simd_vec(simd, acc0, acc2);

        // Horizontal reduce SIMD to scalar
        let mut result = Op::reduce_simd_vec(simd, acc0);

        // Handle tail elements
        for &v in in_tail {
            result = Op::combine_scalar(result, v);
        }

        result
    }
}

/// Perform a reduce operation on contiguous slices using SIMD dispatch
#[inline(always)]
fn reduce_op_contiguous<E: SimdElement, Op: SimdReduceOp<E>>(input: &[E]) -> E {
    Arch::new().dispatch(ReduceOpDispatch::<E, Op> {
        input,
        _op: std::marker::PhantomData,
    })
}

/// Full reduction on tensor (handles strided case)
fn reduce_tensor_op<E: SimdElement, const RANK: usize, Op: SimdReduceOp<E>>(
    tensor: &ConcreteTensor<E, RANK>,
) -> E {
    if tensor.layout.is_contiguous() {
        reduce_op_contiguous::<E, Op>(&tensor.backing)
    } else {
        // Fall back to scalar iteration for strided tensors
        let mut result = Op::identity();
        for indices in IndexIterator::new(&tensor.layout.shape) {
            let idx = tensor.layout.linear_index(&indices);
            result = Op::combine_scalar(result, tensor.backing[idx]);
        }
        result
    }
}

/// Reduce along a specific axis, returning tensor with OUT_RANK dimensions
fn reduce_tensor_axis<
    E: SimdElement + Default,
    const RANK: usize,
    const OUT_RANK: usize,
    const AXIS: usize,
    Op: SimdReduceOp<E>,
>(
    tensor: &ConcreteTensor<E, RANK>,
) -> ConcreteTensor<E, OUT_RANK> {
    // Compute output shape (remove AXIS dimension)
    let in_shape = tensor.layout.shape.as_ref();
    let mut out_shape = [0usize; OUT_RANK];
    let mut j = 0;
    for i in 0..RANK {
        if i != AXIS {
            out_shape[j] = in_shape[i];
            j += 1;
        }
    }

    let mut output = ConcreteTensor::<E, OUT_RANK>::zeros(out_shape);
    let reduce_dim = in_shape[AXIS];

    // Pre-compute strides for the reduction axis for faster linear index calculation
    let axis_stride = tensor.layout.strides[AXIS];

    // Iterate over output indices and reduce along AXIS
    // Use fixed-size array to avoid allocation
    let mut in_indices = [0usize; RANK];
    for out_indices in IndexIterator::new(&out_shape) {
        // Build base input indices (with AXIS = 0)
        let mut j = 0;
        for i in 0..RANK {
            if i == AXIS {
                in_indices[i] = 0;
            } else {
                in_indices[i] = out_indices[j];
                j += 1;
            }
        }

        // Get base index and reduce along axis using stride
        let base_idx = tensor.layout.linear_index(&in_indices);
        let mut acc = Op::identity();
        for k in 0..reduce_dim {
            let in_idx = base_idx + k * axis_stride;
            acc = Op::combine_scalar(acc, tensor.backing[in_idx]);
        }
        let out_idx = output.layout.linear_index(&out_indices);
        output.backing[out_idx] = acc;
    }

    output
}

/// Helper struct for dispatching binary operations via Arch::dispatch
struct BinaryOpDispatch<'a, E: SimdElement, Op: SimdBinaryOp<E>> {
    lhs: &'a [E],
    rhs: &'a [E],
    out: &'a mut [E],
    _op: std::marker::PhantomData<Op>,
}

impl<E: SimdElement, Op: SimdBinaryOp<E>> WithSimd for BinaryOpDispatch<'_, E, Op> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let (lhs_simd, lhs_tail) = E::as_simd::<S>(self.lhs);
        let (rhs_simd, rhs_tail) = E::as_simd::<S>(self.rhs);
        let (out_simd, out_tail) = E::as_mut_simd::<S>(self.out);

        for ((a, b), c) in lhs_simd
            .iter()
            .zip(rhs_simd.iter())
            .zip(out_simd.iter_mut())
        {
            *c = Op::apply_simd_vec(simd, *a, *b);
        }

        for ((a, b), c) in lhs_tail
            .iter()
            .zip(rhs_tail.iter())
            .zip(out_tail.iter_mut())
        {
            *c = Op::apply_scalar(*a, *b);
        }
    }
}

/// Perform a binary operation on contiguous slices using SIMD dispatch
#[inline(always)]
fn binary_op_contiguous<E: SimdElement, Op: SimdBinaryOp<E>>(lhs: &[E], rhs: &[E], out: &mut [E]) {
    Arch::new().dispatch(BinaryOpDispatch::<E, Op> {
        lhs,
        rhs,
        out,
        _op: std::marker::PhantomData,
    });
}

/// Helper struct for dispatching unary operations via Arch::dispatch
struct UnaryOpDispatch<'a, E: SimdElement, Op: SimdUnaryOp<E>> {
    input: &'a [E],
    out: &'a mut [E],
    _op: std::marker::PhantomData<Op>,
}

impl<E: SimdElement, Op: SimdUnaryOp<E>> WithSimd for UnaryOpDispatch<'_, E, Op> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let (in_simd, in_tail) = E::as_simd::<S>(self.input);
        let (out_simd, out_tail) = E::as_mut_simd::<S>(self.out);

        for (a, c) in in_simd.iter().zip(out_simd.iter_mut()) {
            *c = Op::apply_simd_vec(simd, *a);
        }

        for (a, c) in in_tail.iter().zip(out_tail.iter_mut()) {
            *c = Op::apply_scalar(*a);
        }
    }
}

/// Perform a unary operation on contiguous slices using SIMD dispatch
#[inline(always)]
fn unary_op_contiguous<E: SimdElement, Op: SimdUnaryOp<E>>(input: &[E], out: &mut [E]) {
    Arch::new().dispatch(UnaryOpDispatch::<E, Op> {
        input,
        out,
        _op: std::marker::PhantomData,
    });
}

/// Trait for SIMD element types with associated SIMD vector type
pub trait SimdElement: Sized + Copy + Default + Pod {
    /// The SIMD vector type for this element (GAT)
    type Simd<S: Simd>: Copy;

    /// Convert slice to SIMD vectors + remainder
    fn as_simd<S: Simd>(slice: &[Self]) -> (&[Self::Simd<S>], &[Self]);
    fn as_mut_simd<S: Simd>(slice: &mut [Self]) -> (&mut [Self::Simd<S>], &mut [Self]);
}

macro_rules! impl_simd_element {
    ($elem:ty, $simd_ty:ident, $as_simd:ident, $as_mut_simd:ident) => {
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
        }
    };
}

impl_simd_element!(f32, f32s, as_simd_f32s, as_mut_simd_f32s);
impl_simd_element!(f64, f64s, as_simd_f64s, as_mut_simd_f64s);
impl_simd_element!(i8, i8s, as_simd_i8s, as_mut_simd_i8s);
impl_simd_element!(i16, i16s, as_simd_i16s, as_mut_simd_i16s);
impl_simd_element!(i32, i32s, as_simd_i32s, as_mut_simd_i32s);
impl_simd_element!(i64, i64s, as_simd_i64s, as_mut_simd_i64s);
impl_simd_element!(u8, u8s, as_simd_u8s, as_mut_simd_u8s);
impl_simd_element!(u16, u16s, as_simd_u16s, as_mut_simd_u16s);
impl_simd_element!(u32, u32s, as_simd_u32s, as_mut_simd_u32s);
impl_simd_element!(u64, u64s, as_simd_u64s, as_mut_simd_u64s);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layout_contiguous() {
        let layout = Layout::contiguous(&[2, 3, 4]);
        assert_eq!(layout.shape.as_ref(), &[2, 3, 4]);
        assert_eq!(layout.strides.as_ref(), &[12, 4, 1]);
        assert_eq!(layout.offset, 0);
        assert!(layout.is_contiguous());
        assert_eq!(layout.num_elements(), 24);
    }

    #[test]
    fn test_layout_linear_index() {
        let layout = Layout::contiguous(&[2, 3, 4]);
        assert_eq!(layout.linear_index(&[0, 0, 0]), 0);
        assert_eq!(layout.linear_index(&[0, 0, 1]), 1);
        assert_eq!(layout.linear_index(&[0, 1, 0]), 4);
        assert_eq!(layout.linear_index(&[1, 0, 0]), 12);
        assert_eq!(layout.linear_index(&[1, 2, 3]), 12 + 8 + 3);
    }

    #[test]
    fn test_layout_with_offset() {
        let layout = Layout::from_parts(
            10,
            vec![2, 3].into_boxed_slice(),
            vec![3, 1].into_boxed_slice(),
        );
        assert_eq!(layout.offset, 10);
        assert_eq!(layout.linear_index(&[0, 0]), 10);
        assert_eq!(layout.linear_index(&[1, 2]), 10 + 3 + 2);
    }

    #[test]
    fn test_concrete_tensor_creation() {
        let tensor: ConcreteTensor<f32, 2> = ConcreteTensor::zeros([2, 3]);
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.data().len(), 6);
        for i in 0..6 {
            assert_eq!(tensor.data()[i], 0.0);
        }
    }

    #[test]
    fn test_concrete_tensor_from_slice() {
        let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let tensor: ConcreteTensor<f32, 2> = ConcreteTensor::from_slice([2, 3], &data);
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.get([0, 0]), 0.0);
        assert_eq!(tensor.get([0, 1]), 1.0);
        assert_eq!(tensor.get([0, 2]), 2.0);
        assert_eq!(tensor.get([1, 0]), 3.0);
        assert_eq!(tensor.get([1, 1]), 4.0);
        assert_eq!(tensor.get([1, 2]), 5.0);
    }

    #[test]
    fn test_concrete_tensor_get_set() {
        let mut tensor: ConcreteTensor<f32, 2> = ConcreteTensor::zeros([2, 3]);
        tensor.set([0, 1], 42.0);
        tensor.set([1, 2], 100.0);
        assert_eq!(tensor.get([0, 1]), 42.0);
        assert_eq!(tensor.get([1, 2]), 100.0);
        assert_eq!(tensor.get([0, 0]), 0.0);
    }

    #[test]
    fn test_add_tensor_contiguous() {
        let lhs_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let rhs_data: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];

        let lhs: ConcreteTensor<f32, 2> = ConcreteTensor::from_slice([2, 3], &lhs_data);
        let rhs: ConcreteTensor<f32, 2> = ConcreteTensor::from_slice([2, 3], &rhs_data);

        let add = Add::new(lhs, rhs);
        let result = add.to_concrete();

        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(result.get([0, 0]), 11.0);
        assert_eq!(result.get([0, 1]), 22.0);
        assert_eq!(result.get([0, 2]), 33.0);
        assert_eq!(result.get([1, 0]), 44.0);
        assert_eq!(result.get([1, 1]), 55.0);
        assert_eq!(result.get([1, 2]), 66.0);
    }

    #[test]
    fn test_add_tensor_1d() {
        let lhs: ConcreteTensor<i32, 1> = ConcreteTensor::from_slice([4], &[1, 2, 3, 4]);
        let rhs: ConcreteTensor<i32, 1> = ConcreteTensor::from_slice([4], &[10, 20, 30, 40]);

        let add = Add::new(lhs, rhs);
        let result = add.to_concrete();

        assert_eq!(result.shape(), &[4]);
        assert_eq!(result.get([0]), 11);
        assert_eq!(result.get([1]), 22);
        assert_eq!(result.get([2]), 33);
        assert_eq!(result.get([3]), 44);
    }

    #[test]
    fn test_add_tensor_3d() {
        let lhs_data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let rhs_data: Vec<f64> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

        let lhs: ConcreteTensor<f64, 3> = ConcreteTensor::from_slice([2, 2, 2], &lhs_data);
        let rhs: ConcreteTensor<f64, 3> = ConcreteTensor::from_slice([2, 2, 2], &rhs_data);

        let add = Add::new(lhs, rhs);
        let result = add.to_concrete();

        assert_eq!(result.shape(), &[2, 2, 2]);
        assert!((result.get([0, 0, 0]) - 1.1).abs() < 1e-10);
        assert!((result.get([0, 0, 1]) - 2.2).abs() < 1e-10);
        assert!((result.get([1, 1, 1]) - 8.8).abs() < 1e-10);
    }

    #[test]
    fn test_add_large_tensor() {
        // Test with a larger tensor to exercise SIMD paths
        let size = 1024;
        let lhs_data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let rhs_data: Vec<f32> = (0..size).map(|i| (i * 2) as f32).collect();

        let lhs: ConcreteTensor<f32, 1> = ConcreteTensor::from_slice([size], &lhs_data);
        let rhs: ConcreteTensor<f32, 1> = ConcreteTensor::from_slice([size], &rhs_data);

        let add = Add::new(lhs, rhs);
        let result = add.to_concrete();

        for i in 0..size {
            assert_eq!(result.get([i]), (i + i * 2) as f32);
        }
    }

    #[test]
    fn test_index_iterator() {
        let shape = [2, 3];
        let indices: Vec<Vec<usize>> = IndexIterator::new(&shape).collect();

        assert_eq!(indices.len(), 6);
        assert_eq!(indices[0], vec![0, 0]);
        assert_eq!(indices[1], vec![0, 1]);
        assert_eq!(indices[2], vec![0, 2]);
        assert_eq!(indices[3], vec![1, 0]);
        assert_eq!(indices[4], vec![1, 1]);
        assert_eq!(indices[5], vec![1, 2]);
    }

    #[test]
    fn test_index_iterator_empty() {
        let shape = [0, 3];
        let indices: Vec<Vec<usize>> = IndexIterator::new(&shape).collect();
        assert!(indices.is_empty());
    }

    #[test]
    fn test_sub_tensor_contiguous() {
        let lhs_data: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let rhs_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let lhs: ConcreteTensor<f32, 2> = ConcreteTensor::from_slice([2, 3], &lhs_data);
        let rhs: ConcreteTensor<f32, 2> = ConcreteTensor::from_slice([2, 3], &rhs_data);

        let sub = Sub::new(lhs, rhs);
        let result = sub.to_concrete();

        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(result.get([0, 0]), 9.0);
        assert_eq!(result.get([0, 1]), 18.0);
        assert_eq!(result.get([0, 2]), 27.0);
        assert_eq!(result.get([1, 0]), 36.0);
        assert_eq!(result.get([1, 1]), 45.0);
        assert_eq!(result.get([1, 2]), 54.0);
    }

    #[test]
    fn test_sub_large_tensor() {
        let size = 1024;
        let lhs_data: Vec<f32> = (0..size).map(|i| (i * 3) as f32).collect();
        let rhs_data: Vec<f32> = (0..size).map(|i| i as f32).collect();

        let lhs: ConcreteTensor<f32, 1> = ConcreteTensor::from_slice([size], &lhs_data);
        let rhs: ConcreteTensor<f32, 1> = ConcreteTensor::from_slice([size], &rhs_data);

        let sub = Sub::new(lhs, rhs);
        let result = sub.to_concrete();

        for i in 0..size {
            assert_eq!(result.get([i]), (i * 2) as f32);
        }
    }

    #[test]
    fn test_mul_tensor_contiguous() {
        let lhs_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let rhs_data: Vec<f32> = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0];

        let lhs: ConcreteTensor<f32, 2> = ConcreteTensor::from_slice([2, 3], &lhs_data);
        let rhs: ConcreteTensor<f32, 2> = ConcreteTensor::from_slice([2, 3], &rhs_data);

        let mul = Mul::new(lhs, rhs);
        let result = mul.to_concrete();

        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(result.get([0, 0]), 2.0);
        assert_eq!(result.get([0, 1]), 6.0);
        assert_eq!(result.get([0, 2]), 12.0);
        assert_eq!(result.get([1, 0]), 20.0);
        assert_eq!(result.get([1, 1]), 30.0);
        assert_eq!(result.get([1, 2]), 42.0);
    }

    #[test]
    fn test_mul_large_tensor() {
        let size = 1024;
        let lhs_data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let rhs_data: Vec<f32> = (0..size).map(|i| 2.0).collect();

        let lhs: ConcreteTensor<f32, 1> = ConcreteTensor::from_slice([size], &lhs_data);
        let rhs: ConcreteTensor<f32, 1> = ConcreteTensor::from_slice([size], &rhs_data);

        let mul = Mul::new(lhs, rhs);
        let result = mul.to_concrete();

        for i in 0..size {
            assert_eq!(result.get([i]), (i * 2) as f32);
        }
    }

    #[test]
    fn test_mul_i32() {
        let lhs: ConcreteTensor<i32, 1> = ConcreteTensor::from_slice([4], &[1, 2, 3, 4]);
        let rhs: ConcreteTensor<i32, 1> = ConcreteTensor::from_slice([4], &[10, 20, 30, 40]);

        let mul = Mul::new(lhs, rhs);
        let result = mul.to_concrete();

        assert_eq!(result.get([0]), 10);
        assert_eq!(result.get([1]), 40);
        assert_eq!(result.get([2]), 90);
        assert_eq!(result.get([3]), 160);
    }

    #[test]
    fn test_div_tensor_contiguous() {
        let lhs_data: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let rhs_data: Vec<f32> = vec![2.0, 4.0, 5.0, 8.0, 10.0, 12.0];

        let lhs: ConcreteTensor<f32, 2> = ConcreteTensor::from_slice([2, 3], &lhs_data);
        let rhs: ConcreteTensor<f32, 2> = ConcreteTensor::from_slice([2, 3], &rhs_data);

        let div = Div::new(lhs, rhs);
        let result = div.to_concrete();

        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(result.get([0, 0]), 5.0);
        assert_eq!(result.get([0, 1]), 5.0);
        assert_eq!(result.get([0, 2]), 6.0);
        assert_eq!(result.get([1, 0]), 5.0);
        assert_eq!(result.get([1, 1]), 5.0);
        assert_eq!(result.get([1, 2]), 5.0);
    }

    #[test]
    fn test_div_large_tensor() {
        let size = 1024;
        let lhs_data: Vec<f64> = (0..size).map(|i| (i * 4) as f64).collect();
        let rhs_data: Vec<f64> = (0..size).map(|_| 2.0).collect();

        let lhs: ConcreteTensor<f64, 1> = ConcreteTensor::from_slice([size], &lhs_data);
        let rhs: ConcreteTensor<f64, 1> = ConcreteTensor::from_slice([size], &rhs_data);

        let div = Div::new(lhs, rhs);
        let result = div.to_concrete();

        for i in 0..size {
            assert_eq!(result.get([i]), (i * 2) as f64);
        }
    }

    #[test]
    fn test_neg_tensor_f32() {
        let data: Vec<f32> = vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0];
        let tensor: ConcreteTensor<f32, 2> = ConcreteTensor::from_slice([2, 3], &data);

        let neg = Neg::new(tensor);
        let result = neg.to_concrete();

        assert_eq!(result.get([0, 0]), -1.0);
        assert_eq!(result.get([0, 1]), 2.0);
        assert_eq!(result.get([0, 2]), -3.0);
        assert_eq!(result.get([1, 0]), 4.0);
        assert_eq!(result.get([1, 1]), -5.0);
        assert_eq!(result.get([1, 2]), 6.0);
    }

    #[test]
    fn test_neg_large_tensor() {
        let size = 1024;
        let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let tensor: ConcreteTensor<f32, 1> = ConcreteTensor::from_slice([size], &data);

        let neg = Neg::new(tensor);
        let result = neg.to_concrete();

        for i in 0..size {
            assert_eq!(result.get([i]), -(i as f32));
        }
    }

    #[test]
    fn test_abs_tensor_f32() {
        let data: Vec<f32> = vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0];
        let tensor: ConcreteTensor<f32, 2> = ConcreteTensor::from_slice([2, 3], &data);

        let abs = Abs::new(tensor);
        let result = abs.to_concrete();

        assert_eq!(result.get([0, 0]), 1.0);
        assert_eq!(result.get([0, 1]), 2.0);
        assert_eq!(result.get([0, 2]), 3.0);
        assert_eq!(result.get([1, 0]), 4.0);
        assert_eq!(result.get([1, 1]), 5.0);
        assert_eq!(result.get([1, 2]), 6.0);
    }

    #[test]
    fn test_abs_i32() {
        let data: Vec<i32> = vec![1, -2, 3, -4];
        let tensor: ConcreteTensor<i32, 1> = ConcreteTensor::from_slice([4], &data);

        let abs = Abs::new(tensor);
        let result = abs.to_concrete();

        assert_eq!(result.get([0]), 1);
        assert_eq!(result.get([1]), 2);
        assert_eq!(result.get([2]), 3);
        assert_eq!(result.get([3]), 4);
    }

    #[test]
    fn test_sqrt_tensor_f32() {
        let data: Vec<f32> = vec![1.0, 4.0, 9.0, 16.0, 25.0, 36.0];
        let tensor: ConcreteTensor<f32, 2> = ConcreteTensor::from_slice([2, 3], &data);

        let sqrt = Sqrt::new(tensor);
        let result = sqrt.to_concrete();

        assert_eq!(result.get([0, 0]), 1.0);
        assert_eq!(result.get([0, 1]), 2.0);
        assert_eq!(result.get([0, 2]), 3.0);
        assert_eq!(result.get([1, 0]), 4.0);
        assert_eq!(result.get([1, 1]), 5.0);
        assert_eq!(result.get([1, 2]), 6.0);
    }

    #[test]
    fn test_sqrt_large_tensor() {
        let size = 1024;
        let data: Vec<f32> = (0..size).map(|i| (i * i) as f32).collect();
        let tensor: ConcreteTensor<f32, 1> = ConcreteTensor::from_slice([size], &data);

        let sqrt = Sqrt::new(tensor);
        let result = sqrt.to_concrete();

        for i in 0..size {
            assert!((result.get([i]) - i as f32).abs() < 1e-5);
        }
    }

    #[test]
    fn test_sqrt_f64() {
        let data: Vec<f64> = vec![1.0, 4.0, 9.0, 16.0];
        let tensor: ConcreteTensor<f64, 1> = ConcreteTensor::from_slice([4], &data);

        let sqrt = Sqrt::new(tensor);
        let result = sqrt.to_concrete();

        assert_eq!(result.get([0]), 1.0);
        assert_eq!(result.get([1]), 2.0);
        assert_eq!(result.get([2]), 3.0);
        assert_eq!(result.get([3]), 4.0);
    }

    #[test]
    fn test_matmul_2x3_3x2() {
        // [1 2 3]   [1 2]   [22 28]
        // [4 5 6] @ [3 4] = [49 64]
        //           [5 6]
        let lhs: ConcreteTensor<f32, 2> =
            ConcreteTensor::from_slice([2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let rhs: ConcreteTensor<f32, 2> =
            ConcreteTensor::from_slice([3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let result = lhs.matmul_ref(&rhs);

        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.get([0, 0]), 22.0);
        assert_eq!(result.get([0, 1]), 28.0);
        assert_eq!(result.get([1, 0]), 49.0);
        assert_eq!(result.get([1, 1]), 64.0);
    }

    #[test]
    fn test_matmul_identity() {
        // Matrix times identity should return the original matrix
        let mat: ConcreteTensor<f32, 2> =
            ConcreteTensor::from_slice([2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let identity: ConcreteTensor<f32, 2> =
            ConcreteTensor::from_slice([3, 3], &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);

        let result = mat.matmul_ref(&identity);

        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(result.get([0, 0]), 1.0);
        assert_eq!(result.get([0, 1]), 2.0);
        assert_eq!(result.get([0, 2]), 3.0);
        assert_eq!(result.get([1, 0]), 4.0);
        assert_eq!(result.get([1, 1]), 5.0);
        assert_eq!(result.get([1, 2]), 6.0);
    }

    #[test]
    fn test_matmul_large() {
        // Test with larger matrices to exercise the gemm path
        let size = 64;
        let lhs_data: Vec<f32> = (0..size * size).map(|i| (i % 10) as f32).collect();
        let rhs_data: Vec<f32> = (0..size * size).map(|i| ((i + 1) % 10) as f32).collect();

        let lhs: ConcreteTensor<f32, 2> = ConcreteTensor::from_slice([size, size], &lhs_data);
        let rhs: ConcreteTensor<f32, 2> = ConcreteTensor::from_slice([size, size], &rhs_data);

        let result = lhs.matmul_ref(&rhs);

        assert_eq!(result.shape(), &[size, size]);

        // Verify a few elements by computing them manually
        // result[0,0] = sum(lhs[0,:] * rhs[:,0])
        let mut expected_00: f32 = 0.0;
        for k in 0..size {
            expected_00 += lhs_data[k] * rhs_data[k * size];
        }
        assert!((result.get([0, 0]) - expected_00).abs() < 1e-3);
    }

    #[test]
    fn test_matmul_f64() {
        // Test f64 path
        let lhs: ConcreteTensor<f64, 2> = ConcreteTensor::from_slice([2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let rhs: ConcreteTensor<f64, 2> = ConcreteTensor::from_slice([2, 2], &[5.0, 6.0, 7.0, 8.0]);

        let result = lhs.matmul_ref(&rhs);

        assert_eq!(result.shape(), &[2, 2]);
        // [1 2] @ [5 6] = [1*5+2*7  1*6+2*8] = [19 22]
        // [3 4]   [7 8]   [3*5+4*7  3*6+4*8]   [43 50]
        assert_eq!(result.get([0, 0]), 19.0);
        assert_eq!(result.get([0, 1]), 22.0);
        assert_eq!(result.get([1, 0]), 43.0);
        assert_eq!(result.get([1, 1]), 50.0);
    }

    #[test]
    #[should_panic(expected = "Matrix dimension mismatch")]
    fn test_matmul_shape_mismatch() {
        let lhs: ConcreteTensor<f32, 2> = ConcreteTensor::from_slice([2, 3], &[1.0; 6]);
        let rhs: ConcreteTensor<f32, 2> = ConcreteTensor::from_slice([2, 2], &[1.0; 4]);

        // This should panic because lhs columns (3) != rhs rows (2)
        let _ = lhs.matmul_ref(&rhs);
    }

    // ========== Reduce Operation Tests ==========

    #[test]
    fn test_sum_1d() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let tensor: ConcreteTensor<f32, 1> = ConcreteTensor::from_slice([5], &data);
        assert_eq!(tensor.sum(), 15.0);
    }

    #[test]
    fn test_sum_2d() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor: ConcreteTensor<f32, 2> = ConcreteTensor::from_slice([2, 3], &data);
        assert_eq!(tensor.sum(), 21.0);
    }

    #[test]
    fn test_sum_large() {
        let size = 10000;
        let data: Vec<f32> = (1..=size).map(|i| i as f32).collect();
        let tensor: ConcreteTensor<f32, 1> = ConcreteTensor::from_slice([size], &data);
        // Sum of 1..n = n*(n+1)/2
        let expected = (size * (size + 1) / 2) as f32;
        assert!((tensor.sum() - expected).abs() < 1.0);
    }

    #[test]
    fn test_max_1d() {
        let data: Vec<f32> = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let tensor: ConcreteTensor<f32, 1> = ConcreteTensor::from_slice([8], &data);
        assert_eq!(tensor.max(), 9.0);
    }

    #[test]
    fn test_max_negative() {
        let data: Vec<f32> = vec![-3.0, -1.0, -4.0, -1.0, -5.0];
        let tensor: ConcreteTensor<f32, 1> = ConcreteTensor::from_slice([5], &data);
        assert_eq!(tensor.max(), -1.0);
    }

    #[test]
    fn test_min_1d() {
        let data: Vec<f32> = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let tensor: ConcreteTensor<f32, 1> = ConcreteTensor::from_slice([8], &data);
        assert_eq!(tensor.min(), 1.0);
    }

    #[test]
    fn test_min_negative() {
        let data: Vec<f32> = vec![-3.0, -1.0, -4.0, -1.0, -5.0];
        let tensor: ConcreteTensor<f32, 1> = ConcreteTensor::from_slice([5], &data);
        assert_eq!(tensor.min(), -5.0);
    }

    #[test]
    fn test_prod_1d() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let tensor: ConcreteTensor<f32, 1> = ConcreteTensor::from_slice([5], &data);
        assert_eq!(tensor.prod(), 120.0); // 5! = 120
    }

    #[test]
    fn test_prod_2d() {
        let data: Vec<f32> = vec![2.0, 3.0, 4.0, 5.0];
        let tensor: ConcreteTensor<f32, 2> = ConcreteTensor::from_slice([2, 2], &data);
        assert_eq!(tensor.prod(), 120.0);
    }

    #[test]
    fn test_reduce_i32() {
        let data: Vec<i32> = vec![1, 2, 3, 4, 5];
        let tensor: ConcreteTensor<i32, 1> = ConcreteTensor::from_slice([5], &data);
        assert_eq!(tensor.sum(), 15);
        assert_eq!(tensor.max(), 5);
        assert_eq!(tensor.min(), 1);
    }

    #[test]
    fn test_reduce_f64() {
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let tensor: ConcreteTensor<f64, 1> = ConcreteTensor::from_slice([4], &data);
        assert_eq!(tensor.sum(), 10.0);
        assert_eq!(tensor.max(), 4.0);
        assert_eq!(tensor.min(), 1.0);
        assert_eq!(tensor.prod(), 24.0);
    }

    #[test]
    fn test_sum_axis_2d_axis0() {
        // [[1, 2, 3],
        //  [4, 5, 6]]
        // sum along axis 0 -> [5, 7, 9]
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor: ConcreteTensor<f32, 2> = ConcreteTensor::from_slice([2, 3], &data);

        let result: ConcreteTensor<f32, 1> = tensor.sum_axis::<1, 0>();

        assert_eq!(result.shape(), &[3]);
        assert_eq!(result.get([0]), 5.0);
        assert_eq!(result.get([1]), 7.0);
        assert_eq!(result.get([2]), 9.0);
    }

    #[test]
    fn test_sum_axis_2d_axis1() {
        // [[1, 2, 3],
        //  [4, 5, 6]]
        // sum along axis 1 -> [6, 15]
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor: ConcreteTensor<f32, 2> = ConcreteTensor::from_slice([2, 3], &data);

        let result: ConcreteTensor<f32, 1> = tensor.sum_axis::<1, 1>();

        assert_eq!(result.shape(), &[2]);
        assert_eq!(result.get([0]), 6.0);
        assert_eq!(result.get([1]), 15.0);
    }

    #[test]
    fn test_max_axis_2d() {
        // [[1, 5, 3],
        //  [4, 2, 6]]
        // max along axis 0 -> [4, 5, 6]
        let data: Vec<f32> = vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
        let tensor: ConcreteTensor<f32, 2> = ConcreteTensor::from_slice([2, 3], &data);

        let result: ConcreteTensor<f32, 1> = tensor.max_axis::<1, 0>();

        assert_eq!(result.shape(), &[3]);
        assert_eq!(result.get([0]), 4.0);
        assert_eq!(result.get([1]), 5.0);
        assert_eq!(result.get([2]), 6.0);
    }

    #[test]
    fn test_min_axis_2d() {
        // [[1, 5, 3],
        //  [4, 2, 6]]
        // min along axis 1 -> [1, 2]
        let data: Vec<f32> = vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
        let tensor: ConcreteTensor<f32, 2> = ConcreteTensor::from_slice([2, 3], &data);

        let result: ConcreteTensor<f32, 1> = tensor.min_axis::<1, 1>();

        assert_eq!(result.shape(), &[2]);
        assert_eq!(result.get([0]), 1.0);
        assert_eq!(result.get([1]), 2.0);
    }

    #[test]
    fn test_prod_axis_2d() {
        // [[1, 2],
        //  [3, 4]]
        // prod along axis 0 -> [3, 8]
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let tensor: ConcreteTensor<f32, 2> = ConcreteTensor::from_slice([2, 2], &data);

        let result: ConcreteTensor<f32, 1> = tensor.prod_axis::<1, 0>();

        assert_eq!(result.shape(), &[2]);
        assert_eq!(result.get([0]), 3.0);
        assert_eq!(result.get([1]), 8.0);
    }

    #[test]
    fn test_sum_axis_3d() {
        // 2x2x2 tensor
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let tensor: ConcreteTensor<f32, 3> = ConcreteTensor::from_slice([2, 2, 2], &data);

        // Sum along axis 0 -> 2x2 tensor
        let result: ConcreteTensor<f32, 2> = tensor.sum_axis::<2, 0>();

        assert_eq!(result.shape(), &[2, 2]);
        // [1,2;3,4] + [5,6;7,8] = [6,8;10,12]
        assert_eq!(result.get([0, 0]), 6.0);
        assert_eq!(result.get([0, 1]), 8.0);
        assert_eq!(result.get([1, 0]), 10.0);
        assert_eq!(result.get([1, 1]), 12.0);
    }

    #[test]
    fn test_reduce_single_element() {
        let tensor: ConcreteTensor<f32, 1> = ConcreteTensor::from_slice([1], &[42.0]);
        assert_eq!(tensor.sum(), 42.0);
        assert_eq!(tensor.max(), 42.0);
        assert_eq!(tensor.min(), 42.0);
        assert_eq!(tensor.prod(), 42.0);
    }
}
