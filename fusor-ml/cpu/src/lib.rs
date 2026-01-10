use std::ops::{Add as StdAdd, Div as StdDiv, Mul as StdMul, Neg as StdNeg, Sub as StdSub};

use aligned_vec::{ABox, AVec};
use generativity::Id;
use pulp::{Arch, Simd, WithSimd};

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

trait Tensor {
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

trait ResolveTensor<M = ()>: Tensor {
    fn to_concrete(&self) -> Self::Concrete;
}

trait ResolvedTensor: Tensor {
    fn shape(&self) -> &[usize];
    fn strides(&self) -> &[usize];
    fn offset(&self) -> usize;
    fn data(&self) -> &ABox<[Self::Elem]>;
}

trait ResolvedTensorMut: ResolvedTensor {
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
struct ConcreteTensor<T: SimdElement, const RANK: usize> {
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
}

impl<T, const RANK: usize> ResolvedTensorMut for ConcreteTensor<T, RANK>
where
    T: SimdElement,
{
    fn data_mut(&mut self) -> &mut ABox<[Self::Elem]> {
        &mut self.backing
    }
}

impl<T: SimdElement, const RANK: usize> ConcreteTensor<T, RANK> {
    /// Create a new tensor with contiguous layout from shape, filled with zeros
    fn zeros(shape: [usize; RANK]) -> Self
    where
        T: Default,
    {
        let layout = Layout::contiguous(&shape);
        let num_elements = layout.num_elements();
        let mut vec: AVec<T> = AVec::with_capacity(64, num_elements);
        for _ in 0..num_elements {
            vec.push(T::default());
        }
        let backing = vec.into_boxed_slice();
        Self { layout, backing }
    }

    /// Create a new tensor from existing data with contiguous layout
    fn from_slice(shape: [usize; RANK], data: &[T]) -> Self {
        let layout = Layout::contiguous(&shape);
        assert_eq!(layout.num_elements(), data.len());
        let mut vec: AVec<T> = AVec::with_capacity(64, data.len());
        for &item in data {
            vec.push(item);
        }
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

/// Macro to define binary tensor operations (Add, Sub, Mul, Div)
macro_rules! define_binary_tensor_op {
    ($name:ident, $std_trait:ident, $simd_op:ty, $error_msg:literal) => {
        struct $name<E: SimdElement, const RANK: usize, T1: Tensor<Elem = E>, T2: Tensor<Elem = E>> {
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
            fn new(lhs: T1, rhs: T2) -> Self {
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
        struct $name<E: SimdElement, const RANK: usize, T: Tensor<Elem = E>> {
            input: T,
            _marker: std::marker::PhantomData<E>,
        }

        impl<E, const RANK: usize, T> $name<E, RANK, T>
        where
            E: SimdElement,
            T: Tensor<Elem = E>,
        {
            fn new(input: T) -> Self {
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
        struct $name<E: SimdElement, const RANK: usize, T: Tensor<Elem = E>> {
            input: T,
            _marker: std::marker::PhantomData<E>,
        }

        impl<E, const RANK: usize, T> $name<E, RANK, T>
        where
            E: SimdElement,
            T: Tensor<Elem = E>,
        {
            fn new(input: T) -> Self {
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
            fn apply_simd_vec<S: Simd>(simd: S, a: <$elem as SimdElement>::Simd<S>) -> <$elem as SimdElement>::Simd<S> {
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

// NegOp for integer types (SIMD not yet implemented in pulp)
macro_rules! impl_unary_op_todo {
    ($op:ty, $scalar_fn:expr, $elem:ty) => {
        impl SimdUnaryOp<$elem> for $op {
            #[inline(always)]
            fn apply_simd_vec<S: Simd>(_simd: S, _a: <$elem as SimdElement>::Simd<S>) -> <$elem as SimdElement>::Simd<S> {
                todo!()
            }

            #[inline(always)]
            fn apply_scalar(val: $elem) -> $elem {
                let f: fn($elem) -> $elem = $scalar_fn;
                f(val)
            }
        }
    };
}

impl_unary_op_todo!(NegOp, |x: i8| x.wrapping_neg(), i8);
impl_unary_op_todo!(NegOp, |x: i16| x.wrapping_neg(), i16);
impl_unary_op_todo!(NegOp, |x: i32| x.wrapping_neg(), i32);
impl_unary_op_todo!(NegOp, |x: i64| x.wrapping_neg(), i64);

// AbsOp (SIMD not yet implemented in pulp)
impl_unary_op_todo!(AbsOp, |x: f32| x.abs(), f32);
impl_unary_op_todo!(AbsOp, |x: f64| x.abs(), f64);
impl_unary_op_todo!(AbsOp, |x: i8| x.abs(), i8);
impl_unary_op_todo!(AbsOp, |x: i16| x.abs(), i16);
impl_unary_op_todo!(AbsOp, |x: i32| x.abs(), i32);
impl_unary_op_todo!(AbsOp, |x: i64| x.abs(), i64);

// Sqrt for floats
impl_unary_op!(SqrtOp, |x: f32| x.sqrt(), sqrt_f32s, f32);
impl_unary_op!(SqrtOp, |x: f64| x.sqrt(), sqrt_f64s, f64);

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

        for ((a, b), c) in lhs_simd.iter().zip(rhs_simd.iter()).zip(out_simd.iter_mut()) {
            *c = Op::apply_simd_vec(simd, *a, *b);
        }

        for ((a, b), c) in lhs_tail.iter().zip(rhs_tail.iter()).zip(out_tail.iter_mut()) {
            *c = Op::apply_scalar(*a, *b);
        }
    }
}

/// Perform a binary operation on contiguous slices using SIMD dispatch
#[inline]
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
#[inline]
fn unary_op_contiguous<E: SimdElement, Op: SimdUnaryOp<E>>(input: &[E], out: &mut [E]) {
    Arch::new().dispatch(UnaryOpDispatch::<E, Op> {
        input,
        out,
        _op: std::marker::PhantomData,
    });
}

/// Trait for SIMD element types with associated SIMD vector type
trait SimdElement: Sized + Copy + Default {
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
// Mask types don't have SIMD slice conversion methods, so omitted

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

    // =============================================================================
    // Tests for Sub
    // =============================================================================

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

    // =============================================================================
    // Tests for Mul
    // =============================================================================

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

    // =============================================================================
    // Tests for Div
    // =============================================================================

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

    // =============================================================================
    // Tests for Neg
    // =============================================================================

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

    // =============================================================================
    // Tests for Abs
    // =============================================================================

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

    // =============================================================================
    // Tests for Sqrt
    // =============================================================================

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
}
