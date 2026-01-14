//! Pairwise (binary) tensor operations: Add, Sub, Mul, Div

use std::ops::{Add as StdAdd, Div as StdDiv, Mul as StdMul, Sub as StdSub};

use pulp::{Arch, Simd, WithSimd};

use crate::{ConcreteTensor, IndexIterator, ResolveTensor, ResolvedTensor, SimdElement, Tensor};

/// Trait for binary operations that have SIMD support
pub trait SimdBinaryOp<E: SimdElement>: Copy {
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
            pub struct $name;
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
pub(crate) fn binary_op_contiguous<E: SimdElement, Op: SimdBinaryOp<E>>(
    lhs: &[E],
    rhs: &[E],
    out: &mut [E],
) {
    Arch::new().dispatch(BinaryOpDispatch::<E, Op> {
        lhs,
        rhs,
        out,
        _op: std::marker::PhantomData,
    });
}

/// Generic helper for binary tensor operations
pub(crate) fn binary_tensor_op<E, const RANK: usize, T1, T2, Op>(
    lhs: &T1,
    rhs: &T2,
) -> ConcreteTensor<E, RANK>
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
            lhs_concrete.data(),
            rhs_concrete.data(),
            output.data_mut(),
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

/// Optimized binary tensor operation that works directly with ConcreteTensor references
/// Avoids cloning by working with references directly
#[inline(always)]
pub(crate) fn binary_tensor_op_ref<E, const RANK: usize, Op>(
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
    let all_contiguous = lhs.layout().is_contiguous() && rhs.layout().is_contiguous();

    if all_contiguous {
        binary_op_contiguous::<E, Op>(lhs.data(), rhs.data(), output.data_mut());
    } else {
        let tensor_shape = lhs.shape();
        for indices in IndexIterator::new(tensor_shape) {
            let lhs_idx = lhs.layout().linear_index(&indices);
            let rhs_idx = rhs.layout().linear_index(&indices);
            let out_idx = output.layout().linear_index(&indices);
            output.data_mut()[out_idx] = Op::apply_scalar(lhs.data()[lhs_idx], rhs.data()[rhs_idx]);
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

// Binary tensor operations
define_binary_tensor_op!(Add, StdAdd, AddOp, "Tensor rank mismatch in Add");
define_binary_tensor_op!(Sub, StdSub, SubOp, "Tensor rank mismatch in Sub");
define_binary_tensor_op!(Mul, StdMul, MulOp, "Tensor rank mismatch in Mul");
define_binary_tensor_op!(Div, StdDiv, DivOp, "Tensor rank mismatch in Div");
