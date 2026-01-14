//! Elementwise (unary) tensor operations: Neg, Abs, Sqrt

use std::ops::Neg as StdNeg;

use pulp::{Arch, Simd, WithSimd};

use crate::{ConcreteTensor, IndexIterator, ResolveTensor, ResolvedTensor, SimdElement, Tensor};

/// Trait for unary operations that have SIMD support
pub trait SimdUnaryOp<E: SimdElement>: Copy {
    /// Apply operation to SIMD vector
    fn apply_simd_vec<S: Simd>(simd: S, a: E::Simd<S>) -> E::Simd<S>;

    /// Apply operation to scalar
    fn apply_scalar(val: E) -> E;
}

// Unary operation markers
macro_rules! define_op_marker {
    ($($name:ident),* $(,)?) => {
        $(
            #[derive(Copy, Clone)]
            pub struct $name;
        )*
    };
}
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
pub(crate) fn unary_op_contiguous<E: SimdElement, Op: SimdUnaryOp<E>>(
    input: &[E],
    out: &mut [E],
) {
    Arch::new().dispatch(UnaryOpDispatch::<E, Op> {
        input,
        out,
        _op: std::marker::PhantomData,
    });
}

/// Generic helper for unary tensor operations
pub(crate) fn unary_tensor_op<E, const RANK: usize, T, Op>(input: &T) -> ConcreteTensor<E, RANK>
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
        unary_op_contiguous::<E, Op>(input_concrete.data(), output.data_mut());
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

/// Optimized unary tensor operation that works directly with ConcreteTensor references
#[inline(always)]
pub(crate) fn unary_tensor_op_ref<E, const RANK: usize, Op>(
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
    let all_contiguous = input.layout().is_contiguous();

    if all_contiguous {
        unary_op_contiguous::<E, Op>(input.data(), output.data_mut());
    } else {
        let tensor_shape = input.shape();
        for indices in IndexIterator::new(tensor_shape) {
            let in_idx = input.layout().linear_index(&indices);
            let out_idx = output.layout().linear_index(&indices);
            output.data_mut()[out_idx] = Op::apply_scalar(input.data()[in_idx]);
        }
    }

    output
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

// Unary tensor operations
define_unary_tensor_op!(Neg, NegOp, StdNeg);
define_unary_tensor_op!(Abs, AbsOp);
define_unary_tensor_op!(Sqrt, SqrtOp);
