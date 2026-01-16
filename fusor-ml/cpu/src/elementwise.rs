//! Elementwise (unary) tensor operations: Neg, Abs, Sqrt

use std::ops::Neg as StdNeg;

use pulp::{Arch, Simd, WithSimd};

use crate::{
    materialize_expr, ConcreteTensor, Expr, IndexIterator, ResolveTensor, ResolvedTensor,
    SimdElement, Tensor,
};

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

/// Optimized unary tensor operation that works directly with ConcreteTensor references
#[inline(always)]
pub(crate) fn unary_tensor_op_ref<E, const RANK: usize, Op>(
    input: &ConcreteTensor<E, RANK>,
) -> ConcreteTensor<E, RANK>
where
    E: SimdElement,
    Op: SimdUnaryOp<E>,
{
    let shape: [usize; RANK] = ResolvedTensor::shape(input)
        .try_into()
        .expect("Shape length mismatch");
    // SAFETY: We write to all elements before returning
    let mut output = ConcreteTensor::<E, RANK>::uninit_unchecked(shape);

    // Output is always contiguous since we just created it
    let all_contiguous = input.layout().is_contiguous();

    if all_contiguous {
        unary_op_contiguous::<E, Op>(input.data(), output.data_mut());
    } else {
        let tensor_shape = ResolvedTensor::shape(input);
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

        impl<E, const RANK: usize, T> Expr for $name<E, RANK, T>
        where
            E: SimdElement,
            $simd_op: SimdUnaryOp<E>,
            T: Expr<Elem = E> + Tensor<Elem = E>,
        {
            type Elem = E;

            #[inline(always)]
            fn eval_scalar(&self, idx: usize) -> E {
                <$simd_op>::apply_scalar(self.input.eval_scalar(idx))
            }

            #[inline(always)]
            fn eval_simd<S: Simd>(&self, simd: S, base_idx: usize) -> E::Simd<S> {
                <$simd_op>::apply_simd_vec(simd, self.input.eval_simd(simd, base_idx))
            }

            fn len(&self) -> usize {
                self.input.len()
            }

            fn shape(&self) -> &[usize] {
                self.input.shape()
            }

            fn is_contiguous(&self) -> bool {
                self.input.is_contiguous()
            }
        }

        impl<E, const RANK: usize, T> ResolveTensor for $name<E, RANK, T>
        where
            E: SimdElement + Default,
            $simd_op: SimdUnaryOp<E>,
            T: Expr<Elem = E> + ResolveTensor<Elem = E, Concrete = ConcreteTensor<E, RANK>>,
        {
            fn to_concrete(&self) -> Self::Concrete {
                let shape: [usize; RANK] = Expr::shape(&self.input)
                    .try_into()
                    .expect("Shape length mismatch");
                materialize_expr(self, shape)
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

        impl<E, const RANK: usize, T> Expr for $name<E, RANK, T>
        where
            E: SimdElement + $std_trait<Output = E>,
            $simd_op: SimdUnaryOp<E>,
            T: Expr<Elem = E> + Tensor<Elem = E>,
        {
            type Elem = E;

            #[inline(always)]
            fn eval_scalar(&self, idx: usize) -> E {
                <$simd_op>::apply_scalar(self.input.eval_scalar(idx))
            }

            #[inline(always)]
            fn eval_simd<S: Simd>(&self, simd: S, base_idx: usize) -> E::Simd<S> {
                <$simd_op>::apply_simd_vec(simd, self.input.eval_simd(simd, base_idx))
            }

            fn len(&self) -> usize {
                self.input.len()
            }

            fn shape(&self) -> &[usize] {
                self.input.shape()
            }

            fn is_contiguous(&self) -> bool {
                self.input.is_contiguous()
            }
        }

        impl<E, const RANK: usize, T> ResolveTensor for $name<E, RANK, T>
        where
            E: SimdElement + $std_trait<Output = E> + Default,
            $simd_op: SimdUnaryOp<E>,
            T: Expr<Elem = E> + ResolveTensor<Elem = E, Concrete = ConcreteTensor<E, RANK>>,
        {
            fn to_concrete(&self) -> Self::Concrete {
                let shape: [usize; RANK] = Expr::shape(&self.input)
                    .try_into()
                    .expect("Shape length mismatch");
                materialize_expr(self, shape)
            }
        }
    };
}

// Unary tensor operations
define_unary_tensor_op!(Neg, NegOp, StdNeg);
define_unary_tensor_op!(Abs, AbsOp);
define_unary_tensor_op!(Sqrt, SqrtOp);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ResolveTensor;

    #[test]
    fn test_neg_expr() {
        let a = ConcreteTensor::<f32, 1>::from_slice([4], &[1.0, -2.0, 3.0, -4.0]);

        let neg_expr: Neg<f32, 1, _> = Neg::new(&a);

        // Test Expr trait methods
        assert_eq!(neg_expr.len(), 4);
        assert_eq!(neg_expr.shape(), &[4]);
        assert!(neg_expr.is_contiguous());

        // Test scalar evaluation
        assert_eq!(neg_expr.eval_scalar(0), -1.0);
        assert_eq!(neg_expr.eval_scalar(1), 2.0);
        assert_eq!(neg_expr.eval_scalar(2), -3.0);
        assert_eq!(neg_expr.eval_scalar(3), 4.0);

        // Test materialization
        let result = neg_expr.to_concrete();
        assert_eq!(result.get([0]), -1.0);
        assert_eq!(result.get([3]), 4.0);
    }

    #[test]
    fn test_abs_expr() {
        let a = ConcreteTensor::<f32, 1>::from_slice([4], &[-1.0, 2.0, -3.0, 4.0]);

        let abs_expr: Abs<f32, 1, _> = Abs::new(&a);

        assert_eq!(abs_expr.eval_scalar(0), 1.0);
        assert_eq!(abs_expr.eval_scalar(1), 2.0);
        assert_eq!(abs_expr.eval_scalar(2), 3.0);
        assert_eq!(abs_expr.eval_scalar(3), 4.0);
    }

    #[test]
    fn test_sqrt_expr() {
        let a = ConcreteTensor::<f32, 1>::from_slice([4], &[1.0, 4.0, 9.0, 16.0]);

        let sqrt_expr: Sqrt<f32, 1, _> = Sqrt::new(&a);

        assert_eq!(sqrt_expr.eval_scalar(0), 1.0);
        assert_eq!(sqrt_expr.eval_scalar(1), 2.0);
        assert_eq!(sqrt_expr.eval_scalar(2), 3.0);
        assert_eq!(sqrt_expr.eval_scalar(3), 4.0);
    }

    #[test]
    fn test_fused_unary_chain() {
        // Test sqrt(abs(neg(x))) as a fused expression
        let x = ConcreteTensor::<f32, 1>::from_slice([4], &[-1.0, -4.0, -9.0, -16.0]);

        let neg_expr: Neg<f32, 1, _> = Neg::new(&x);
        let abs_expr: Abs<f32, 1, _> = Abs::new(neg_expr);
        let sqrt_expr: Sqrt<f32, 1, _> = Sqrt::new(abs_expr);

        // neg(-1) = 1, abs(1) = 1, sqrt(1) = 1
        assert_eq!(sqrt_expr.eval_scalar(0), 1.0);
        // neg(-4) = 4, abs(4) = 4, sqrt(4) = 2
        assert_eq!(sqrt_expr.eval_scalar(1), 2.0);
        // neg(-9) = 9, abs(9) = 9, sqrt(9) = 3
        assert_eq!(sqrt_expr.eval_scalar(2), 3.0);
        // neg(-16) = 16, abs(16) = 16, sqrt(16) = 4
        assert_eq!(sqrt_expr.eval_scalar(3), 4.0);

        // Materialize and verify
        let result = sqrt_expr.to_concrete();
        assert_eq!(result.get([0]), 1.0);
        assert_eq!(result.get([1]), 2.0);
        assert_eq!(result.get([2]), 3.0);
        assert_eq!(result.get([3]), 4.0);
    }
}
