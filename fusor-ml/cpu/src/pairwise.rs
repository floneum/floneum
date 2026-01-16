//! Pairwise (binary) tensor operations: Add, Sub, Mul, Div

use std::ops::{Add as StdAdd, Div as StdDiv, Mul as StdMul, Sub as StdSub};

use pulp::{Arch, Simd, WithSimd};

use crate::{
    materialize_expr, ConcreteTensor, Expr, IndexIterator, ResolveTensor, ResolvedTensor,
    SimdElement, Tensor,
};

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
    let shape: [usize; RANK] = ResolvedTensor::shape(lhs)
        .try_into()
        .expect("Shape length mismatch");
    // SAFETY: We write to all elements before returning
    let mut output = ConcreteTensor::<E, RANK>::uninit_unchecked(shape);

    // Fast path: all contiguous (common case)
    // Output is always contiguous since we just created it
    let all_contiguous = lhs.layout().is_contiguous() && rhs.layout().is_contiguous();

    if all_contiguous {
        binary_op_contiguous::<E, Op>(lhs.data(), rhs.data(), output.data_mut());
    } else {
        let tensor_shape = ResolvedTensor::shape(lhs);
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

        impl<E, const RANK: usize, T1, T2> Expr for $name<E, RANK, T1, T2>
        where
            E: SimdElement + $std_trait<Output = E>,
            $simd_op: SimdBinaryOp<E>,
            T1: Expr<Elem = E> + Tensor<Elem = E>,
            T2: Expr<Elem = E> + Tensor<Elem = E>,
        {
            type Elem = E;

            #[inline(always)]
            fn eval_scalar(&self, idx: usize) -> E {
                <$simd_op>::apply_scalar(
                    self.lhs.eval_scalar(idx),
                    self.rhs.eval_scalar(idx),
                )
            }

            #[inline(always)]
            fn eval_simd<S: Simd>(&self, simd: S, base_idx: usize) -> E::Simd<S> {
                <$simd_op>::apply_simd_vec(
                    simd,
                    self.lhs.eval_simd(simd, base_idx),
                    self.rhs.eval_simd(simd, base_idx),
                )
            }

            fn len(&self) -> usize {
                self.lhs.len()
            }

            fn shape(&self) -> &[usize] {
                self.lhs.shape()
            }

            fn is_contiguous(&self) -> bool {
                self.lhs.is_contiguous() && self.rhs.is_contiguous()
            }
        }

        impl<E, const RANK: usize, T1, T2> ResolveTensor for $name<E, RANK, T1, T2>
        where
            E: SimdElement + $std_trait<Output = E> + Default,
            $simd_op: SimdBinaryOp<E>,
            T1: Expr<Elem = E> + ResolveTensor<Elem = E, Concrete = ConcreteTensor<E, RANK>>,
            T2: Expr<Elem = E> + ResolveTensor<Elem = E, Concrete = ConcreteTensor<E, RANK>>,
        {
            fn to_concrete(&self) -> Self::Concrete {
                let shape: [usize; RANK] = Expr::shape(&self.lhs)
                    .try_into()
                    .expect("Shape length mismatch");
                materialize_expr(self, shape)
            }
        }
    };
}

// Binary tensor operations
define_binary_tensor_op!(Add, StdAdd, AddOp, "Tensor rank mismatch in Add");
define_binary_tensor_op!(Sub, StdSub, SubOp, "Tensor rank mismatch in Sub");
define_binary_tensor_op!(Mul, StdMul, MulOp, "Tensor rank mismatch in Mul");
define_binary_tensor_op!(Div, StdDiv, DivOp, "Tensor rank mismatch in Div");

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ResolveTensor;

    #[test]
    fn test_add_expr() {
        let a = ConcreteTensor::<f32, 1>::from_slice([4], &[1.0, 2.0, 3.0, 4.0]);
        let b = ConcreteTensor::<f32, 1>::from_slice([4], &[10.0, 20.0, 30.0, 40.0]);

        let add_expr: Add<f32, 1, _, _> = Add::new(&a, &b);

        // Test Expr trait methods
        assert_eq!(add_expr.len(), 4);
        assert_eq!(add_expr.shape(), &[4]);
        assert!(add_expr.is_contiguous());

        // Test scalar evaluation
        assert_eq!(add_expr.eval_scalar(0), 11.0);
        assert_eq!(add_expr.eval_scalar(1), 22.0);
        assert_eq!(add_expr.eval_scalar(3), 44.0);

        // Test materialization
        let result = add_expr.to_concrete();
        assert_eq!(result.get([0]), 11.0);
        assert_eq!(result.get([3]), 44.0);
    }

    #[test]
    fn test_mul_expr() {
        let a = ConcreteTensor::<f32, 1>::from_slice([4], &[1.0, 2.0, 3.0, 4.0]);
        let b = ConcreteTensor::<f32, 1>::from_slice([4], &[2.0, 3.0, 4.0, 5.0]);

        let mul_expr: Mul<f32, 1, _, _> = Mul::new(&a, &b);

        assert_eq!(mul_expr.eval_scalar(0), 2.0);
        assert_eq!(mul_expr.eval_scalar(1), 6.0);
        assert_eq!(mul_expr.eval_scalar(2), 12.0);
        assert_eq!(mul_expr.eval_scalar(3), 20.0);
    }

    #[test]
    fn test_fused_expr_mul_add() {
        let x = ConcreteTensor::<f32, 1>::from_slice([4], &[1.0, 2.0, 3.0, 4.0]);
        let y = ConcreteTensor::<f32, 1>::from_slice([4], &[2.0, 2.0, 2.0, 2.0]);
        let z = ConcreteTensor::<f32, 1>::from_slice([4], &[10.0, 10.0, 10.0, 10.0]);

        // Create fused expression: x * y + z
        let mul_expr: Mul<f32, 1, _, _> = Mul::new(&x, &y);
        let add_expr: Add<f32, 1, _, _> = Add::new(mul_expr, &z);

        // Verify the fused expression evaluates correctly
        assert_eq!(add_expr.eval_scalar(0), 12.0); // 1*2 + 10
        assert_eq!(add_expr.eval_scalar(1), 14.0); // 2*2 + 10
        assert_eq!(add_expr.eval_scalar(2), 16.0); // 3*2 + 10
        assert_eq!(add_expr.eval_scalar(3), 18.0); // 4*2 + 10

        // Materialize and verify
        let result = add_expr.to_concrete();
        assert_eq!(result.get([0]), 12.0);
        assert_eq!(result.get([3]), 18.0);
    }

    #[test]
    fn test_sub_div_expr() {
        let a = ConcreteTensor::<f32, 1>::from_slice([3], &[10.0, 20.0, 30.0]);
        let b = ConcreteTensor::<f32, 1>::from_slice([3], &[2.0, 4.0, 5.0]);

        let sub_expr: Sub<f32, 1, _, _> = Sub::new(&a, &b);
        assert_eq!(sub_expr.eval_scalar(0), 8.0);
        assert_eq!(sub_expr.eval_scalar(1), 16.0);
        assert_eq!(sub_expr.eval_scalar(2), 25.0);

        let div_expr: Div<f32, 1, _, _> = Div::new(&a, &b);
        assert_eq!(div_expr.eval_scalar(0), 5.0);
        assert_eq!(div_expr.eval_scalar(1), 5.0);
        assert_eq!(div_expr.eval_scalar(2), 6.0);
    }
}
