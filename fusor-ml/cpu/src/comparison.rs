//! Comparison tensor operations: Eq, Lt, Lte, Gt, Gte
//! Returns 1.0/0.0 for floats or 1/0 for integers to match GPU backend semantics.

use pulp::{Arch, Simd, WithSimd};

use crate::{
    ConcreteTensor, IndexIterator, ResolvedTensor, SimdElement,
};

/// Trait for comparison operations
pub trait SimdComparisonOp<E: SimdElement>: Copy {
    /// Apply comparison to SIMD vectors, returning mask as numeric (1.0 or 0.0)
    fn apply_simd_vec<S: Simd>(simd: S, a: E::Simd<S>, b: E::Simd<S>) -> E::Simd<S>;

    /// Apply comparison to scalars, returning 1 or 0 in the element type
    fn apply_scalar(a: E, b: E) -> E;
}

// Comparison operation markers
macro_rules! define_cmp_marker {
    ($($name:ident),* $(,)?) => {
        $(
            #[derive(Copy, Clone)]
            pub struct $name;
        )*
    };
}
define_cmp_marker!(EqOp, NeOp, LtOp, LteOp, GtOp, GteOp);

// Helper trait for types that can represent 0 and 1
trait NumericBool: SimdElement {
    fn zero() -> Self;
    fn one() -> Self;
}

impl NumericBool for f32 {
    fn zero() -> Self { 0.0 }
    fn one() -> Self { 1.0 }
}

impl NumericBool for f64 {
    fn zero() -> Self { 0.0 }
    fn one() -> Self { 1.0 }
}

impl NumericBool for i8 {
    fn zero() -> Self { 0 }
    fn one() -> Self { 1 }
}

impl NumericBool for i16 {
    fn zero() -> Self { 0 }
    fn one() -> Self { 1 }
}

impl NumericBool for i32 {
    fn zero() -> Self { 0 }
    fn one() -> Self { 1 }
}

impl NumericBool for i64 {
    fn zero() -> Self { 0 }
    fn one() -> Self { 1 }
}

impl NumericBool for u8 {
    fn zero() -> Self { 0 }
    fn one() -> Self { 1 }
}

impl NumericBool for u16 {
    fn zero() -> Self { 0 }
    fn one() -> Self { 1 }
}

impl NumericBool for u32 {
    fn zero() -> Self { 0 }
    fn one() -> Self { 1 }
}

impl NumericBool for u64 {
    fn zero() -> Self { 0 }
    fn one() -> Self { 1 }
}

// Macro for scalar-only comparison ops (convert boolean mask to 1.0/0.0)
macro_rules! impl_scalar_comparison_op {
    ($op:ty, $cmp_fn:expr, $elem:ty) => {
        impl SimdComparisonOp<$elem> for $op {
            #[inline(always)]
            fn apply_simd_vec<S: Simd>(
                _simd: S,
                a: <$elem as SimdElement>::Simd<S>,
                b: <$elem as SimdElement>::Simd<S>,
            ) -> <$elem as SimdElement>::Simd<S> {
                // Process each lane with scalar comparison
                let lane_count =
                    std::mem::size_of::<<$elem as SimdElement>::Simd<S>>() / std::mem::size_of::<$elem>();
                let mut temp_out = [<$elem>::default(); crate::MAX_SIMD_LANES];

                // Safe: cast SIMD refs to scalar slices via bytemuck
                let slice_a: &[$elem] = pulp::bytemuck::cast_slice(std::slice::from_ref(&a));
                let slice_b: &[$elem] = pulp::bytemuck::cast_slice(std::slice::from_ref(&b));

                let cmp: fn($elem, $elem) -> bool = $cmp_fn;
                for i in 0..lane_count {
                    temp_out[i] = if cmp(slice_a[i], slice_b[i]) { <$elem as NumericBool>::one() } else { <$elem as NumericBool>::zero() };
                }

                // Safe: reconstruct SIMD from scalar slice via as_simd
                let (simd_slice, _) = <$elem as SimdElement>::as_simd::<S>(&temp_out[..lane_count]);
                simd_slice[0]
            }

            #[inline(always)]
            fn apply_scalar(a: $elem, b: $elem) -> $elem {
                let cmp: fn($elem, $elem) -> bool = $cmp_fn;
                if cmp(a, b) { <$elem as NumericBool>::one() } else { <$elem as NumericBool>::zero() }
            }
        }
    };
}

// Implement comparison ops for all numeric types
macro_rules! impl_all_comparisons {
    ($($elem:ty),*) => {
        $(
            impl_scalar_comparison_op!(EqOp, |a: $elem, b: $elem| a == b, $elem);
            impl_scalar_comparison_op!(NeOp, |a: $elem, b: $elem| a != b, $elem);
            impl_scalar_comparison_op!(LtOp, |a: $elem, b: $elem| a < b, $elem);
            impl_scalar_comparison_op!(LteOp, |a: $elem, b: $elem| a <= b, $elem);
            impl_scalar_comparison_op!(GtOp, |a: $elem, b: $elem| a > b, $elem);
            impl_scalar_comparison_op!(GteOp, |a: $elem, b: $elem| a >= b, $elem);
        )*
    };
}

impl_all_comparisons!(f32, f64, i8, i16, i32, i64, u8, u16, u32, u64);

/// Helper struct for dispatching comparison operations via Arch::dispatch
struct ComparisonOpDispatch<'a, E: SimdElement, Op: SimdComparisonOp<E>> {
    lhs: &'a [E],
    rhs: &'a [E],
    out: &'a mut [E],
    _op: std::marker::PhantomData<Op>,
}

impl<E: SimdElement, Op: SimdComparisonOp<E>> WithSimd for ComparisonOpDispatch<'_, E, Op> {
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

/// Perform a comparison operation on contiguous slices
#[inline(always)]
fn comparison_op_contiguous<E: SimdElement, Op: SimdComparisonOp<E>>(
    lhs: &[E],
    rhs: &[E],
    out: &mut [E],
) {
    Arch::new().dispatch(ComparisonOpDispatch::<E, Op> {
        lhs,
        rhs,
        out,
        _op: std::marker::PhantomData,
    });
}

/// Comparison tensor operation (tensor vs tensor)
#[inline(always)]
pub(crate) fn comparison_tensor_op_ref<E, const R: usize, Op>(
    lhs: &ConcreteTensor<E, R>,
    rhs: &ConcreteTensor<E, R>,
) -> ConcreteTensor<E, R>
where
    E: SimdElement,
    Op: SimdComparisonOp<E>,
{
    let shape: [usize; R] = ResolvedTensor::shape(lhs)
        .try_into()
        .expect("Shape length mismatch");
    let mut output = ConcreteTensor::<E, R>::uninit_unchecked(shape);

    let all_contiguous = lhs.layout().is_contiguous() && rhs.layout().is_contiguous();

    if all_contiguous {
        comparison_op_contiguous::<E, Op>(lhs.data(), rhs.data(), output.data_mut());
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

/// Comparison tensor operation (tensor vs scalar)
#[inline(always)]
pub(crate) fn comparison_scalar_op_ref<E, const R: usize, Op>(
    lhs: &ConcreteTensor<E, R>,
    scalar: E,
) -> ConcreteTensor<E, R>
where
    E: SimdElement,
    Op: SimdComparisonOp<E>,
{
    let shape: [usize; R] = ResolvedTensor::shape(lhs)
        .try_into()
        .expect("Shape length mismatch");
    let mut output = ConcreteTensor::<E, R>::uninit_unchecked(shape);

    // For scalar comparison, we process element by element
    for (i, &val) in lhs.data().iter().enumerate() {
        output.data_mut()[i] = Op::apply_scalar(val, scalar);
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eq_scalar() {
        assert_eq!(EqOp::apply_scalar(1.0f32, 1.0f32), 1.0);
        assert_eq!(EqOp::apply_scalar(1.0f32, 2.0f32), 0.0);
    }

    #[test]
    fn test_lt_scalar() {
        assert_eq!(LtOp::apply_scalar(1.0f32, 2.0f32), 1.0);
        assert_eq!(LtOp::apply_scalar(2.0f32, 1.0f32), 0.0);
        assert_eq!(LtOp::apply_scalar(1.0f32, 1.0f32), 0.0);
    }

    #[test]
    fn test_comparison_tensor() {
        let a = ConcreteTensor::<f32, 1>::from_slice([4], &[1.0, 2.0, 3.0, 4.0]);
        let b = ConcreteTensor::<f32, 1>::from_slice([4], &[2.0, 2.0, 2.0, 2.0]);

        let result = comparison_tensor_op_ref::<f32, 1, LtOp>(&a, &b);
        assert_eq!(result.get([0]), 1.0); // 1 < 2
        assert_eq!(result.get([1]), 0.0); // 2 < 2
        assert_eq!(result.get([2]), 0.0); // 3 < 2
        assert_eq!(result.get([3]), 0.0); // 4 < 2
    }
}
