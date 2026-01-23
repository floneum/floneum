//! Expression template system for lazy operation fusion
//!
//! This module provides the `Expr` trait and related types that enable
//! automatic fusion of elementwise operations. When multiple operations
//! are chained (e.g., `x.mul_ref(&y).add_ref(&z).sqrt_ref()`), they are
//! evaluated in a single SIMD loop instead of multiple passes.

use pulp::{Arch, Simd, WithSimd};

use crate::{ConcreteTensor, ResolvedTensor, SimdElement};

/// Trait for expressions that can be evaluated element-wise with SIMD.
///
/// This trait enables lazy evaluation and fusion of operations. Types that
/// implement `Expr` can be composed into expression trees that are evaluated
/// in a single pass when materialized.
pub trait Expr {
    /// The element type this expression produces
    type Elem: SimdElement;

    /// Evaluate at a single scalar index.
    ///
    /// This is used for:
    /// - Tail elements that don't fill a complete SIMD vector
    /// - Non-contiguous tensor access patterns
    fn eval_scalar(&self, idx: usize) -> Self::Elem;

    /// Evaluate a SIMD chunk starting at the given base index.
    ///
    /// The returned SIMD vector contains multiple consecutive elements
    /// starting at `base_idx`. The caller must ensure that there are
    /// enough elements remaining to fill a complete SIMD vector.
    fn eval_simd<S: Simd>(&self, simd: S, base_idx: usize) -> <Self::Elem as SimdElement>::Simd<S>;

    /// Total number of elements in this expression.
    fn len(&self) -> usize;

    /// The shape of this expression as a slice.
    fn shape(&self) -> &[usize];

    /// Whether this expression is contiguous in memory.
    ///
    /// When all inputs are contiguous, the evaluator can use faster
    /// direct SIMD loads instead of gathering individual elements.
    fn is_contiguous(&self) -> bool;

    /// Returns true if the expression has no elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Implement Expr for references to Expr types
impl<E: Expr> Expr for &E {
    type Elem = E::Elem;

    #[inline(always)]
    fn eval_scalar(&self, idx: usize) -> Self::Elem {
        (*self).eval_scalar(idx)
    }

    #[inline(always)]
    fn eval_simd<S: Simd>(&self, simd: S, base_idx: usize) -> <Self::Elem as SimdElement>::Simd<S> {
        (*self).eval_simd(simd, base_idx)
    }

    fn len(&self) -> usize {
        (*self).len()
    }

    fn shape(&self) -> &[usize] {
        (*self).shape()
    }

    fn is_contiguous(&self) -> bool {
        (*self).is_contiguous()
    }
}

/// Helper to get SIMD lane count for a given element type and SIMD architecture
#[inline(always)]
fn simd_lane_count<E: SimdElement, S: Simd>() -> usize {
    std::mem::size_of::<E::Simd<S>>() / std::mem::size_of::<E>()
}

/// Evaluates an expression into an output slice using SIMD.
///
/// This is the core evaluation loop that fuses all operations in an
/// expression tree into a single pass over the data.
struct ExprEvaluator<'a, E: Expr> {
    expr: &'a E,
    output: &'a mut [E::Elem],
    base_offset: usize,
}

impl<E: Expr> WithSimd for ExprEvaluator<'_, E> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let lane_count = simd_lane_count::<E::Elem, S>();
        let (out_simd, out_tail) = E::Elem::as_mut_simd::<S>(self.output);

        // Main SIMD loop - evaluates full vectors
        for (i, out) in out_simd.iter_mut().enumerate() {
            let base_idx = self.base_offset + i * lane_count;
            *out = self.expr.eval_simd(simd, base_idx);
        }

        // Scalar tail - handles remaining elements
        let simd_len = out_simd.len() * lane_count;
        for (i, out) in out_tail.iter_mut().enumerate() {
            *out = self.expr.eval_scalar(self.base_offset + simd_len + i);
        }
    }
}

/// Materialize an expression into a new ConcreteTensor.
///
/// This evaluates the entire expression tree in a single fused SIMD loop,
/// writing the results into a newly allocated tensor.
#[inline]
#[must_use = "this allocates a new tensor; discarding it wastes computation"]
pub fn materialize_expr<E: Expr, const R: usize>(
    expr: &E,
    shape: [usize; R],
) -> ConcreteTensor<E::Elem, R> {
    let mut output = ConcreteTensor::uninit_unchecked(shape);

    Arch::new().dispatch(ExprEvaluator {
        expr,
        output: output.data_mut(),
        base_offset: 0,
    });

    output
}

/// Convert a linear index to logical indices for a given shape.
///
/// This is used for strided tensor access where we need to map
/// a flat iteration index to multi-dimensional tensor coordinates.
#[inline]
pub(crate) fn linear_to_indices<const R: usize>(mut linear: usize, shape: &[usize]) -> [usize; R] {
    debug_assert_eq!(shape.len(), R);
    let mut indices = [0usize; R];

    // Work backwards through dimensions (row-major order)
    for i in (0..R).rev() {
        indices[i] = linear % shape[i];
        linear /= shape[i];
    }

    indices
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_to_indices_1d() {
        let shape = [5];
        assert_eq!(linear_to_indices::<1>(0, &shape), [0]);
        assert_eq!(linear_to_indices::<1>(4, &shape), [4]);
    }

    #[test]
    fn test_linear_to_indices_2d() {
        let shape = [3, 4]; // 3 rows, 4 cols
        assert_eq!(linear_to_indices::<2>(0, &shape), [0, 0]);
        assert_eq!(linear_to_indices::<2>(1, &shape), [0, 1]);
        assert_eq!(linear_to_indices::<2>(4, &shape), [1, 0]);
        assert_eq!(linear_to_indices::<2>(11, &shape), [2, 3]);
    }

    #[test]
    fn test_linear_to_indices_3d() {
        let shape = [2, 3, 4]; // 2x3x4 tensor
        assert_eq!(linear_to_indices::<3>(0, &shape), [0, 0, 0]);
        assert_eq!(linear_to_indices::<3>(1, &shape), [0, 0, 1]);
        assert_eq!(linear_to_indices::<3>(4, &shape), [0, 1, 0]);
        assert_eq!(linear_to_indices::<3>(12, &shape), [1, 0, 0]);
    }

    #[test]
    fn test_concrete_tensor_expr() {
        let tensor = ConcreteTensor::<f32, 1>::from_slice([4], &[1.0, 2.0, 3.0, 4.0]);

        // Test Expr trait methods
        assert_eq!(Expr::len(&tensor), 4);
        assert_eq!(Expr::shape(&tensor), &[4]);
        assert!(Expr::is_contiguous(&tensor));
        assert!(!Expr::is_empty(&tensor));

        // Test scalar evaluation
        assert_eq!(tensor.eval_scalar(0), 1.0);
        assert_eq!(tensor.eval_scalar(1), 2.0);
        assert_eq!(tensor.eval_scalar(3), 4.0);
    }

    #[test]
    fn test_materialize_expr_basic() {
        let tensor = ConcreteTensor::<f32, 1>::from_slice([4], &[1.0, 2.0, 3.0, 4.0]);

        // Materialize a simple expression (identity)
        let result: ConcreteTensor<f32, 1> = materialize_expr(&tensor, [4]);

        for i in 0..4 {
            assert_eq!(result.get([i]), tensor.get([i]));
        }
    }

    #[test]
    fn test_expr_reference_impl() {
        let tensor = ConcreteTensor::<f32, 1>::from_slice([3], &[1.0, 2.0, 3.0]);
        let tensor_ref = &tensor;

        // Test that references implement Expr correctly
        assert_eq!(Expr::len(tensor_ref), 3);
        assert_eq!(Expr::shape(tensor_ref), &[3]);
        assert!(Expr::is_contiguous(tensor_ref));
        assert_eq!(tensor_ref.eval_scalar(0), 1.0);
        assert_eq!(tensor_ref.eval_scalar(2), 3.0);
    }
}
