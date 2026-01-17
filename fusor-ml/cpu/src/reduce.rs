//! Reduction operations: sum, max, min, prod (full tensor and axis-wise)

use pulp::{Arch, Simd, WithSimd};

use crate::{ConcreteTensor, IndexIterator, LastRank, ResolvedTensor, SimdElement};

/// Trait for reduction operations that have SIMD support
pub trait SimdReduceOp<E: SimdElement>: Copy {
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
macro_rules! define_op_marker {
    ($($name:ident),* $(,)?) => {
        $(
            #[derive(Copy, Clone)]
            pub struct $name;
        )*
    };
}
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
                // Safe: cast SIMD ref to scalar slice via bytemuck
                let slice: &[$elem] = pulp::bytemuck::cast_slice(std::slice::from_ref(&v));
                slice
                    .iter()
                    .copied()
                    .fold($identity, |acc, x| <$op as ScalarCombine<$elem>>::combine(acc, x))
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
pub(crate) fn reduce_tensor_op<E: SimdElement, const R: usize, Op: SimdReduceOp<E>>(
    tensor: &ConcreteTensor<E, R>,
) -> E {
    if tensor.layout().is_contiguous() {
        reduce_op_contiguous::<E, Op>(tensor.data())
    } else {
        // Fall back to scalar iteration for strided tensors
        let mut result = Op::identity();
        for indices in IndexIterator::new(tensor.shape()) {
            let idx = tensor.layout().linear_index(&indices);
            result = Op::combine_scalar(result, tensor.data()[idx]);
        }
        result
    }
}

/// Reduce along a specific axis, returning tensor with OUT_RANK dimensions
pub(crate) fn reduce_tensor_axis<
    E: SimdElement + Default,
    const R: usize,
    const OUT_RANK: usize,
    const AXIS: usize,
    Op: SimdReduceOp<E>,
>(
    tensor: &ConcreteTensor<E, R>,
) -> ConcreteTensor<E, OUT_RANK>
where
    ConcreteTensor<E, R>: LastRank<OUT_RANK, E>,
{
    // Compute output shape (remove AXIS dimension)
    let in_shape = tensor.shape();
    let mut out_shape = [0usize; OUT_RANK];
    let mut j = 0;
    for i in 0..R {
        if i != AXIS {
            out_shape[j] = in_shape[i];
            j += 1;
        }
    }

    let mut output = ConcreteTensor::<E, OUT_RANK>::zeros(out_shape);
    let reduce_dim = in_shape[AXIS];

    // Pre-compute strides for the reduction axis for faster linear index calculation
    let axis_stride = tensor.strides()[AXIS];

    // Iterate over output indices and reduce along AXIS
    // Use fixed-size array to avoid allocation
    let mut in_indices = [0usize; R];
    for out_indices in IndexIterator::new(&out_shape) {
        // Build base input indices (with AXIS = 0)
        let mut j = 0;
        for i in 0..R {
            if i == AXIS {
                in_indices[i] = 0;
            } else {
                in_indices[i] = out_indices[j];
                j += 1;
            }
        }

        // Get base index and reduce along axis using stride
        let base_idx = tensor.layout().linear_index(&in_indices);
        let mut acc = Op::identity();
        for k in 0..reduce_dim {
            let in_idx = base_idx + k * axis_stride;
            acc = Op::combine_scalar(acc, tensor.data()[in_idx]);
        }
        let out_idx = output.layout().linear_index(&out_indices);
        output.data_mut()[out_idx] = acc;
    }

    output
}
