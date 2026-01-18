//! CPU tensor operations with SIMD acceleration

use std::ops::Deref;

use aligned_vec::ABox;
use pulp::Simd;
use pulp::bytemuck::Pod;

// Module declarations
mod cast;
mod comparison;
mod concrete_tensor;
mod conditional;
mod elementwise;
mod expr;
mod index;
mod matmul;
mod pairwise;
mod quantized;
mod reduce;
mod scalar;
mod slice_assign;
mod tensor;

/// Maximum number of SIMD lanes supported for strided tensor gather operations.
/// This covers AVX-512 with 64 x i8 lanes. Current architectures don't exceed this,
/// but this constant provides a clear point for future updates if needed.
pub(crate) const MAX_SIMD_LANES: usize = 64;

// Re-export public types
pub use concrete_tensor::ConcreteTensor;
pub use elementwise::{Abs, Cos, Exp, Exp2, Log, Log2, Neg, Sin, Sqrt, Tan, Tanh};
pub use expr::{Expr, materialize_expr};
pub use pairwise::{Add, Div, Mul, Sub};
pub use scalar::{AddScalar, DivScalar, MulScalar, SubScalar};
pub use quantized::{Dequantize, QuantizedTensor};
pub use tensor::{FloatOps, Tensor};

// Re-export GGUF types for convenience
pub use fusor_gguf::{BlockQ4_0, BlockQ4K, BlockQ5_0, BlockQ6K, BlockQ8_0, GgmlType, GgufBlock};

// Re-export TensorSlice from fusor-types
pub use fusor_types::TensorSlice;

/// A buffer holding CPU tensor data as bytes.
///
/// This type is the CPU equivalent of fusor-core's `MappedBuffer` for GPU tensors.
/// It holds the raw bytes of tensor data and implements `Deref<Target = [u8]>`
/// to work with `TensorSlice`.
pub struct CpuMappedBuffer {
    bytes: Box<[u8]>,
}

impl CpuMappedBuffer {
    /// Create a new CpuMappedBuffer from a boxed byte slice.
    pub fn new(bytes: Box<[u8]>) -> Self {
        Self { bytes }
    }
}

impl Deref for CpuMappedBuffer {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.bytes
    }
}

// Re-export operation traits and markers for public bounds
pub use cast::CastTo;
pub use comparison::{EqOp, GtOp, GteOp, LtOp, LteOp, NeOp, SimdComparisonOp};
pub use conditional::IsNonZero;
pub use elementwise::{
    AbsOp, CosOp, Exp2Op, ExpOp, Log2Op, LogOp, NegOp, SimdUnaryOp, SinOp, SqrtOp, TanOp, TanhOp,
};
pub use matmul::MatmulImpl;
pub use pairwise::{AddOp, DivOp, MulOp, SimdBinaryOp, SubOp};
pub use reduce::{MaxOp, MinOp, ProdOp, SimdReduceOp, SumOp};

// Re-export internal types used by other modules
pub(crate) use concrete_tensor::IndexIterator;

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

pub trait TensorBacking<const R: usize> {
    type Elem: SimdElement;
}

// Blanket implementation for references
impl<const R: usize, T: TensorBacking<R>> TensorBacking<R> for &T {
    type Elem = T::Elem;
}

pub trait ResolveTensor<const R: usize, M = ()>: TensorBacking<R> {
    fn to_concrete(&self) -> ConcreteTensor<Self::Elem, R>;
}

pub trait ResolvedTensor<const R: usize>: TensorBacking<R> {
    fn shape(&self) -> &[usize];
    fn strides(&self) -> &[usize];
    fn offset(&self) -> usize;
    fn data(&self) -> &ABox<[Self::Elem]>;
    fn data_mut(&mut self) -> &mut ABox<[Self::Elem]>;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_add_operator() {
        // Use Tensor::from_slice directly - cleaner API
        let a: Tensor<1, ConcreteTensor<f32, 1>> = Tensor::from_slice([4], &[1.0, 2.0, 3.0, 4.0]);
        let b: Tensor<1, ConcreteTensor<f32, 1>> =
            Tensor::from_slice([4], &[10.0, 20.0, 30.0, 40.0]);

        // Use + operator and eval() to get result
        let result = (a + b).eval();

        assert_eq!(result.get([0]), 11.0);
        assert_eq!(result.get([1]), 22.0);
        assert_eq!(result.get([2]), 33.0);
        assert_eq!(result.get([3]), 44.0);
    }

    #[test]
    fn test_tensor_sub_operator() {
        let a: Tensor<1, ConcreteTensor<f32, 1>> =
            Tensor::from_slice([4], &[10.0, 20.0, 30.0, 40.0]);
        let b: Tensor<1, ConcreteTensor<f32, 1>> = Tensor::from_slice([4], &[1.0, 2.0, 3.0, 4.0]);

        let result = (a - b).eval();

        assert_eq!(result.get([0]), 9.0);
        assert_eq!(result.get([1]), 18.0);
        assert_eq!(result.get([2]), 27.0);
        assert_eq!(result.get([3]), 36.0);
    }

    #[test]
    fn test_tensor_mul_operator() {
        let a: Tensor<1, ConcreteTensor<f32, 1>> = Tensor::from_slice([4], &[1.0, 2.0, 3.0, 4.0]);
        let b: Tensor<1, ConcreteTensor<f32, 1>> = Tensor::from_slice([4], &[2.0, 3.0, 4.0, 5.0]);

        let result = (a * b).eval();

        assert_eq!(result.get([0]), 2.0);
        assert_eq!(result.get([1]), 6.0);
        assert_eq!(result.get([2]), 12.0);
        assert_eq!(result.get([3]), 20.0);
    }

    #[test]
    fn test_tensor_div_operator() {
        let a: Tensor<1, ConcreteTensor<f32, 1>> =
            Tensor::from_slice([4], &[10.0, 20.0, 30.0, 40.0]);
        let b: Tensor<1, ConcreteTensor<f32, 1>> = Tensor::from_slice([4], &[2.0, 4.0, 5.0, 8.0]);

        let result = (a / b).eval();

        assert_eq!(result.get([0]), 5.0);
        assert_eq!(result.get([1]), 5.0);
        assert_eq!(result.get([2]), 6.0);
        assert_eq!(result.get([3]), 5.0);
    }

    #[test]
    fn test_tensor_neg_operator() {
        let a: Tensor<1, ConcreteTensor<f32, 1>> = Tensor::from_slice([4], &[1.0, -2.0, 3.0, -4.0]);

        let result = (-a).eval();

        assert_eq!(result.get([0]), -1.0);
        assert_eq!(result.get([1]), 2.0);
        assert_eq!(result.get([2]), -3.0);
        assert_eq!(result.get([3]), 4.0);
    }

    #[test]
    fn test_tensor_chained_operators() {
        // Test (a + b) * c with lazy evaluation
        let a: Tensor<1, ConcreteTensor<f32, 1>> = Tensor::from_slice([4], &[1.0, 2.0, 3.0, 4.0]);
        let b: Tensor<1, ConcreteTensor<f32, 1>> = Tensor::from_slice([4], &[1.0, 1.0, 1.0, 1.0]);
        let c: Tensor<1, ConcreteTensor<f32, 1>> = Tensor::from_slice([4], &[2.0, 2.0, 2.0, 2.0]);

        let result = ((a + b) * c).eval();

        // (1+1)*2=4, (2+1)*2=6, (3+1)*2=8, (4+1)*2=10
        assert_eq!(result.get([0]), 4.0);
        assert_eq!(result.get([1]), 6.0);
        assert_eq!(result.get([2]), 8.0);
        assert_eq!(result.get([3]), 10.0);
    }

    #[test]
    fn test_tensor_2d_operators() {
        let a: Tensor<2, ConcreteTensor<f32, 2>> =
            Tensor::from_slice([2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b: Tensor<2, ConcreteTensor<f32, 2>> =
            Tensor::from_slice([2, 3], &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);

        let result = (a + b).eval();

        assert_eq!(result.get([0, 0]), 11.0);
        assert_eq!(result.get([0, 2]), 33.0);
        assert_eq!(result.get([1, 0]), 44.0);
        assert_eq!(result.get([1, 2]), 66.0);
    }

    #[test]
    fn test_tensor_methods() {
        // Test the new methods on Tensor
        let a: Tensor<1, ConcreteTensor<f32, 1>> = Tensor::from_slice([4], &[1.0, 2.0, 3.0, 4.0]);
        let b: Tensor<1, ConcreteTensor<f32, 1>> =
            Tensor::from_slice([4], &[10.0, 20.0, 30.0, 40.0]);

        // Test &Tensor + &Tensor operator
        let result = (&a + &b).eval();
        assert_eq!(result.get([0]), 11.0);

        // Test sum reduction
        let c: Tensor<1, ConcreteTensor<f32, 1>> = Tensor::from_slice([4], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(c.sum(), 10.0);

        // Test max/min
        assert_eq!(c.max(), 4.0);
        assert_eq!(c.min(), 1.0);
    }

    #[test]
    fn test_tensor_zeros() {
        let t: Tensor<2, ConcreteTensor<f32, 2>> = Tensor::zeros([3, 4]);
        assert_eq!(t.get([0, 0]), 0.0);
        assert_eq!(t.get([2, 3]), 0.0);
    }
}
