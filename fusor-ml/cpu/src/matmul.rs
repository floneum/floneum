//! Matrix multiplication operations

use std::ops::{Add as StdAdd, Mul as StdMul};

use crate::{ConcreteTensor, ResolvedTensor, SimdElement};

/// Trait for dispatching matmul to the appropriate implementation
pub trait MatmulImpl: SimdElement + Default + StdAdd<Output = Self> + StdMul<Output = Self> {
    fn matmul_contiguous(
        lhs: &[Self],
        rhs: &[Self],
        out: &mut [Self],
        m: usize,
        k: usize,
        n: usize,
    );
}

/// Optimized matmul for contiguous tensors using gemm crate
#[inline(always)]
fn matmul_contiguous_gemm<T: 'static>(
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
        matmul_contiguous_gemm(lhs, rhs, out, m, k, n, 0.0, 1.0);
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
        matmul_contiguous_gemm(lhs, rhs, out, m, k, n, 0.0, 1.0);
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
pub(crate) fn matmul_strided<T: SimdElement + Default + StdAdd<Output = T> + StdMul<Output = T>>(
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
                let lhs_idx = lhs.layout().linear_index(&[i, l]);
                let rhs_idx = rhs.layout().linear_index(&[l, j]);
                sum = sum + lhs.data()[lhs_idx] * rhs.data()[rhs_idx];
            }
            let out_idx = out.layout().linear_index(&[i, j]);
            out.data_mut()[out_idx] = sum;
        }
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
        let lhs_contiguous = self.layout().is_contiguous();
        let rhs_contiguous = rhs.layout().is_contiguous();

        if lhs_contiguous && rhs_contiguous {
            T::matmul_contiguous(self.data(), rhs.data(), output.data_mut(), m, k, n);
        } else {
            // Slow path for non-contiguous tensors
            matmul_strided(self, rhs, &mut output);
        }

        output
    }
}
