//! Matrix multiplication operations

use std::ops::{Add as StdAdd, Mul as StdMul};

use crate::{ConcreteTensor, ResolvedTensor, SimdElement};

/// Trait for dispatching matmul to the appropriate implementation
pub trait MatmulImpl:
    SimdElement + Default + StdAdd<Output = Self> + StdMul<Output = Self>
{
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

/// Generic batched matrix multiplication for N-dimensional tensors (N >= 2)
/// Shape: [...batch_dims, M, K] @ [...batch_dims, K, N] -> [...batch_dims, M, N]
/// All batch dimensions must match between lhs and rhs.
pub fn batched_matmul<T: SimdElement + MatmulImpl, const R: usize>(
    lhs: &ConcreteTensor<T, R>,
    rhs: &ConcreteTensor<T, R>,
) -> ConcreteTensor<T, R> {
    const {
        assert!(
            R >= 2,
            "Matrix multiplication requires at least 2 dimensions"
        )
    };

    let lhs_shape = lhs.shape();
    let rhs_shape = rhs.shape();

    // Last two dimensions are matrix dimensions
    let m = lhs_shape[R - 2];
    let k = lhs_shape[R - 1];
    let k2 = rhs_shape[R - 2];
    let n = rhs_shape[R - 1];

    assert_eq!(
        k, k2,
        "Matrix dimension mismatch: lhs columns ({}) != rhs rows ({})",
        k, k2
    );

    // Check batch dimensions match
    for i in 0..(R - 2) {
        assert_eq!(
            lhs_shape[i], rhs_shape[i],
            "Batch dimension {} mismatch: {} != {}",
            i, lhs_shape[i], rhs_shape[i]
        );
    }

    // Compute output shape
    let mut out_shape: [usize; R] = lhs_shape.try_into().expect("Shape length mismatch");
    out_shape[R - 1] = n;

    let mut output = ConcreteTensor::<T, R>::uninit_unchecked(out_shape);

    // Compute total batch size (product of all batch dimensions)
    let batch_size: usize = lhs_shape[..R - 2].iter().product();

    let lhs_contiguous = lhs.layout().is_contiguous();
    let rhs_contiguous = rhs.layout().is_contiguous();
    let lhs_matrix_size = m * k;
    let rhs_matrix_size = k * n;
    let out_matrix_size = m * n;

    if lhs_contiguous && rhs_contiguous {
        // Fast path: iterate over flattened batch dimension
        for b in 0..batch_size {
            let lhs_offset = b * lhs_matrix_size;
            let rhs_offset = b * rhs_matrix_size;
            let out_offset = b * out_matrix_size;

            T::matmul_contiguous(
                &lhs.data()[lhs_offset..lhs_offset + lhs_matrix_size],
                &rhs.data()[rhs_offset..rhs_offset + rhs_matrix_size],
                &mut output.data_mut()[out_offset..out_offset + out_matrix_size],
                m,
                k,
                n,
            );
        }
    } else {
        // Slow path for non-contiguous tensors
        // We need to iterate over batch indices
        let batch_dims = &lhs_shape[..R - 2];
        let mut batch_indices = vec![0usize; R - 2];

        for _ in 0..batch_size {
            // Perform matmul for this batch
            for i in 0..m {
                for j in 0..n {
                    let mut sum = T::default();
                    for l in 0..k {
                        // Build full index for lhs: [...batch_indices, i, l]
                        let mut lhs_idx_arr = [0usize; R];
                        let mut rhs_idx_arr = [0usize; R];
                        let mut out_idx_arr = [0usize; R];

                        for (idx, &b) in batch_indices.iter().enumerate() {
                            lhs_idx_arr[idx] = b;
                            rhs_idx_arr[idx] = b;
                            out_idx_arr[idx] = b;
                        }
                        lhs_idx_arr[R - 2] = i;
                        lhs_idx_arr[R - 1] = l;
                        rhs_idx_arr[R - 2] = l;
                        rhs_idx_arr[R - 1] = j;

                        let lhs_idx = lhs.layout().linear_index(&lhs_idx_arr);
                        let rhs_idx = rhs.layout().linear_index(&rhs_idx_arr);
                        sum = sum + lhs.data()[lhs_idx] * rhs.data()[rhs_idx];
                    }

                    let mut out_idx_arr = [0usize; R];
                    for (idx, &b) in batch_indices.iter().enumerate() {
                        out_idx_arr[idx] = b;
                    }
                    out_idx_arr[R - 2] = i;
                    out_idx_arr[R - 1] = j;
                    let out_idx = output.layout().linear_index(&out_idx_arr);
                    output.data_mut()[out_idx] = sum;
                }
            }

            // Increment batch indices (like a multi-digit counter)
            for d in (0..batch_indices.len()).rev() {
                batch_indices[d] += 1;
                if batch_indices[d] < batch_dims[d] {
                    break;
                }
                batch_indices[d] = 0;
            }
        }
    }

    output
}

/// Matrix multiplication for N-dimensional tensors (N >= 2)
/// For 2D: [M, K] @ [K, N] -> [M, N]
/// For ND: [...batch, M, K] @ [...batch, K, N] -> [...batch, M, N]
impl<T: SimdElement + MatmulImpl, const R: usize> ConcreteTensor<T, R> {
    /// Matrix multiplication (batched for rank > 2)
    /// Panics if R < 2
    #[inline]
    pub fn matmul_ref(&self, rhs: &Self) -> Self {
        batched_matmul(self, rhs)
    }
}
