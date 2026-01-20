//! Support for GGUF quantized tensors
//!
//! This module provides `QuantizedTensor` for storing and operating on
//! quantized data from GGUF files. It supports:
//! - Multiple quantization types (Q4_0, Q5_0, Q8_0, Q4K, Q6K)
//! - Eager full dequantization to f32
//! - Lazy dequantization via the `Dequantize` expression type
//! - Efficient block-by-block matrix multiplication

use aligned_vec::{ABox, AVec};
use fusor_gguf::GgufBlock;
use pulp::Simd;

use crate::expr::Expr;
use crate::{ConcreteTensor, MAX_SIMD_LANES, ResolvedTensor, SimdElement};

/// A tensor storing quantized blocks.
///
/// `QuantizedTensor<B, R>` stores data in quantized block format where `B` is
/// the block type (e.g., `BlockQ4_0`) and `R` is the tensor rank.
///
/// The innermost dimension must be a multiple of the block size. For example,
/// a [3, 256] tensor with Q4_0 quantization (block size 32) stores 3 rows of
/// 8 blocks each.
#[derive(Clone)]
pub struct QuantizedTensor<B: GgufBlock, const R: usize> {
    /// The logical shape in elements (not blocks)
    element_shape: [usize; R],
    /// The quantized blocks stored in row-major order
    blocks: ABox<[B]>,
}

impl<B: GgufBlock, const R: usize> QuantizedTensor<B, R> {
    /// Create a quantized tensor from pre-existing blocks.
    ///
    /// # Arguments
    /// * `element_shape` - The logical shape in elements (not blocks).
    ///   The innermost dimension must be a multiple of `B::BLOCK_SIZE`.
    /// * `blocks` - The quantized blocks in row-major order.
    ///
    /// # Panics
    /// Panics if:
    /// - The innermost dimension is not a multiple of the block size
    /// - The number of blocks doesn't match the shape
    pub fn from_blocks(element_shape: [usize; R], blocks: ABox<[B]>) -> Self {
        assert!(R > 0, "Tensor must have at least rank 1");
        let inner_dim = element_shape[R - 1];
        assert!(
            inner_dim % B::BLOCK_SIZE == 0,
            "Innermost dimension ({}) must be a multiple of block size ({})",
            inner_dim,
            B::BLOCK_SIZE
        );

        let expected_blocks = Self::compute_block_count(&element_shape);
        assert_eq!(
            blocks.len(),
            expected_blocks,
            "Expected {} blocks for shape {:?}, got {}",
            expected_blocks,
            element_shape,
            blocks.len()
        );

        Self {
            element_shape,
            blocks,
        }
    }

    /// Create a quantized tensor from raw bytes.
    ///
    /// This interprets the bytes as a slice of blocks using bytemuck.
    ///
    /// # Arguments
    /// * `element_shape` - The logical shape in elements (not blocks).
    /// * `bytes` - Raw bytes that will be cast to blocks.
    ///
    /// # Panics
    /// Panics if:
    /// - The bytes length is not a multiple of the block size
    /// - The innermost dimension is not a multiple of the block size
    /// - The number of blocks doesn't match the shape
    pub fn from_raw_bytes(element_shape: [usize; R], bytes: &[u8]) -> Self {
        let blocks_slice: &[B] = pulp::bytemuck::cast_slice(bytes);
        let mut vec: AVec<B> = AVec::with_capacity(64, blocks_slice.len());
        vec.extend_from_slice(blocks_slice);
        Self::from_blocks(element_shape, vec.into_boxed_slice())
    }

    /// Compute the number of blocks needed for a given element shape.
    fn compute_block_count(element_shape: &[usize; R]) -> usize {
        let total_elements: usize = element_shape.iter().product();
        total_elements / B::BLOCK_SIZE
    }

    /// Returns the logical element shape (not block shape).
    pub fn element_shape(&self) -> &[usize; R] {
        &self.element_shape
    }

    /// Returns the total number of logical elements.
    pub fn element_count(&self) -> usize {
        self.element_shape.iter().product()
    }

    /// Returns the number of blocks.
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Returns a reference to the underlying blocks.
    pub fn blocks(&self) -> &[B] {
        &self.blocks
    }

    /// Eagerly dequantize the entire tensor to f32.
    ///
    /// This allocates a new `ConcreteTensor<f32, R>` and dequantizes all blocks.
    /// For large tensors, consider using `dequantize_lazy()` instead.
    pub fn dequantize(&self) -> ConcreteTensor<f32, R> {
        let mut output = ConcreteTensor::<f32, R>::uninit_unchecked(self.element_shape);
        let out_data = output.data_mut();

        for (block_idx, block) in self.blocks.iter().enumerate() {
            let dequantized = block.dequantize();
            let start = block_idx * B::BLOCK_SIZE;
            out_data[start..start + B::BLOCK_SIZE].copy_from_slice(dequantized.as_ref());
        }

        output
    }

    /// Create a lazy dequantization expression.
    ///
    /// This returns a `Dequantize` expression that implements `Expr`,
    /// allowing it to be composed with other operations before materialization.
    pub fn dequantize_lazy(&self) -> Dequantize<'_, B, R> {
        Dequantize { source: self }
    }
}

/// Lazy dequantization expression.
///
/// This implements `Expr` for lazy evaluation of dequantized values.
/// Instead of dequantizing the entire tensor upfront, values are
/// dequantized on-demand during expression evaluation.
pub struct Dequantize<'a, B: GgufBlock, const R: usize> {
    source: &'a QuantizedTensor<B, R>,
}

impl<B: GgufBlock, const R: usize> Expr for Dequantize<'_, B, R>
where
    B::Dequantized: AsRef<[f32]>,
{
    type Elem = f32;

    #[inline(always)]
    fn eval_scalar(&self, idx: usize) -> f32 {
        let block_idx = idx / B::BLOCK_SIZE;
        let elem_idx = idx % B::BLOCK_SIZE;
        self.source.blocks[block_idx].dequantize().as_ref()[elem_idx]
    }

    #[inline(always)]
    fn eval_simd<S: Simd>(&self, _simd: S, base_idx: usize) -> <f32 as SimdElement>::Simd<S> {
        // Block boundaries don't typically align with SIMD lanes,
        // so we fall back to scalar gathering
        let lane_count =
            std::mem::size_of::<<f32 as SimdElement>::Simd<S>>() / std::mem::size_of::<f32>();
        let mut temp = [0.0f32; MAX_SIMD_LANES];
        for i in 0..lane_count {
            temp[i] = self.eval_scalar(base_idx + i);
        }
        let (simd_vec, _) = f32::as_simd::<S>(&temp[..lane_count]);
        simd_vec[0]
    }

    fn len(&self) -> usize {
        self.source.element_count()
    }

    fn shape(&self) -> &[usize] {
        &self.source.element_shape
    }

    fn is_contiguous(&self) -> bool {
        // Quantized data is not contiguous in the f32 sense
        false
    }
}

/// Matrix multiplication with a quantized RHS.
///
/// Computes `self @ rhs` where `self` is an f32 tensor and `rhs` is quantized.
/// This processes blocks one at a time to avoid the memory cost of full dequantization.
impl ConcreteTensor<f32, 2> {
    /// Matrix multiplication: self (M x K) @ rhs (K x N) -> (M x N)
    ///
    /// This is optimized for the case where the RHS (weights) are quantized.
    /// Instead of dequantizing the entire RHS matrix, it processes block-by-block
    /// with SIMD acceleration.
    pub fn q_mat_mul<B: GgufBlock + Sync>(
        &self,
        rhs: &QuantizedTensor<B, 2>,
    ) -> ConcreteTensor<f32, 2>
    where
        B::Dequantized: AsRef<[f32]>,
    {
        let lhs_shape = <Self as ResolvedTensor<2>>::shape(self);
        let m = lhs_shape[0];
        let k = lhs_shape[1];
        let rhs_shape = rhs.element_shape();
        let k2 = rhs_shape[0];
        let n = rhs_shape[1];

        assert_eq!(
            k, k2,
            "Matrix dimension mismatch: lhs columns ({}) != rhs rows ({})",
            k, k2
        );

        let mut output = ConcreteTensor::<f32, 2>::zeros([m, n]);
        let lhs_data = self.data();
        let out_data = output.data_mut();

        // Number of blocks per row in the RHS
        let blocks_per_row = n / B::BLOCK_SIZE;

        // Use SIMD-optimized inner loop
        pulp::Arch::new().dispatch(QMatmulSimd {
            lhs_data,
            rhs_blocks: rhs.blocks(),
            out_data,
            m,
            k,
            n,
            blocks_per_row,
            _phantom: std::marker::PhantomData::<B>,
        });

        output
    }
}

/// SIMD-accelerated quantized matmul kernel
struct QMatmulSimd<'a, B: GgufBlock> {
    lhs_data: &'a [f32],
    rhs_blocks: &'a [B],
    out_data: &'a mut [f32],
    m: usize,
    k: usize,
    n: usize,
    blocks_per_row: usize,
    _phantom: std::marker::PhantomData<B>,
}

impl<B: GgufBlock + Sync> pulp::WithSimd for QMatmulSimd<'_, B>
where
    B::Dequantized: AsRef<[f32]>,
{
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let Self {
            lhs_data,
            rhs_blocks,
            out_data,
            m,
            k,
            n,
            blocks_per_row,
            ..
        } = self;

        // Use parallel processing for larger matrices (threshold chosen to balance parallelism overhead)
        if m >= 8 {
            use rayon::prelude::*;

            // Process rows in parallel
            let out_chunks: Vec<&mut [f32]> = out_data.chunks_mut(n).collect();
            out_chunks
                .into_par_iter()
                .enumerate()
                .for_each(|(i, out_row)| {
                    let lhs_row = &lhs_data[i * k..(i + 1) * k];
                    process_row_simd::<B, S>(simd, lhs_row, rhs_blocks, out_row, k, blocks_per_row);
                });
        } else {
            // Sequential processing for small matrices
            for i in 0..m {
                let lhs_row = &lhs_data[i * k..(i + 1) * k];
                let out_row = &mut out_data[i * n..(i + 1) * n];
                process_row_simd::<B, S>(simd, lhs_row, rhs_blocks, out_row, k, blocks_per_row);
            }
        }
    }
}

/// Process a single output row with SIMD
#[inline(always)]
fn process_row_simd<B: GgufBlock, S: Simd>(
    simd: S,
    lhs_row: &[f32],
    rhs_blocks: &[B],
    out_row: &mut [f32],
    k: usize,
    blocks_per_row: usize,
) where
    B::Dequantized: AsRef<[f32]>,
{
    // For each block column in the output
    for block_col in 0..blocks_per_row {
        let out_col_start = block_col * B::BLOCK_SIZE;
        let out_block = &mut out_row[out_col_start..out_col_start + B::BLOCK_SIZE];

        // Split output block into SIMD-aligned parts
        let (out_simd, out_tail) = S::as_mut_simd_f32s(out_block);

        // Accumulate contributions from all k rows
        for rhs_row in 0..k {
            let lhs_val = lhs_row[rhs_row];
            let lhs_splat = simd.splat_f32s(lhs_val);

            let block_idx = rhs_row * blocks_per_row + block_col;
            let dequantized = rhs_blocks[block_idx].dequantize();
            let dequantized_slice = dequantized.as_ref();

            // SIMD part
            let (deq_simd, deq_tail) = S::as_simd_f32s(dequantized_slice);
            for (out_vec, &deq_vec) in out_simd.iter_mut().zip(deq_simd.iter()) {
                *out_vec = simd.mul_add_f32s(lhs_splat, deq_vec, *out_vec);
            }

            // Scalar tail
            for (out_val, &deq_val) in out_tail.iter_mut().zip(deq_tail.iter()) {
                *out_val += lhs_val * deq_val;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fusor_gguf::BlockQ8_0;
    use pulp::bytemuck;

    /// Helper to create a Q8_0 block from scale and data using bytemuck
    fn make_q8_0_block(scale: f32, data: [i8; 32]) -> BlockQ8_0 {
        // Q8_0 layout: scale (f16, 2 bytes) + data (32 i8, 32 bytes) = 34 bytes
        let mut bytes = [0u8; 34];
        let scale_f16 = half::f16::from_f32(scale);
        bytes[0..2].copy_from_slice(&scale_f16.to_le_bytes());
        bytes[2..34].copy_from_slice(bytemuck::cast_slice(&data));
        *bytemuck::from_bytes(&bytes)
    }

    #[test]
    fn test_quantized_tensor_creation() {
        // Create a simple Q8_0 tensor (block size 32)
        // Shape [2, 64] = 128 elements = 4 blocks
        let shape = [2, 64];
        let num_blocks = 4;

        // Create some test blocks
        let mut blocks_vec: AVec<BlockQ8_0> = AVec::with_capacity(64, num_blocks);
        for i in 0..num_blocks {
            let mut data = [0i8; 32];
            for j in 0..32 {
                data[j] = ((i * 32 + j) % 128) as i8;
            }
            blocks_vec.push(make_q8_0_block(1.0, data));
        }
        let blocks = blocks_vec.into_boxed_slice();

        let tensor = QuantizedTensor::from_blocks(shape, blocks);

        assert_eq!(tensor.element_shape(), &[2, 64]);
        assert_eq!(tensor.element_count(), 128);
        assert_eq!(tensor.block_count(), 4);
    }

    #[test]
    fn test_dequantize_q8_0() {
        // Q8_0: scale * i8 values
        let shape = [1, 32];
        let mut data = [0i8; 32];
        for i in 0..32 {
            data[i] = i as i8;
        }

        let mut blocks_vec: AVec<BlockQ8_0> = AVec::with_capacity(64, 1);
        blocks_vec.push(make_q8_0_block(0.5, data));
        let blocks = blocks_vec.into_boxed_slice();

        let quantized = QuantizedTensor::from_blocks(shape, blocks);
        let dequantized = quantized.dequantize();

        // Verify dequantization: each value should be scale * data[i] = 0.5 * i
        for i in 0..32 {
            let expected = 0.5 * (i as f32);
            let actual = dequantized.get([0, i]);
            assert!(
                (actual - expected).abs() < 1e-6,
                "Mismatch at index {}: expected {}, got {}",
                i,
                expected,
                actual
            );
        }
    }

    #[test]
    fn test_lazy_dequantize() {
        let shape = [1, 32];
        let mut data = [0i8; 32];
        for i in 0..32 {
            data[i] = (i as i8) - 16;
        }

        let mut blocks_vec: AVec<BlockQ8_0> = AVec::with_capacity(64, 1);
        blocks_vec.push(make_q8_0_block(2.0, data));
        let blocks = blocks_vec.into_boxed_slice();

        let quantized = QuantizedTensor::from_blocks(shape, blocks);
        let lazy = quantized.dequantize_lazy();

        // Test Expr trait methods
        assert_eq!(lazy.len(), 32);
        assert_eq!(lazy.shape(), &[1, 32]);
        assert!(!lazy.is_contiguous());

        // Test scalar evaluation
        for i in 0..32 {
            let expected = 2.0 * ((i as f32) - 16.0);
            let actual = lazy.eval_scalar(i);
            assert!(
                (actual - expected).abs() < 1e-6,
                "Mismatch at index {}: expected {}, got {}",
                i,
                expected,
                actual
            );
        }
    }

    #[test]
    fn test_matmul_quantized_simple() {
        // LHS: [2, 32] f32 tensor (all ones)
        // RHS: [32, 32] quantized (identity-like pattern)
        let lhs = ConcreteTensor::<f32, 2>::from_slice([2, 32], &vec![1.0f32; 64]);

        // Create a Q8_0 tensor where each block dequantizes to known values
        let shape = [32, 32];
        let num_blocks = 32; // 32 rows * 32 cols / 32 block_size = 32 blocks

        let mut blocks_vec: AVec<BlockQ8_0> = AVec::with_capacity(64, num_blocks);
        for _ in 0..num_blocks {
            let data = [1i8; 32]; // Each element is 1.0 after dequant
            blocks_vec.push(make_q8_0_block(1.0, data));
        }
        let blocks = blocks_vec.into_boxed_slice();
        let rhs = QuantizedTensor::from_blocks(shape, blocks);

        let result = lhs.q_mat_mul(&rhs);

        // Each output element should be sum of 32 ones = 32.0
        assert_eq!(<ConcreteTensor<f32, 2> as Expr>::shape(&result), &[2, 32]);
        for i in 0..2 {
            for j in 0..32 {
                let val = result.get([i, j]);
                assert!(
                    (val - 32.0).abs() < 1e-4,
                    "Mismatch at [{}, {}]: expected 32.0, got {}",
                    i,
                    j,
                    val
                );
            }
        }
    }

    #[test]
    fn test_matmul_quantized_vs_dequantize() {
        // Compare matmul_quantized with dequantize + regular matmul
        let lhs = ConcreteTensor::<f32, 2>::from_slice(
            [3, 64],
            &(0..192).map(|x| x as f32 * 0.1).collect::<Vec<_>>(),
        );

        // Create a quantized tensor
        let shape = [64, 64];
        let num_blocks = 64 * 64 / 32; // 128 blocks

        let mut blocks_vec: AVec<BlockQ8_0> = AVec::with_capacity(64, num_blocks);
        for i in 0..num_blocks {
            let mut data = [0i8; 32];
            for j in 0..32 {
                data[j] = ((i + j) % 100) as i8 - 50;
            }
            blocks_vec.push(make_q8_0_block(0.1, data));
        }
        let blocks = blocks_vec.into_boxed_slice();
        let rhs = QuantizedTensor::from_blocks(shape, blocks);

        // Compute using quantized matmul
        let result_quantized = lhs.q_mat_mul(&rhs);

        // Compute using dequantize + regular matmul
        let rhs_dequantized = rhs.dequantize();
        let result_dequantized = lhs.matmul_ref(&rhs_dequantized);

        // Results should match
        assert_eq!(
            <ConcreteTensor<f32, 2> as Expr>::shape(&result_quantized),
            <ConcreteTensor<f32, 2> as Expr>::shape(&result_dequantized)
        );
        for i in 0..3 {
            for j in 0..64 {
                let q_val = result_quantized.get([i, j]);
                let d_val = result_dequantized.get([i, j]);
                assert!(
                    (q_val - d_val).abs() < 1e-3,
                    "Mismatch at [{}, {}]: quantized={}, dequantized={}",
                    i,
                    j,
                    q_val,
                    d_val
                );
            }
        }
    }

    #[test]
    fn test_from_raw_bytes() {
        // Test creating a quantized tensor from raw bytes
        let shape = [1, 32];

        // Create raw bytes for one Q8_0 block: scale (f16) + data (32 i8)
        let mut bytes = vec![0u8; 34];
        let scale_f16 = half::f16::from_f32(1.0);
        bytes[0..2].copy_from_slice(&scale_f16.to_le_bytes());
        for i in 0..32 {
            bytes[2 + i] = i as u8;
        }

        let tensor = QuantizedTensor::<BlockQ8_0, 2>::from_raw_bytes(shape, &bytes);

        assert_eq!(tensor.element_shape(), &[1, 32]);
        assert_eq!(tensor.block_count(), 1);

        // Verify dequantization
        let dequantized = tensor.dequantize();
        for i in 0..32 {
            let expected = i as f32;
            let actual = dequantized.get([0, i]);
            assert!(
                (actual - expected).abs() < 1e-6,
                "Mismatch at index {}: expected {}, got {}",
                i,
                expected,
                actual
            );
        }
    }

    #[test]
    #[should_panic(expected = "Innermost dimension")]
    fn test_invalid_shape_not_multiple_of_block_size() {
        let shape = [2, 33]; // 33 is not a multiple of 32
        let blocks_vec: AVec<BlockQ8_0> = AVec::new(64);
        let blocks = blocks_vec.into_boxed_slice();
        let _ = QuantizedTensor::from_blocks(shape, blocks);
    }

    #[test]
    #[should_panic(expected = "Expected")]
    fn test_invalid_block_count() {
        let shape = [2, 64]; // Expects 4 blocks
        let mut blocks_vec: AVec<BlockQ8_0> = AVec::with_capacity(64, 2);
        for _ in 0..2 {
            blocks_vec.push(make_q8_0_block(1.0, [0i8; 32]));
        }
        let blocks = blocks_vec.into_boxed_slice();
        let _ = QuantizedTensor::from_blocks(shape, blocks);
    }
}
