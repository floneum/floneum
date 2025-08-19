# Matrix Multiplication Parameter Tuning

This directory contains a parameter tuning tool for optimizing matrix multiplication performance at different sizes.

## Overview

The `tune_matmul_params` binary benchmarks different parameter configurations for both SGEMV (matrix-vector) and SGEMM (matrix-matrix) operations across various matrix sizes. It then generates optimal parameter configurations based on the benchmark results.

## Parameters Being Tuned

### SGEMV (Matrix-Vector Multiplication)
- **Chunk Size**: Number of elements processed per workgroup (1, 2, 4, 8)
- **Vector Size**: SIMD vector width for operations (1, 2, 4)

### SGEMM (Matrix-Matrix Multiplication)
- **Block M/N/K Size**: Tile sizes for shared memory blocking
- **Thread M/N Size**: Per-thread tile sizes
- **Double Buffer**: Whether to use double buffering for overlapped computation

## Usage

Run the tuning script:

```bash
cd fusor-ml/core
cargo run --bin tune_matmul_params --features tuning --release
```

The script will:

1. **Test Matrix Sizes**: Tests various matrix dimensions including:
   - Vector operations: 64x1x64 up to 1024x1x1024
   - Square matrices: 64x64x64 up to 1024x1024x1024  
   - Rectangular matrices: Various aspect ratios

2. **Benchmark Parameters**: For each size, tests multiple parameter combinations:
   - SGEMV: 12 different parameter combinations
   - SGEMM: ~324 different parameter combinations (varies based on constraints)

3. **Output Results**: Shows best parameters for each matrix size and generates optimized code.

## Sample Output

```
=== PARAMETER TUNING RESULTS SUMMARY ===

SGEMV (Matrix-Vector) Results:
Matrix Size     Chunk Size      Vector Size     Time (μs)
----------------------------------------------------------------
64x64          4               4               12.34
128x128        2               4               45.67
256x256        4               2               123.45

SGEMM (Matrix-Matrix) Results:
Matrix Size     Block M Block N Block K Thread M Thread N Double Buf    Time (μs)
------------------------------------------------------------------------------------------------------------------------
64x64x64       128     64      8       4        4        false           234.56
128x128x128    256     128     16      8        8        true            567.89

=== GENERATED OPTIMAL PARAMETER CONFIGURATION ===

// Add this function to your matmul module to get optimized parameters
pub fn get_optimal_params(m: usize, n: usize, k: usize) -> MatMulParams {
    match (m, n, k) {
        (64, 1, 64) => MatMulParams::Vector(SgemvParams::new(4, 4)),
        (64, 64, 64) => MatMulParams::MatMul(SgemmParams::new(false, 128, 64, 8, 4, 4)),
        // ... more configurations
        // Default fallback
        (_, 1, _) => MatMulParams::Vector(SgemvParams::default()),
        (_, _, _) => MatMulParams::MatMul(SgemmParams::default()),
    }
}
```

## Implementation Details

### Benchmark Methodology
- **Warmup**: 3 iterations to eliminate cold start effects
- **Measurement**: 5 iterations averaged for final timing
- **Error Handling**: Failed configurations are automatically excluded

### Parameter Generation
- **Constraint Validation**: Ensures workgroup sizes don't exceed GPU limits
- **Comprehensive Coverage**: Tests wide range of parameters while avoiding invalid combinations

### Matrix Sizes Tested
The script focuses on common use cases:
- Small vectors (64-1024 elements)
- Small to medium matrices (64x64 to 1024x1024)
- Rectangular matrices with various aspect ratios
- Transformer-style dimensions (4096x4096, etc.)

## Integration

To integrate the results into your code:

1. Copy the generated `get_optimal_params` function to your matmul module
2. Use it to select parameters based on matrix dimensions:

```rust
let optimal_params = get_optimal_params(m, n, k);
let result = matrix_a.mat_mul_with_parameters(&matrix_b, optimal_params);
```

## Notes

- Run on target hardware for best results since optimal parameters are hardware-dependent
- Results may vary based on GPU model, memory bandwidth, and compute capabilities
- Consider running multiple times and averaging results for production use
- The script automatically handles both small and large matrix sizes appropriately