use fusor_core::matmul::{MatMulParams, sgemm::SgemmParams, sgemv::SgemvParams};
use fusor_core::{Device, Tensor};
use std::collections::HashMap;
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
struct BenchmarkResult {
    params: MatMulParams,
    operation_type: OperationType,
    matrix_size: MatrixSize,
    duration: Duration,
    runs: u32,
    error: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
enum OperationType {
    MatrixVector,
    MatrixMatrix,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct MatrixSize {
    m: usize,
    n: usize,
    k: usize,
}

impl MatrixSize {
    fn new(m: usize, n: usize, k: usize) -> Self {
        Self { m, n, k }
    }
}

struct ParameterTuner {
    device: Device,
    warmup_time: Duration,
    benchmark_time: Duration,
}

impl ParameterTuner {
    async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let device = Device::new().await?;
        Ok(Self {
            device,
            warmup_time: Duration::from_millis(100),
            benchmark_time: Duration::from_millis(500),
        })
    }

    fn generate_sgemv_params(&self) -> Vec<SgemvParams> {
        let mut params = Vec::new();

        // Test different chunk sizes
        let chunk_sizes = [1, 2, 4, 8, 16, 32, 64];
        // Test different vector sizes
        let vector_sizes = [1, 2, 4];

        for chunk_size in chunk_sizes {
            for vector_size in vector_sizes {
                params.push(SgemvParams::new(chunk_size, vector_size));
            }
        }

        params
    }

    fn generate_sgemm_params(&self) -> Vec<SgemmParams> {
        let mut params = Vec::new();

        // Test different thread sizes (must divide evenly into block sizes)
        let thread_sizes = [1, 2, 4];

        // Test different block sizes
        let block_m_multipliers = [2, 4, 8, 16, 32, 64];
        let block_n_multipliers = [2, 4, 8, 16, 32, 64];
        let block_k_sizes = [2, 4, 8, 16, 32, 64];

        // Test double buffering
        let double_buffer_options = [false, true];

        for &thread in thread_sizes.iter() {
            for &block_m_mult in block_m_multipliers.iter() {
                for &block_n_mult in block_n_multipliers.iter() {
                    for &block_k in block_k_sizes.iter() {
                        for &double_buffer in double_buffer_options.iter() {
                            let thread_m = thread;
                            let thread_n = thread;
                            let block_m = thread_m * block_m_mult;
                            let block_n = thread_n * block_n_mult;

                            // Ensure valid workgroup size constraints
                            if (block_m * block_n) % (thread_m * thread_n) != 0 {
                                continue;
                            }
                            let threads_per_workgroup = (block_m * block_n) / (thread_m * thread_n);
                            if threads_per_workgroup == 0
                                || threads_per_workgroup > 256
                                || threads_per_workgroup < block_k
                                || threads_per_workgroup < block_n
                                || threads_per_workgroup % block_k != 0
                                || threads_per_workgroup % block_n != 0
                            {
                                continue;
                            }
                            let threads_per_k_a: u32 = threads_per_workgroup / block_k;
                            let threads_per_n_b: u32 = threads_per_workgroup / block_n;
                            if block_m % threads_per_k_a != 0 || block_k % threads_per_n_b != 0 {
                                continue;
                            }
                            params.push(SgemmParams::new(
                                double_buffer,
                                block_m,
                                block_n,
                                block_k,
                                thread_m,
                                thread_n,
                            ));
                        }
                    }
                }
            }
        }

        params
    }

    fn get_test_sizes() -> Vec<MatrixSize> {
        let m_sized = [32, 64, 128, 256, 512, 1024, 2048];
        let n_sized = [1];
        let k_sized = [32, 64, 128, 256, 512, 1024, 2048];
        let mut sizes = Vec::new();
        for m in m_sized {
            for n in n_sized {
                for k in k_sized {
                    sizes.push(MatrixSize { m, n, k });
                }
            }
        }
        sizes
    }

    async fn benchmark_configuration(
        &self,
        params: MatMulParams,
        size: MatrixSize,
    ) -> BenchmarkResult {
        let operation_type = if size.n == 1 {
            OperationType::MatrixVector
        } else {
            OperationType::MatrixMatrix
        };

        // Create test data
        let matrix_a = self.create_test_matrix(size.m, size.k);
        let matrix_b = self.create_test_matrix(size.k, size.n);

        // Warmup runs
        let start_time = Instant::now();
        while start_time.elapsed() < self.warmup_time {
            if let Err(e) = self.run_single_matmul(&matrix_a, &matrix_b, &params).await {
                return BenchmarkResult {
                    params: params.clone(),
                    operation_type,
                    matrix_size: size,
                    duration: Duration::from_secs(0),
                    runs: 0,
                    error: Some(format!("Warmup failed: {}", e)),
                };
            }
        }

        // Benchmark runs
        let start_time = Instant::now();
        let mut total_duration = Duration::from_secs(0);
        let mut runs = 0;
        while start_time.elapsed() < self.benchmark_time {
            match self.run_single_matmul(&matrix_a, &matrix_b, &params).await {
                Ok(duration) => {
                    total_duration += duration;
                    runs += 1;
                }
                Err(e) => {
                    return BenchmarkResult {
                        params: params.clone(),
                        operation_type,
                        matrix_size: size,
                        duration: Duration::from_secs(0),
                        runs: 0,
                        error: Some(format!("Benchmark failed: {}", e)),
                    };
                }
            }
        }

        BenchmarkResult {
            params,
            operation_type,
            matrix_size: size,
            duration: total_duration,
            runs,
            error: None,
        }
    }

    fn create_test_matrix(&self, rows: usize, cols: usize) -> Tensor<2, f32> {
        let data: Vec<Vec<f32>> = (0..rows)
            .map(|i| (0..cols).map(|j| (i + j) as f32 * 0.01).collect())
            .collect();
        Tensor::new(&self.device, &data)
    }

    async fn run_single_matmul(
        &self,
        matrix_a: &Tensor<2, f32>,
        matrix_b: &Tensor<2, f32>,
        params: &MatMulParams,
    ) -> Result<Duration, Box<dyn std::error::Error>> {
        let start = Instant::now();
        let result = matrix_a.mat_mul_with_parameters(matrix_b, params.clone());
        result.materialize().await;
        Ok(start.elapsed())
    }

    async fn run_tuning(&self) -> Vec<BenchmarkResult> {
        let test_sizes = Self::get_test_sizes();
        let mut results = Vec::new();

        for size in test_sizes.iter() {
            println!("Testing matrix size: {}x{}x{}", size.m, size.n, size.k);

            if size.n == 1 {
                // Matrix-vector multiplication - test SGEMV parameters
                let sgemv_params = self.generate_sgemv_params();
                let total_params = sgemv_params.len();
                println!(
                    "  Testing {} SGEMV parameter configurations",
                    sgemv_params.len()
                );

                for (i, params) in sgemv_params.into_iter().enumerate() {
                    print!("\r  Progress: {}/{}", i + 1, total_params);
                    use std::io::{self, Write};
                    io::stdout().flush().unwrap();

                    let matmul_params = MatMulParams::Vector(params);
                    let result = self
                        .benchmark_configuration(matmul_params, size.clone())
                        .await;
                    results.push(result);
                }
            } else {
                // Matrix-matrix multiplication - test SGEMM parameters
                let sgemm_params = self.generate_sgemm_params();
                let total_params = sgemm_params.len();
                println!("  Testing {} SGEMM parameter configurations", total_params);

                for (i, params) in sgemm_params.into_iter().enumerate() {
                    print!("\r  Progress: {}/{}", i + 1, total_params);
                    use std::io::{self, Write};
                    io::stdout().flush().unwrap();

                    let matmul_params = MatMulParams::MatMul(params);
                    let result = self
                        .benchmark_configuration(matmul_params, size.clone())
                        .await;
                    results.push(result);
                }
            }
            println!(); // New line after progress
        }

        results
    }

    fn analyze_results(
        &self,
        results: Vec<BenchmarkResult>,
    ) -> HashMap<MatrixSize, BenchmarkResult> {
        let mut best_results: HashMap<MatrixSize, BenchmarkResult> = HashMap::new();

        for result in results {
            if result.error.is_some() {
                continue; // Skip failed results
            }

            let size = result.matrix_size.clone();
            let average_time = result.duration.div_f64(result.runs as f64);
            println!("  Analyzing result for matrix size: {:?}", size);
            println!("    Average Time: {:?}", average_time);
            println!("    Parameters: {:?}", result.params);
            println!();
            match best_results.get(&size) {
                None => {
                    best_results.insert(size, result);
                }
                Some(current_best) => {
                    if average_time < current_best.duration.div_f64(current_best.runs as f64) {
                        best_results.insert(size, result);
                    }
                }
            }
        }

        best_results
    }

    fn print_summary(&self, best_results: HashMap<MatrixSize, BenchmarkResult>) {
        println!("\n=== PARAMETER TUNING RESULTS SUMMARY ===\n");

        let mut sgemv_results: Vec<_> = best_results
            .iter()
            .filter(|(_, result)| result.operation_type == OperationType::MatrixVector)
            .collect();
        sgemv_results.sort_by_key(|(size, _)| (size.m, size.k));

        let mut sgemm_results: Vec<_> = best_results
            .iter()
            .filter(|(_, result)| result.operation_type == OperationType::MatrixMatrix)
            .collect();
        sgemm_results.sort_by_key(|(size, _)| (size.m, size.n, size.k));

        println!("SGEMV (Matrix-Vector) Results:");
        println!(
            "{:<15} {:<15} {:<15} {:>15}",
            "Matrix Size", "Chunk Size", "Vector Size", "Time (μs)"
        );
        println!("{:-<70}", "");

        for (size, result) in sgemv_results {
            if let MatMulParams::Vector(ref params) = result.params {
                println!(
                    "{:<15} {:<15} {:<15} {:>15.2}",
                    format!("{}x{}", size.m, size.k),
                    params.chunk_size(),
                    params.vector_size(),
                    result.duration.as_micros() as f64
                );
            }
        }

        println!("\nSGEMM (Matrix-Matrix) Results:");
        println!(
            "{:<15} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:>15}",
            "Matrix Size",
            "Block M",
            "Block N",
            "Block K",
            "Thread M",
            "Thread N",
            "Double Buf",
            "Time (μs)"
        );
        println!("{:-<120}", "");

        for (size, result) in sgemm_results {
            if let MatMulParams::MatMul(ref params) = result.params {
                println!(
                    "{:<15} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:>15.2}",
                    format!("{}x{}x{}", size.m, size.n, size.k),
                    params.block_m_size(),
                    params.block_n_size(),
                    params.block_k_size(),
                    params.thread_m_size(),
                    params.thread_n_size(),
                    params.double_buffer(),
                    result.duration.as_micros() as f64
                );
            }
        }
    }

    fn generate_optimized_code(&self, best_results: HashMap<MatrixSize, BenchmarkResult>) {
        println!("\n=== GENERATED OPTIMAL PARAMETER CONFIGURATION ===\n");

        println!("// Add this function to your matmul module to get optimized parameters");
        println!("pub fn get_optimal_params(m: usize, n: usize, k: usize) -> MatMulParams {{");
        println!("    match (m, n, k) {{");

        for (size, result) in best_results.iter() {
            match &result.params {
                MatMulParams::Vector(params) => {
                    println!(
                        "        ({}, 1, {}) => MatMulParams::Vector(SgemvParams::new({}, {})),",
                        size.m,
                        size.k,
                        params.chunk_size(),
                        params.vector_size()
                    );
                }
                MatMulParams::MatMul(params) => {
                    println!(
                        "        ({}, {}, {}) => MatMulParams::MatMul(SgemmParams::new({}, {}, {}, {}, {}, {})),",
                        size.m,
                        size.n,
                        size.k,
                        params.double_buffer(),
                        params.block_m_size(),
                        params.block_n_size(),
                        params.block_k_size(),
                        params.thread_m_size(),
                        params.thread_n_size()
                    );
                }
            }
        }

        println!("        // Default fallback");
        println!("        (_, 1, _) => MatMulParams::Vector(SgemvParams::default()),");
        println!("        (_, _, _) => MatMulParams::MatMul(SgemmParams::default()),");
        println!("    }}");
        println!("}}");
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Matrix Multiplication Parameter Tuning Tool");
    println!("===========================================\n");

    let tuner = ParameterTuner::new().await?;

    println!("Starting parameter tuning...");
    let results = tuner.run_tuning().await;

    println!("\nAnalyzing results...");
    let best_results = tuner.analyze_results(results);

    tuner.print_summary(best_results.clone());
    tuner.generate_optimized_code(best_results);

    println!("\nTuning complete!");

    Ok(())
}
