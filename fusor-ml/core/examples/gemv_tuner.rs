use fusor_core::matmul::sgemv::SgemvParams;
use fusor_core::matmul::{MatMulParams, get_optimal_params};
use fusor_core::{Device, Tensor};
use std::time::{Duration, Instant};

async fn benchmark_params(
    device: &Device,
    m: usize,
    k: usize,
    params: &SgemvParams,
    iterations: u32,
) -> f64 {
    // Create matrices: A is m×k, B is k×1
    let a_data: Vec<Vec<f32>> = (0..m)
        .map(|i| (0..k).map(|j| (i + j) as f32 * 0.01).collect())
        .collect();
    let b_data: Vec<f32> = (0..k).map(|i| i as f32 * 0.01).collect();

    let tensor_a = Tensor::new(device, &a_data);
    let tensor_b = Tensor::new(device, &vec![b_data]);

    let matmul_params = MatMulParams::Vector(params.clone());

    // Warmup
    for _ in 0..10 {
        let result = tensor_a.mat_mul_with_parameters(&tensor_b.t(), matmul_params.clone());
        result.materialize().await;
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        let result = tensor_a.mat_mul_with_parameters(&tensor_b.t(), matmul_params.clone());
        result.materialize().await;
    }
    let duration = start.elapsed();

    // Calculate GFLOPS
    let ops_per_iter = 2.0 * m as f64 * k as f64; // multiply + add
    let total_ops = ops_per_iter * iterations as f64;
    total_ops / duration.as_secs_f64() / 1e9
}

async fn tune_shape(device: &Device, m: usize, k: usize) {
    println!("\n=== Tuning {}×{} ===", m, k);

    // Get current heuristic
    let heuristic = match get_optimal_params(m, 1, k) {
        MatMulParams::Vector(params) => params,
        MatMulParams::MatMul(_) => SgemvParams::default(),
    };

    println!(
        "Current: chunk={}, vector={}",
        heuristic.chunk_size(),
        heuristic.vector_size(),
    );

    // Test parameter combinations
    let test_params = (0..6).step_by(1).flat_map(|chunk_exp| {
        (0..3).map(move |vector_exp| {
            SgemvParams::new(
                1 << (chunk_exp),  // 16, 32
                1 << (vector_exp), // 2, 4, 8
            )
        })
    });

    let iterations = 250;
    let mut results = Vec::new();

    for params in test_params {
        let gflops = benchmark_params(device, m, k, &params, iterations).await;
        results.push((params, gflops));
        print!(".");
    }
    println!();

    // Sort by performance
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Show results
    println!("Top 5 results:");
    for (i, (params, gflops)) in results.iter().take(5).enumerate() {
        println!(
            "  {}. chunk={}, vector={} → {:e} GFLOPS",
            i + 1,
            params.chunk_size(),
            params.vector_size(),
            gflops
        );
    }

    // print the heuristic performance
    if let Some((_, heuristic_perf)) = results.iter().find(|(p, _)| {
        p.chunk_size() == heuristic.chunk_size() && p.vector_size() == heuristic.vector_size()
    }) {
        println!("Heuristic performance: {:e} GFLOPS", heuristic_perf);
    } else {
        println!("Heuristic parameters not found in tested configurations.");
    }

    // Compare with heuristic
    if let Some((_, heuristic_perf)) = results.iter().find(|(p, _)| {
        p.chunk_size() == heuristic.chunk_size() && p.vector_size() == heuristic.vector_size()
    }) {
        let best_perf = results[0].1;
        let improvement = (best_perf / heuristic_perf - 1.0) * 100.0;
        println!("🚀 Best is {:.1}% faster than heuristic", improvement);
    }
}

async fn run_tuning() {
    let device = Device::new().await.expect("Failed to create device");

    println!("GEMV Parameter Tuning");
    println!("====================");

    // Test key matrix shapes
    let sizes = [512, 1024, 2048, 4096];
    let shapes = (sizes.iter().copied())
        .flat_map(|m| sizes.iter().copied().map(move |k| (m, k)))
        // .filter(|(m, k)| m >= k)
    ;

    for (m, k) in shapes {
        tune_shape(&device, m, k).await;
    }

    println!("\n=== Summary ===");
    println!("Tuning complete! Check results above for optimization opportunities.");
}

#[tokio::main]
async fn main() {
    run_tuning().await;
}
