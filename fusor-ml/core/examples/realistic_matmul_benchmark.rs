use fusor_core::*;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new().await?;

    println!("Realistic MatMul Performance Benchmark");
    println!("=====================================");

    // Test various matrix sizes to understand scaling behavior
    let sizes = vec![
        (32, 32, 32),
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
    ];

    for (m, k, n) in sizes {
        println!("\nTesting {}x{}x{} matrix multiplication:", m, k, n);

        // Create random test data
        let data_a: Vec<Vec<f32>> = (0..m)
            .map(|_| (0..k).map(|_| rand::random::<f32>()).collect())
            .collect();
        let data_b: Vec<Vec<f32>> = (0..k)
            .map(|_| (0..n).map(|_| rand::random::<f32>()).collect())
            .collect();

        let tensor_a = Tensor::new(&device, &data_a);
        let tensor_b = Tensor::new(&device, &data_b);

        // Warm up the GPU
        for _ in 0..3 {
            let _ = tensor_a.mat_mul(&tensor_b).as_slice().await?;
        }

        // Benchmark multiple runs
        let num_runs = 10;
        let mut times = Vec::new();

        for _ in 0..num_runs {
            let start = Instant::now();
            let result = tensor_a.mat_mul(&tensor_b);
            let _ = result.as_slice().await?; // Force execution
            let elapsed = start.elapsed();
            times.push(elapsed);
        }

        // Calculate statistics
        let avg_time = times.iter().sum::<std::time::Duration>() / num_runs as u32;
        let min_time = *times.iter().min().unwrap();
        let max_time = *times.iter().max().unwrap();

        // Calculate performance metrics
        let flops = (m as f64) * (k as f64) * (n as f64) * 2.0; // 2 ops per multiply-accumulate
        let gflops = flops / (avg_time.as_secs_f64() * 1e9);
        let memory_bytes = ((m * k) + (k * n) + (m * n)) * 4; // 4 bytes per f32
        let bandwidth_gb_s = memory_bytes as f64 / (avg_time.as_secs_f64() * 1e9);
        let arithmetic_intensity = flops / memory_bytes as f64;

        println!("  Average time: {:.3}ms", avg_time.as_secs_f64() * 1000.0);
        println!("  Min time: {:.3}ms", min_time.as_secs_f64() * 1000.0);
        println!("  Max time: {:.3}ms", max_time.as_secs_f64() * 1000.0);
        println!("  Performance: {:.1} GFLOPS", gflops);
        println!("  Memory bandwidth: {:.1} GB/s", bandwidth_gb_s);
        println!(
            "  Arithmetic intensity: {:.2} FLOPS/byte",
            arithmetic_intensity
        );

        // Performance analysis
        if arithmetic_intensity < 1.0 {
            println!("  → Memory bound (low arithmetic intensity)");
        } else if gflops < 100.0 {
            println!("  → Compute bound or low occupancy");
        } else {
            println!("  → Good performance balance");
        }
    }

    // Test different shapes to understand memory access patterns
    println!("\n\nTesting Different Matrix Shapes:");
    println!("================================");

    let shapes = vec![
        (1024, 64, 1024), // Tall-skinny A
        (64, 1024, 1024), // Wide-short A
        (1024, 1024, 64), // Skinny B
        (512, 512, 512),  // Square (baseline)
    ];

    for (m, k, n) in shapes {
        println!("\nShape {}x{}x{}:", m, k, n);

        let data_a: Vec<Vec<f32>> = (0..m)
            .map(|_| (0..k).map(|_| rand::random::<f32>()).collect())
            .collect();
        let data_b: Vec<Vec<f32>> = (0..k)
            .map(|_| (0..n).map(|_| rand::random::<f32>()).collect())
            .collect();

        let tensor_a = Tensor::new(&device, &data_a);
        let tensor_b = Tensor::new(&device, &data_b);

        // Single timing run
        let start = Instant::now();
        let result = tensor_a.mat_mul(&tensor_b);
        let _ = result.as_slice().await?;
        let elapsed = start.elapsed();

        let flops = (m as f64) * (k as f64) * (n as f64) * 2.0;
        let gflops = flops / (elapsed.as_secs_f64() * 1e9);
        let memory_bytes = ((m * k) + (k * n) + (m * n)) * 4;
        let arithmetic_intensity = flops / memory_bytes as f64;

        println!("  Time: {:.3}ms", elapsed.as_secs_f64() * 1000.0);
        println!("  Performance: {:.1} GFLOPS", gflops);
        println!("  Arithmetic intensity: {:.2}", arithmetic_intensity);
    }

    println!("\n\nBenchmark Complete!");
    println!("===================");
    println!("Key insights:");
    println!("1. Look for performance scaling with matrix size");
    println!("2. Identify whether workload is memory or compute bound");
    println!("3. Check if certain shapes perform better/worse");
    println!("4. Compare arithmetic intensity vs actual performance");

    Ok(())
}
