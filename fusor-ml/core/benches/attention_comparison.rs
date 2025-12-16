use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fusor_core::{Device, Tensor};
use std::hint::black_box;

// Comprehensive tensor sizes for thorough testing
const SIZES: [[usize; 4]; 8] = [
    [1, 8, 128, 64],  // Small: 1 batch, 8 heads, 128 seq, 64 head_dim
    [1, 8, 256, 64],  // Medium sequence
    [1, 8, 512, 64],  // Large sequence
    [1, 8, 1024, 64], // Very large sequence
    [2, 8, 128, 64],  // Batch of 2
    [4, 8, 128, 64],  // Batch of 4
    [1, 32, 128, 64], // More heads
    [1, 8, 128, 128], // Larger head dimension
];

/// Calculate theoretical FLOPs for attention computation
/// Standard attention: Q @ K^T + softmax + @ V
/// Flash attention: Same computation but memory-efficient
fn calculate_attention_flops(
    batch: usize,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
) -> f64 {
    let batch_heads = batch as f64 * num_heads as f64;
    let seq = seq_len as f64;
    let dim = head_dim as f64;

    // Q @ K^T: batch_heads * seq * seq * dim
    let qk_flops = batch_heads * seq * seq * dim * 2.0; // 2 ops per multiply-add

    // Softmax: batch_heads * seq * seq (exp + sum + divide)
    let softmax_flops = batch_heads * seq * seq * 3.0;

    // @ V: batch_heads * seq * seq * dim
    let attn_v_flops = batch_heads * seq * seq * dim * 2.0; // 2 ops per multiply-add

    qk_flops + softmax_flops + attn_v_flops
}

/// Calculate memory bandwidth requirements (bytes moved)
fn calculate_memory_bandwidth(
    batch: usize,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
) -> f64 {
    let elements = batch * num_heads * seq_len * head_dim;
    let bytes_per_element = std::mem::size_of::<f32>();

    // Input tensors: Q, K, V
    let input_bytes = elements * 3 * bytes_per_element;

    // Intermediate: attention matrix (batch * heads * seq * seq)
    let attn_matrix_bytes = batch * num_heads * seq_len * seq_len * bytes_per_element;

    // Output tensor
    let output_bytes = elements * bytes_per_element;

    (input_bytes + attn_matrix_bytes + output_bytes) as f64
}

async fn setup_tensors(
    device: &Device,
    batch: usize,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
) -> (Tensor<4, f32>, Tensor<4, f32>, Tensor<4, f32>) {
    let q_data: Vec<Vec<Vec<Vec<f32>>>> = (0..batch)
        .map(|_| {
            (0..num_heads)
                .map(|_| {
                    (0..seq_len)
                        .map(|_| {
                            (0..head_dim)
                                .map(|_| rand::random::<f32>() * 2.0 - 1.0)
                                .collect()
                        })
                        .collect()
                })
                .collect()
        })
        .collect();

    let k_data: Vec<Vec<Vec<Vec<f32>>>> = (0..batch)
        .map(|_| {
            (0..num_heads)
                .map(|_| {
                    (0..seq_len)
                        .map(|_| {
                            (0..head_dim)
                                .map(|_| rand::random::<f32>() * 2.0 - 1.0)
                                .collect()
                        })
                        .collect()
                })
                .collect()
        })
        .collect();

    let v_data: Vec<Vec<Vec<Vec<f32>>>> = (0..batch)
        .map(|_| {
            (0..num_heads)
                .map(|_| {
                    (0..seq_len)
                        .map(|_| {
                            (0..head_dim)
                                .map(|_| rand::random::<f32>() * 2.0 - 1.0)
                                .collect()
                        })
                        .collect()
                })
                .collect()
        })
        .collect();

    let q = Tensor::new(device, &q_data);
    let k = Tensor::new(device, &k_data);
    let v = Tensor::new(device, &v_data);

    // Ensure they are on GPU
    _ = q.as_slice().await.unwrap();
    _ = k.as_slice().await.unwrap();
    _ = v.as_slice().await.unwrap();

    (q, k, v)
}

fn bench_attention_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention_comparison");

    for &[batch, num_heads, seq_len, head_dim] in &SIZES {
        let size_str = format!("{}x{}x{}x{}", batch, num_heads, seq_len, head_dim);

        // Calculate performance metrics
        let theoretical_flops = calculate_attention_flops(batch, num_heads, seq_len, head_dim);
        let memory_bandwidth = calculate_memory_bandwidth(batch, num_heads, seq_len, head_dim);

        println!("\n=== Benchmark: {} ===", size_str);
        println!("Theoretical FLOPs: {:.2} GFLOPs", theoretical_flops / 1e9);
        println!("Memory bandwidth: {:.2} GB", memory_bandwidth / 1e9);

        // Benchmark standard attention (multiple kernels)
        group.bench_with_input(
            BenchmarkId::new("standard", &size_str),
            &(batch, num_heads, seq_len, head_dim),
            |b, &(batch, num_heads, seq_len, head_dim)| {
                b.to_async(criterion::async_executor::FuturesExecutor)
                    .iter_custom(|iters| async move {
                        let device = Device::new().await.unwrap();
                        let (q, k, v) =
                            setup_tensors(&device, batch, num_heads, seq_len, head_dim).await;
                        let scale = 1.0 / (head_dim as f32).sqrt();

                        let start = std::time::Instant::now();
                        for _ in 0..iters {
                            // Standard attention: Q @ K^T * scale -> softmax -> @ V
                            let scores = q.mat_mul(&k.t()) * scale;
                            let attn_weights = scores.softmax_last_dim();
                            let output = attn_weights.mat_mul(&v);
                            let _ = black_box(output.as_slice().await.unwrap());
                        }
                        let elapsed = start.elapsed();

                        // Calculate achieved performance
                        let total_flops = theoretical_flops * iters as f64;
                        let gflops_per_sec = total_flops / elapsed.as_secs_f64() / 1e9;
                        println!("Standard attention: {:.2} GFLOPs/sec", gflops_per_sec);

                        elapsed
                    });
            },
        );

        // Benchmark flash attention
        group.bench_with_input(
            BenchmarkId::new("flash", &size_str),
            &(batch, num_heads, seq_len, head_dim),
            |b, &(batch, num_heads, seq_len, head_dim)| {
                b.to_async(criterion::async_executor::FuturesExecutor)
                    .iter_custom(|iters| async move {
                        let device = Device::new().await.unwrap();
                        let (q, k, v) =
                            setup_tensors(&device, batch, num_heads, seq_len, head_dim).await;
                        let scale = 1.0 / (head_dim as f32).sqrt();

                        let start = std::time::Instant::now();
                        for _ in 0..iters {
                            let output = q.flash_attention(&k, &v, scale, None);
                            let _ = black_box(output.as_slice().await.unwrap());
                        }
                        let elapsed = start.elapsed();

                        // Calculate achieved performance
                        let total_flops = theoretical_flops * iters as f64;
                        let gflops_per_sec = total_flops / elapsed.as_secs_f64() / 1e9;
                        println!("Flash attention: {:.2} GFLOPs/sec", gflops_per_sec);

                        elapsed
                    });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_attention_comparison);
criterion_main!(benches);
