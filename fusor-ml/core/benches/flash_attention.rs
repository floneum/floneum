use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use fusor_core::{Device, Tensor};
use std::hint::black_box;

const SIZES: [[usize; 4]; 4] = [
    [1, 8, 128, 64],    // Small
    [1, 8, 256, 64],    // Medium
    [1, 8, 512, 64],    // Large
    [1, 32, 128, 64],   // More heads
];

async fn setup_tensors(device: &Device, batch: usize, num_heads: usize, seq_len: usize, head_dim: usize) -> (Tensor<4, f32>, Tensor<4, f32>, Tensor<4, f32>) {
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

fn bench_flash_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("flash_attention");

    for &[batch, num_heads, seq_len, head_dim] in &SIZES {
        let size_str = format!("{}x{}x{}x{}", batch, num_heads, seq_len, head_dim);

        // Benchmark standard attention (multiple kernels)
        group.bench_with_input(
            BenchmarkId::new("standard", &size_str),
            &(batch, num_heads, seq_len, head_dim),
            |b, &(batch, num_heads, seq_len, head_dim)| {
                b.to_async(criterion::async_executor::FuturesExecutor)
                    .iter_custom(|iters| async move {
                        let device = Device::new().await.unwrap();
                        let (q, k, v) = setup_tensors(&device, batch, num_heads, seq_len, head_dim).await;
                        let scale = 1.0 / (head_dim as f32).sqrt();

                        let start = std::time::Instant::now();
                        for _ in 0..iters {
                            // Standard attention: Q @ K^T * scale -> softmax -> @ V
                            let scores = q.mat_mul(&k.t()) * scale;
                            let attn_weights = scores.softmax_last_dim();
                            let output = attn_weights.mat_mul(&v);
                            let _ = black_box(output.as_slice().await.unwrap());
                        }
                        start.elapsed()
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
                        let (q, k, v) = setup_tensors(&device, batch, num_heads, seq_len, head_dim).await;
                        let scale = 1.0 / (head_dim as f32).sqrt();

                        let start = std::time::Instant::now();
                        for _ in 0..iters {
                            let output = q.flash_attention(&k, &v, scale);
                            let _ = black_box(output.as_slice().await.unwrap());
                        }
                        start.elapsed()
                    });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_flash_attention);
criterion_main!(benches);