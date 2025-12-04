use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use fusor_core::{Device, Tensor};
use std::hint::black_box;

const SIZES: [[usize; 4]; 8] = [
    [1, 8, 128, 64],    // Small: 1 batch, 8 heads, 128 seq, 64 head_dim
    [1, 8, 256, 64],    // Medium sequence
    [1, 8, 512, 64],    // Large sequence
    [1, 8, 1024, 64],   // Very large sequence
    [2, 8, 128, 64],    // Batch of 2
    [4, 8, 128, 64],    // Batch of 4
    [1, 32, 128, 64],   // More heads
    [1, 8, 128, 128],   // Larger head dimension
];

fn bench_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention");

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

                        // Create random Q, K, V tensors
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

                        let q = Tensor::new(&device, &q_data);
                        let k = Tensor::new(&device, &k_data);
                        let v = Tensor::new(&device, &v_data);

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
    }

    group.finish();
}

criterion_group!(benches, bench_attention);
criterion_main!(benches);
