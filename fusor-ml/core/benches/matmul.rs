#![allow(unused)]
use std::time::Duration;

use candle_core::backend::BackendDevice;
use criterion::BatchSize;
use fusor_core::{Device, Tensor};
use futures::executor::block_on;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::{criterion_group, criterion_main};

use criterion::async_executor::FuturesExecutor;

const SIZES: [[usize; 2]; 8] = [
    [128, 128],
    [128, 1],
    [1024, 1024],
    [1024, 1],
    [2048, 2048],
    [2048, 1],
    [4096, 4096],
    [4096, 1],
];

fn matmul(c: &mut Criterion) {
    {
        let mut group = c.benchmark_group("matmul-wgpu");

        let device = block_on(Device::new()).unwrap();

        for [m, n] in SIZES {
            let device = device.clone();
            group.bench_with_input(
                BenchmarkId::new("matmul-wgpu", format!("{}x{}", m, n)),
                &(m, n),
                move |b, &(m, n)| {
                    let device = device.clone();
                    b.to_async(FuturesExecutor).iter_custom(async |iters| {
                        let mut sum = Duration::ZERO;
                        while sum.is_zero() {
                            for _ in 0..iters {
                                let tensor = Tensor::new(&device, &vec![vec![1.; n]; m]);
                                _ = tensor.as_slice().await.unwrap();
                                let new = tensor.mat_mul(&tensor.t());
                                let start = std::time::Instant::now();
                                new.materialize().await;
                                sum += start.elapsed();
                            }
                        }
                        sum
                    });
                },
            );
        }
    }

    {
        let mut group = c.benchmark_group("matmul-ndarray");
        let group = group.sample_size(20);

        for [m, n] in SIZES {
            group.bench_with_input(
                BenchmarkId::new("matmul-ndarray", format!("{}x{}", m, n)),
                &(m, n),
                move |b, &(m, n)| {
                    b.to_async(FuturesExecutor).iter_batched(
                        || {
                            let matrix = ndarray::Array2::<f32>::ones((m, n));
                            (matrix.clone(), matrix.clone())
                        },
                        |(tensor_a, tensor_b)| async move { tensor_a.dot(&tensor_b.t()) },
                        BatchSize::LargeInput,
                    );
                },
            );
        }
    }

    {
        let candle_device = candle_core::Device::Cpu;
        bench_candle_with_device(candle_device, "matmul-candle-cpu", c);
    }

    #[cfg(target_os = "macos")]
    {
        let candle_device = candle_core::Device::Metal(candle_core::MetalDevice::new(0).unwrap());
        bench_candle_with_device(candle_device, "matmul-candle-metal", c);
    }
}

fn bench_candle_with_device(candle_device: candle_core::Device, name: &str, c: &mut Criterion) {
    let mut group = c.benchmark_group(name);
    let group = group.sample_size(20);
    for [m, n] in SIZES {
        let candle_device = candle_device.clone();
        group.bench_with_input(
            BenchmarkId::new(name, format!("{}x{}", m, n)),
            &(m, n),
            move |b, &(m, n)| {
                b.to_async(FuturesExecutor).iter_batched(
                    {
                        let candle_device = candle_device.clone();
                        let random_data: Vec<Vec<f32>> = (0..m)
                            .map(|_| (0..n).map(|_| 1.).collect::<Vec<f32>>())
                            .collect();
                        move || {
                            candle_core::Tensor::from_iter(
                                random_data.iter().flat_map(|x| x.iter().copied()),
                                &candle_device,
                            )
                            .unwrap()
                            .reshape(&[m, n])
                            .unwrap()
                        }
                    },
                    {
                        let candle_device = candle_device.clone();
                        move |tensor| {
                            let candle_device = candle_device.clone();
                            async move {
                                let output = tensor.matmul(&tensor.t().unwrap()).unwrap();
                                candle_device.synchronize().unwrap();
                                output
                            }
                        }
                    },
                    BatchSize::LargeInput,
                );
            },
        );
    }
}

criterion_group!(benches, matmul);
criterion_main!(benches);
