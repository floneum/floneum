#![allow(unused)]
use std::sync::Arc;
use std::time::Duration;

use candle_core::backend::BackendDevice;
use criterion::BatchSize;
use fusor_core::Sum;
use fusor_core::{Device, Tensor};
use futures::executor::block_on;
use ndarray::Axis;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::{criterion_group, criterion_main};

use criterion::async_executor::FuturesExecutor;

const SIZES: [usize; 2] = [100, 1000];

fn bench_sum_reduce(c: &mut Criterion) {
    {
        let mut group = c.benchmark_group("sum-wgpu");
        let group = group.sample_size(20);
        for size in SIZES {
            let device = block_on(Device::new()).unwrap();

            group.bench_with_input(BenchmarkId::new("sum-wgpu", size), &size, move |b, &s| {
                let device = device.clone();
                b.to_async(FuturesExecutor).iter_custom(async |iters| {
                    let mut sum = Duration::ZERO;
                    while sum.is_zero() {
                        for _ in 0..iters {
                            let tensor = Tensor::new(&device, &vec![vec![1.; size]; size]);
                            _ = tensor.as_slice().await.unwrap();
                            let new = tensor.sum(0);
                            let start = std::time::Instant::now();
                            new.materialize().await;
                            sum += start.elapsed();
                        }
                    }
                    sum
                })
            });
        }
    }

    {
        let mut group = c.benchmark_group("sum-ndarray");
        let group = group.sample_size(20);
        for size in SIZES {
            group.bench_with_input(
                BenchmarkId::new("sum-ndarray", size),
                &size,
                move |b, &s| {
                    b.to_async(FuturesExecutor).iter_batched(
                        || ndarray::Array2::<f32>::ones((s, s)),
                        |tensor| async move { tensor.sum_axis(Axis(0)) },
                        BatchSize::LargeInput,
                    );
                },
            );
        }
    }
    
    {
        let candle_device = candle_core::Device::Cpu;
        bench_candle_with_device(
            candle_device,
            "sum-candle-cpu",
            c,
        );
    }

    #[cfg(target_os = "macos")]
    {
        let candle_device = candle_core::Device::Metal(candle_core::MetalDevice::new(0).unwrap());
        bench_candle_with_device(candle_device, "sum-candle-metal", c);
    }
}

fn bench_candle_with_device(candle_device: candle_core::Device, name: &str, c: &mut Criterion) {
    let mut group = c.benchmark_group(name);
    let group = group.sample_size(20);
    for size in SIZES {
        let candle_device = candle_device.clone();
        group.bench_with_input(
            BenchmarkId::new(name, size),
            &size,
            move |b, &s| {
                b.to_async(FuturesExecutor).iter_batched(
                    {
                        let candle_device = candle_device.clone();
                        let random_data: Vec<Vec<f32>> = (0..size)
                            .map(|_| (0..size).map(|_| 1.).collect::<Vec<f32>>())
                            .collect();
                        move || {
                            candle_core::Tensor::from_iter(
                                random_data.iter().flat_map(|x| x.iter().copied()),
                                &candle_device,
                            )
                            .unwrap()
                            .reshape(&[size, size])
                            .unwrap()
                        }
                    },
                    {
                        let candle_device = candle_device.clone();
                        move |tensor| {
                            let candle_device = candle_device.clone();
                            async move {
                                let output = tensor.sum(0).unwrap();
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

criterion_group!(benches, bench_sum_reduce);
criterion_main!(benches);
