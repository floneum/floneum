#![allow(unused)]
use std::sync::Arc;
use std::time::Duration;

use criterion::{BatchSize, black_box};
use futures::executor::block_on;
use fusor_ml_core::{Device, Tensor};

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::{criterion_group, criterion_main};

use criterion::async_executor::FuturesExecutor;

const SIZES: [usize; 2] = [100, 1000];

fn bench_add_const(c: &mut Criterion) {
    {
        let mut group = c.benchmark_group("add-const-wgpu");
        let group = group.sample_size(20);
        for size in SIZES {
            let device = block_on(Device::new()).unwrap();
            std::thread::spawn({
                let device = device.clone();
                move || loop {
                    device.wgpu_device().poll(wgpu::PollType::Wait).unwrap();
                }
            });

            group.bench_with_input(
                BenchmarkId::new("add-const-wgpu", size),
                &size,
                move |b, &s| {
                    let device = device.clone();
                    b.to_async(FuturesExecutor).iter_custom(async |iters| {
                        let mut sum = Duration::ZERO;
                        while sum.is_zero() {
                            for _ in 0..iters {
                                let tensor = Tensor::new(&device, &vec![vec![1.; size]; size]);
                                _ = tensor.as_slice().await.unwrap();
                                let new = tensor + 1.;
                                let timing = new.all_timing_information().await;
                                sum += timing.iter().map(|x| x.elapsed()).sum::<Duration>();
                            }
                        }
                        sum
                    })
                },
            );
        }
    }

    {
        let mut group = c.benchmark_group("add-const-ndarray");
        let group = group.sample_size(20);
        for size in SIZES {
            group.bench_with_input(
                BenchmarkId::new("add-const-ndarray", size),
                &size,
                move |b, &s| {
                    b.to_async(FuturesExecutor).iter_batched(
                        || ndarray::Array2::<f32>::ones((s, s)),
                        |tensor| async move { tensor.map(|x| x + 1.) },
                        BatchSize::LargeInput,
                    );
                },
            );
        }
    }
}

criterion_group!(benches, bench_add_const);
criterion_main!(benches);
