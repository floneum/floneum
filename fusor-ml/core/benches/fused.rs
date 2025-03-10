#![allow(unused)]
use std::sync::Arc;
use std::time::Duration;

use criterion::{BatchSize, black_box};
use fusor_ml_core::PerformanceQueries;
use fusor_ml_core::{Device, Tensor};
use futures::executor::block_on;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::{criterion_group, criterion_main};

use criterion::async_executor::FuturesExecutor;

const SIZES: [usize; 2] = [100, 1000];

fn fused(c: &mut Criterion) {
    {
        let mut group = c.benchmark_group("add-const-fused-wgpu");
        let group = group.sample_size(20);
        for size in SIZES {
            let device = block_on(Device::new()).unwrap();
            std::thread::spawn({
                let device = device.clone();
                move || loop {
                    device.wgpu_device().poll(wgpu::PollType::Wait).unwrap();
                }
            });
            let tensor = Tensor::new(&device, &vec![vec![1.; size]; size]);
            block_on(tensor.as_slice()).unwrap();

            group.bench_with_input(
                BenchmarkId::new("add-const-fused-wgpu", size),
                &size,
                move |b, &s| {
                    let device = device.clone();
                    b.to_async(FuturesExecutor).iter_custom(async |iters| {
                        let mut sum = Duration::ZERO;
                        while sum.is_zero() {
                            for _ in 0..iters {
                                let tensor = Tensor::new(&device, &vec![vec![1.; size]; size]);
                                _ = tensor.as_slice().await.unwrap();
                                let new = (tensor + 1.) + 1.;
                                sum += new
                                    .all_timing_information()
                                    .await
                                    .iter()
                                    .map(|x| x.elapsed())
                                    .sum::<Duration>();
                            }
                        }
                        sum
                    })
                },
            );
        }
    }

    {
        let mut group = c.benchmark_group("add-const-separate-wgpu");
        let group = group.sample_size(20);
        for size in SIZES {
            let device = block_on(Device::new()).unwrap();
            std::thread::spawn({
                let device = device.clone();
                move || loop {
                    device.wgpu_device().poll(wgpu::PollType::Wait).unwrap();
                }
            });
            let tensor = Tensor::new(&device, &vec![vec![1.; size]; size]);
            block_on(tensor.as_slice()).unwrap();

            group.bench_with_input(
                BenchmarkId::new("add-const-separate-wgpu", size),
                &size,
                move |b, &s| {
                    let device = device.clone();
                    b.to_async(FuturesExecutor).iter_custom(async |iters| {
                        let mut sum = Duration::ZERO;
                        while sum.is_zero() {
                            for _ in 0..iters {
                                for _ in 0..2 {
                                    let tensor = Tensor::new(&device, &vec![vec![1.; size]; size]);
                                    _ = tensor.as_slice().await.unwrap();
                                    let new = tensor + 1.;
                                    sum += new
                                        .all_timing_information()
                                        .await
                                        .iter()
                                        .map(|x| x.elapsed())
                                        .sum::<Duration>();
                                }
                            }
                        }
                        sum
                    })
                },
            );
        }
    }
}

criterion_group!(benches, fused);
criterion_main!(benches);
