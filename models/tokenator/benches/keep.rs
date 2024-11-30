#![feature(portable_simd)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::simd::{Mask, Simd};
use tokenator::*;

pub fn keep_values_idx_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("keep");
    group.throughput(criterion::Throughput::Elements(8));
    group.bench_function(BenchmarkId::new("keep_values_idx", 8), |b| {
        b.iter_batched(
            || {
                (0..100)
                    .map(|_| Mask::<i8, 8>::from_bitmask(rand::random::<u64>()))
                    .collect::<Vec<_>>()
            },
            |random| {
                for mask in random {
                    black_box(keep_values_idx(mask));
                }
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.throughput(criterion::Throughput::Elements(16));
    group.bench_function(BenchmarkId::new("keep_values_idx", 16), |b| {
        b.iter_batched(
            || {
                (0..100)
                    .map(|_| Mask::<i16, 16>::from_bitmask(rand::random::<u64>()))
                    .collect::<Vec<_>>()
            },
            |random| {
                for mask in random {
                    black_box(keep_values_idx(mask));
                }
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.throughput(criterion::Throughput::Elements(32));
    group.bench_function(BenchmarkId::new("keep_values_idx", 32), |b| {
        b.iter_batched(
            || {
                (0..100)
                    .map(|_| Mask::<i32, 32>::from_bitmask(rand::random::<u64>()))
                    .collect::<Vec<_>>()
            },
            |random| {
                for mask in random {
                    black_box(keep_values_idx(mask));
                }
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.throughput(criterion::Throughput::Elements(64));
    group.bench_function(BenchmarkId::new("keep_values_idx", 64), |b| {
        b.iter_batched(
            || {
                (0..100)
                    .map(|_| Mask::<i64, 64>::from_bitmask(rand::random::<u64>()))
                    .collect::<Vec<_>>()
            },
            |random| {
                for mask in random {
                    black_box(keep_values_idx(mask));
                }
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

pub fn swizzle_values_idx_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("swizzle");
    group.throughput(criterion::Throughput::Elements(8));
    group.bench_function(BenchmarkId::new("swizzle_values_idx", 8), |b| {
        b.iter_batched(
            || {
                (0..100)
                    .map(|_| {
                        (
                            Mask::<i8, 8>::from_bitmask(rand::random::<u64>()),
                            Simd::from_array(std::array::from_fn(|_| rand::random::<u8>())),
                        )
                    })
                    .collect::<Vec<_>>()
            },
            |random| {
                for (mask, values) in random {
                    black_box(keep_values(mask, values));
                }
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.throughput(criterion::Throughput::Elements(16));
    group.bench_function(BenchmarkId::new("swizzle_values_idx", 16), |b| {
        b.iter_batched(
            || {
                (0..100)
                    .map(|_| {
                        (
                            Mask::<i16, 16>::from_bitmask(rand::random::<u64>()),
                            Simd::from_array(std::array::from_fn(|_| rand::random::<u16>())),
                        )
                    })
                    .collect::<Vec<_>>()
            },
            |random| {
                for (mask, values) in random {
                    black_box(keep_values(mask, values));
                }
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.throughput(criterion::Throughput::Elements(32));
    group.bench_function(BenchmarkId::new("swizzle_values_idx", 32), |b| {
        b.iter_batched(
            || {
                (0..100)
                    .map(|_| {
                        (
                            Mask::<i32, 32>::from_bitmask(rand::random::<u64>()),
                            Simd::from_array(std::array::from_fn(|_| rand::random::<u32>())),
                        )
                    })
                    .collect::<Vec<_>>()
            },
            |random| {
                for (mask, values) in random {
                    black_box(keep_values(mask, values));
                }
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.throughput(criterion::Throughput::Elements(64));
    group.bench_function(BenchmarkId::new("swizzle_values_idx", 64), |b| {
        b.iter_batched(
            || {
                (0..100)
                    .map(|_| {
                        (
                            Mask::<i64, 64>::from_bitmask(rand::random::<u64>()),
                            Simd::from_array(std::array::from_fn(|_| rand::random::<u64>())),
                        )
                    })
                    .collect::<Vec<_>>()
            },
            |random| {
                for (mask, values) in random {
                    black_box(keep_values(mask, values));
                }
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

criterion_group!(benches, keep_values_idx_bench, swizzle_values_idx_bench);
criterion_main!(benches);
