#![feature(portable_simd)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::simd::{Mask, Simd};
use tokenator::*;

pub fn keep_values_idx_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("keep");
    group.throughput(criterion::Throughput::Elements(8 * 100));
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
    group.throughput(criterion::Throughput::Elements(16 * 100));
    group.bench_function(BenchmarkId::new("keep_values_idx", 16), |b| {
        b.iter_batched(
            || {
                (0..100)
                    .map(|_| Mask::<i8, 16>::from_bitmask(rand::random::<u64>()))
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
    group.throughput(criterion::Throughput::Elements(32 * 100));
    group.bench_function(BenchmarkId::new("keep_values_idx", 32), |b| {
        b.iter_batched(
            || {
                (0..100)
                    .map(|_| Mask::<i8, 32>::from_bitmask(rand::random::<u64>()))
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
    group.throughput(criterion::Throughput::Elements(64 * 100));
    group.bench_function(BenchmarkId::new("keep_values_idx", 64), |b| {
        b.iter_batched(
            || {
                (0..100)
                    .map(|_| Mask::<i8, 64>::from_bitmask(rand::random::<u64>()))
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
    group.throughput(criterion::Throughput::Elements(8 * 100));
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
    group.throughput(criterion::Throughput::Elements(16 * 100));
    group.bench_function(BenchmarkId::new("swizzle_values_idx", 16), |b| {
        b.iter_batched(
            || {
                (0..100)
                    .map(|_| {
                        (
                            Mask::<i8, 16>::from_bitmask(rand::random::<u64>()),
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
    group.throughput(criterion::Throughput::Elements(32 * 100));
    group.bench_function(BenchmarkId::new("swizzle_values_idx", 32), |b| {
        b.iter_batched(
            || {
                (0..100)
                    .map(|_| {
                        (
                            Mask::<i8, 32>::from_bitmask(rand::random::<u64>()),
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
    group.throughput(criterion::Throughput::Elements(64 * 100));
    group.bench_function(BenchmarkId::new("swizzle_values_idx", 64), |b| {
        b.iter_batched(
            || {
                (0..100)
                    .map(|_| {
                        (
                            Mask::<i8, 64>::from_bitmask(rand::random::<u64>()),
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
}

pub fn swizzle_values_idx_bench_precomputed(c: &mut Criterion) {
    let mut group = c.benchmark_group("swizzle_precomputed");
    group.throughput(criterion::Throughput::Elements(8 * 100));
    group.bench_function(BenchmarkId::new("swizzle_values_idx_precomputed", 8), |b| {
        b.iter_batched(
            || {
                (0..100)
                    .map(|_| {
                        (
                            PreparedKeep::<8>::new(rand::random::<u64>().to_le_bytes()),
                            Simd::<_, 8>::from_array(std::array::from_fn(|_| rand::random::<u8>())),
                        )
                    })
                    .collect::<Vec<_>>()
            },
            |random| {
                for (mask, values) in random {
                    black_box(mask.swizzle_values(values));
                }
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.bench_function(BenchmarkId::new("swizzle_values_idx_computed", 8), |b| {
        b.iter_batched(
            || {
                (0..100)
                    .map(|_| {
                        (
                            rand::random::<u64>(),
                            Simd::<_, 8>::from_array(std::array::from_fn(|_| rand::random::<u8>())),
                        )
                    })
                    .collect::<Vec<_>>()
            },
            |random| {
                for (mask, values) in random {
                    let mask = PreparedKeep::<8>::new(mask.to_le_bytes());
                    black_box(mask.swizzle_values(values));
                }
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.throughput(criterion::Throughput::Elements(16 * 100));
    group.bench_function(
        BenchmarkId::new("swizzle_values_idx_precomputed", 16),
        |b| {
            b.iter_batched(
                || {
                    (0..100)
                        .map(|_| {
                            (
                                PreparedKeep::<16>::new(rand::random::<u64>().to_le_bytes()),
                                Simd::<_, 16>::from_array(std::array::from_fn(|_| {
                                    rand::random::<u8>()
                                })),
                            )
                        })
                        .collect::<Vec<_>>()
                },
                |random| {
                    for (mask, values) in random {
                        black_box(mask.swizzle_values(values));
                    }
                },
                criterion::BatchSize::SmallInput,
            );
        },
    );
    group.bench_function(BenchmarkId::new("swizzle_values_idx_computed", 16), |b| {
        b.iter_batched(
            || {
                (0..100)
                    .map(|_| {
                        (
                            rand::random::<u64>(),
                            Simd::<_, 16>::from_array(std::array::from_fn(|_| {
                                rand::random::<u8>()
                            })),
                        )
                    })
                    .collect::<Vec<_>>()
            },
            |random| {
                for (mask, values) in random {
                    let mask = PreparedKeep::<16>::new(mask.to_le_bytes());
                    black_box(mask.swizzle_values(values));
                }
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.throughput(criterion::Throughput::Elements(32 * 100));
    group.bench_function(
        BenchmarkId::new("swizzle_values_idx_precomputed", 32),
        |b| {
            b.iter_batched(
                || {
                    (0..100)
                        .map(|_| {
                            (
                                PreparedKeep::<32>::new(rand::random::<u64>().to_le_bytes()),
                                Simd::<_, 32>::from_array(std::array::from_fn(|_| {
                                    rand::random::<u8>()
                                })),
                            )
                        })
                        .collect::<Vec<_>>()
                },
                |random| {
                    for (mask, values) in random {
                        black_box(mask.swizzle_values(values));
                    }
                },
                criterion::BatchSize::SmallInput,
            );
        },
    );
    group.bench_function(BenchmarkId::new("swizzle_values_idx_computed", 32), |b| {
        b.iter_batched(
            || {
                (0..100)
                    .map(|_| {
                        (
                            rand::random::<u64>(),
                            Simd::<_, 32>::from_array(std::array::from_fn(|_| {
                                rand::random::<u8>()
                            })),
                        )
                    })
                    .collect::<Vec<_>>()
            },
            |random| {
                for (mask, values) in random {
                    let mask = PreparedKeep::<32>::new(mask.to_le_bytes());
                    black_box(mask.swizzle_values(values));
                }
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.throughput(criterion::Throughput::Elements(64 * 100));
    group.bench_function(
        BenchmarkId::new("swizzle_values_idx_precomputed", 64),
        |b| {
            b.iter_batched(
                || {
                    (0..100)
                        .map(|_| {
                            (
                                PreparedKeep::<64>::new(rand::random::<u64>().to_le_bytes()),
                                Simd::<_, 64>::from_array(std::array::from_fn(|_| {
                                    rand::random::<u8>()
                                })),
                            )
                        })
                        .collect::<Vec<_>>()
                },
                |random| {
                    for (mask, values) in random {
                        black_box(mask.swizzle_values(values));
                    }
                },
                criterion::BatchSize::SmallInput,
            );
        },
    );
    group.bench_function(BenchmarkId::new("swizzle_values_idx_computed", 64), |b| {
        b.iter_batched(
            || {
                (0..100)
                    .map(|_| {
                        (
                            rand::random::<u64>(),
                            Simd::<_, 64>::from_array(std::array::from_fn(|_| {
                                rand::random::<u8>()
                            })),
                        )
                    })
                    .collect::<Vec<_>>()
            },
            |random| {
                for (mask, values) in random {
                    let mask = PreparedKeep::<64>::new(mask.to_le_bytes());
                    black_box(mask.swizzle_values(values));
                }
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

criterion_group!(
    benches,
    keep_values_idx_bench,
    swizzle_values_idx_bench,
    swizzle_values_idx_bench_precomputed
);
criterion_main!(benches);
