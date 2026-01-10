use candle_core::{Device, Tensor as CandleTensor};
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use fusor_cpu::{Add, ConcreteTensor, ResolveTensor};

const SIZES: &[usize] = &[64, 256];

fn fusor_tensor_f32(size: usize) -> ConcreteTensor<f32, 1> {
    let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
    ConcreteTensor::from_slice([size], &data)
}

fn candle_tensor_f32(size: usize) -> CandleTensor {
    let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
    CandleTensor::from_vec(data, size, &Device::Cpu).unwrap()
}

fn bench_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("add_f32");

    for &size in SIZES {
        group.throughput(Throughput::Elements(size as u64));

        // Fusor
        group.bench_with_input(BenchmarkId::new("fusor", size), &size, |b, &size| {
            let lhs = fusor_tensor_f32(size);
            let rhs = fusor_tensor_f32(size);
            b.iter(|| {
                let op = Add::new(black_box(lhs.clone()), black_box(rhs.clone()));
                black_box(op.to_concrete())
            });
        });

        // Candle
        group.bench_with_input(BenchmarkId::new("candle", size), &size, |b, &size| {
            let lhs = candle_tensor_f32(size);
            let rhs = candle_tensor_f32(size);
            b.iter(|| black_box(black_box(&lhs).add(black_box(&rhs)).unwrap()));
        });
    }

    group.finish();
}

fn bench_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("creation_f32");

    for &size in SIZES {
        group.throughput(Throughput::Elements(size as u64));

        // Fusor zeros
        group.bench_with_input(BenchmarkId::new("fusor_zeros", size), &size, |b, &size| {
            b.iter(|| black_box(ConcreteTensor::<f32, 1>::zeros([size])));
        });

        // Candle zeros
        group.bench_with_input(BenchmarkId::new("candle_zeros", size), &size, |b, &size| {
            b.iter(|| {
                black_box(CandleTensor::zeros(size, candle_core::DType::F32, &Device::Cpu).unwrap())
            });
        });

        // Fusor from_slice
        group.bench_with_input(
            BenchmarkId::new("fusor_from_slice", size),
            &size,
            |b, &size| {
                let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
                b.iter(|| black_box(ConcreteTensor::from_slice([size], &data)));
            },
        );

        // Candle from_vec
        group.bench_with_input(
            BenchmarkId::new("candle_from_vec", size),
            &size,
            |b, &size| {
                let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
                b.iter(|| {
                    black_box(CandleTensor::from_vec(data.clone(), size, &Device::Cpu).unwrap())
                });
            },
        );
    }

    group.finish();
}

fn bench_add_2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("add_f32_2d");

    let matrix_sizes: &[(usize, usize)] = &[(32, 32), (64, 64), (128, 128), (256, 256), (512, 512)];

    for &(rows, cols) in matrix_sizes {
        let size = rows * cols;
        group.throughput(Throughput::Elements(size as u64));

        // Fusor
        group.bench_with_input(
            BenchmarkId::new("fusor", format!("{}x{}", rows, cols)),
            &(rows, cols),
            |b, &(rows, cols)| {
                let data: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.1).collect();
                let lhs: ConcreteTensor<f32, 2> = ConcreteTensor::from_slice([rows, cols], &data);
                let rhs: ConcreteTensor<f32, 2> = ConcreteTensor::from_slice([rows, cols], &data);
                b.iter(|| {
                    let op = Add::new(black_box(lhs.clone()), black_box(rhs.clone()));
                    black_box(op.to_concrete())
                });
            },
        );

        // Candle
        group.bench_with_input(
            BenchmarkId::new("candle", format!("{}x{}", rows, cols)),
            &(rows, cols),
            |b, &(rows, cols)| {
                let data: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.1).collect();
                let lhs = CandleTensor::from_vec(data.clone(), (rows, cols), &Device::Cpu).unwrap();
                let rhs = CandleTensor::from_vec(data, (rows, cols), &Device::Cpu).unwrap();
                b.iter(|| black_box(black_box(&lhs).add(black_box(&rhs)).unwrap()));
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_add, bench_creation, bench_add_2d,);
criterion_main!(benches);
