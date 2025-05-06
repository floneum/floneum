#![allow(unused)]
use std::time::Duration;

use candle_core::MetalDevice;
use candle_core::backend::BackendDevice;
use candle_nn::Module;
use criterion::BatchSize;
use fusor_core::QMatrix;
use fusor_core::{Device, Tensor};
use futures::executor::block_on;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::{criterion_group, criterion_main};

use criterion::async_executor::FuturesExecutor;

fn qmatmul(c: &mut Criterion) {
    use crate::Device;
    use crate::Tensor;
    use candle_core::Module;
    use fusor_gguf::GgufMetadata;

    let url = "https://huggingface.co/unsloth/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q4_K_M.gguf";
    let bytes = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async move { reqwest::get(url).await.unwrap().bytes().await.unwrap() });

    for size in [576, 576 * 4] {
        let random_data: Vec<Vec<f32>> = (0..size)
            .map(|_| (0..576).map(|_| rand::random()).collect())
            .collect();

        {
            let mut group = c.benchmark_group("qmatmul-wgpu");

            let device = block_on(Device::new()).unwrap();

            let mut reader = std::io::Cursor::new(&bytes);
            let metadata = GgufMetadata::read(&mut reader).unwrap();
            let q_matrix_metadata = metadata.tensor_infos.get("blk.0.attn_q.weight").unwrap();

            let q_matrix = QMatrix::read(
                &device,
                q_matrix_metadata,
                &mut reader,
                metadata.tensor_data_offset,
            )
            .unwrap();

            let device = device.clone();
            let random_data = random_data.clone();
            group.bench_with_input(
                BenchmarkId::new("qmatmul-wgpu", size),
                &size,
                move |b, &s| {
                    let device = device.clone();
                    let random_data = random_data.clone();
                    b.to_async(FuturesExecutor).iter_custom(async |iters| {
                        let mut sum = Duration::ZERO;
                        while sum.is_zero() {
                            for _ in 0..iters {
                                let tensor = Tensor::new(&device, &random_data);
                                _ = tensor.as_slice().await.unwrap();

                                let new = tensor.q_mat_mul(&q_matrix);
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

        {
            let candle_device = candle_core::Device::Cpu;
            bench_candle_with_device(
                &bytes,
                size,
                random_data.clone(),
                candle_device,
                "qmatmul-candle-cpu",
                c,
            );
        }

        #[cfg(target_os = "macos")]
        {
            let candle_device = candle_core::Device::Metal(MetalDevice::new(0).unwrap());
            bench_candle_with_device(
                &bytes,
                size,
                random_data.clone(),
                candle_device,
                "qmatmul-candle-metal",
                c,
            );
        }
    }
}

fn bench_candle_with_device(
    bytes: &[u8],
    size: usize,
    random_data: Vec<Vec<f32>>,
    candle_device: candle_core::Device,
    name: &str,
    c: &mut Criterion,
) {
    let mut reader = std::io::Cursor::new(&bytes);
            let candle_metadata =
                candle_core::quantized::gguf_file::Content::read(&mut reader).unwrap();
            let candle_q_matrix_metadata = candle_metadata
                .tensor_infos
                .get("blk.0.attn_q.weight")
                .unwrap();
            let candle_q_tensor = candle_q_matrix_metadata
                .read(
                    &mut reader,
                    candle_metadata.tensor_data_offset,
                    &candle_device,
                )
                .unwrap();
            let candle_q_matrix =
                candle_core::quantized::QMatMul::from_qtensor(candle_q_tensor).unwrap();
            let mut group = c.benchmark_group(name);
            let group = group.sample_size(20);

            group.bench_with_input(
                BenchmarkId::new(name, size),
                &size,
                move |b, &s| {
                    b.to_async(FuturesExecutor).iter_batched(
                        || {
                            let candle_b = candle_core::Tensor::from_iter(
                                random_data.iter().flat_map(|x| x.iter().copied()),
                                &candle_device,
                            )
                            .unwrap()
                            .reshape(&[size, 576])
                            .unwrap();
                            candle_device.synchronize().unwrap();
                            (
                                candle_b.clone(),
                                candle_q_matrix.clone(),
                                candle_device.clone(),
                            )
                        },
                        |(tensor_a, tensor_b, candle_device)| async move {
                            tensor_b.forward(&tensor_a).unwrap();
                            candle_device.synchronize().unwrap();
                        },
                        BatchSize::LargeInput,
                    );
                },
            );
}

criterion_group!(benches, qmatmul);
criterion_main!(benches);
