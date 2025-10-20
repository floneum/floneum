#![allow(unused)]
use std::time::Duration;

use candle_core::MetalDevice;
use candle_core::backend::BackendDevice;
use candle_nn::{Module, VarBuilder};
use criterion::BatchSize;
use fusor_core::QMatrix;
use fusor_core::layers::Linear;
use fusor_core::{Device, Tensor};
use futures::executor::block_on;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::{criterion_group, criterion_main};

use criterion::async_executor::FuturesExecutor;
use kalosm_common::Cache;
use kalosm_model_types::FileSource;

// Benchmark LayerNorm operation
fn layer_norm(c: &mut Criterion) {
    use crate::Device;
    use crate::Tensor;

    let source = FileSource::HuggingFace {
        model_id: "CompendiumLabs/bge-large-en-v1.5-gguf".to_string(),
        revision: "main".to_string(),
        file: "bge-large-en-v1.5-q4_k_m.gguf".to_string(),
    };
    let bytes = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async move {
            let cache = Cache::default();
            let path = cache.get(&source, |_| {}).await.unwrap();
            tokio::fs::read(&path).await.unwrap()
        });

    for batch_size in [1, 32, 512] {
        for seq_len in [13, 128, 512] {
            if batch_size * seq_len >= 512 * 128 {
                // Skip too large combinations for LayerNorm
                continue;
            }
            let hidden_size = 1024;
            let random_data: Vec<Vec<Vec<f32>>> = (0..batch_size)
                .map(|_| {
                    (0..seq_len)
                        .map(|_| (0..hidden_size).map(|_| rand::random()).collect())
                        .collect()
                })
                .collect();

            // Fusor LayerNorm benchmark
            {
                let mut reader = std::io::Cursor::new(&bytes);
                let mut var_builder = fusor_core::VarBuilder::from_gguf(&mut reader).unwrap();
                let device = block_on(Device::new()).unwrap();

                // Load layer norm weights from the model
                let weight: Tensor<1, f32> = var_builder
                    .pp("blk.0.attn_output_norm")
                    .get("weight", &device)
                    .unwrap()
                    .dequantize();
                let bias: Option<Tensor<1, f32>> = var_builder
                    .pp("blk.0.attn_output_norm")
                    .get("bias", &device)
                    .ok()
                    .map(|b| b.dequantize());

                let mut group =
                    c.benchmark_group(format!("layer_norm-fusor-{batch_size}x{seq_len}"));

                let device = device.clone();
                let random_data = random_data.clone();
                group.bench_with_input(
                    BenchmarkId::new("layer_norm-fusor", format!("{batch_size}x{seq_len}")),
                    &(batch_size, seq_len),
                    move |b, &(batch_size, seq_len)| {
                        let device = device.clone();
                        let random_data = random_data.clone();
                        b.to_async(FuturesExecutor).iter_custom(async |iters| {
                            let tensor = Tensor::new(&device, &random_data);
                            tensor.materialize().await;
                            let mut sum = Duration::ZERO;
                            while sum.is_zero() {
                                for _ in 0..iters {
                                    let start = std::time::Instant::now();
                                    let normalized =
                                        tensor.layer_norm(&weight, bias.as_ref(), 1e-12, true);
                                    normalized.materialize().await;
                                    sum += start.elapsed();
                                }
                            }
                            sum
                        });
                    },
                );
            }

            // Candle LayerNorm benchmark
            {
                let candle_device = candle_core::Device::Cpu;
                bench_candle_layer_norm(
                    &bytes,
                    batch_size,
                    seq_len,
                    1024,
                    random_data.clone(),
                    candle_device,
                    "layer_norm-candle-cpu",
                    c,
                );
            }

            // Candle LayerNorm benchmark on Metal (macOS only)
            #[cfg(target_os = "macos")]
            {
                let candle_device = candle_core::Device::Metal(MetalDevice::new(0).unwrap());
                bench_candle_layer_norm(
                    &bytes,
                    batch_size,
                    seq_len,
                    1024,
                    random_data.clone(),
                    candle_device,
                    "layer_norm-candle-metal",
                    c,
                );
            }
        }
    }
}

fn bench_candle_layer_norm(
    bytes: &[u8],
    batch_size: usize,
    seq_len: usize,
    hidden_size: usize,
    random_data: Vec<Vec<Vec<f32>>>,
    candle_device: candle_core::Device,
    name: &str,
    c: &mut Criterion,
) {
    use candle_nn::LayerNorm;

    let var_builder = candle_transformers::quantized_var_builder::VarBuilder::from_gguf_buffer(
        bytes,
        &candle_device,
    )
    .unwrap()
    .pp("blk.0.attn_output_norm");

    let weight = var_builder
        .get_no_shape("weight")
        .unwrap()
        .dequantize(&candle_device)
        .unwrap();
    let bias = var_builder
        .get_no_shape("bias")
        .unwrap()
        .dequantize(&candle_device)
        .unwrap();

    let layer_norm = candle_nn::LayerNorm::new(weight, bias, 1e-12);

    let mut group = c.benchmark_group(format!("{name}-{batch_size}x{seq_len}"));
    group.sample_size(20);

    group.bench_with_input(
        BenchmarkId::new(name, format!("{batch_size}x{seq_len}")),
        &(batch_size, seq_len),
        move |b, &(batch_size, seq_len)| {
            b.to_async(FuturesExecutor).iter_batched(
                || {
                    let candle_tensor = candle_core::Tensor::from_iter(
                        random_data
                            .iter()
                            .flat_map(|b| b.iter().flat_map(|s| s.iter().copied())),
                        &candle_device,
                    )
                    .unwrap()
                    .reshape(&[batch_size, seq_len, hidden_size])
                    .unwrap();
                    candle_device.synchronize().unwrap();
                    (candle_tensor, layer_norm.clone(), candle_device.clone())
                },
                |(tensor, layer_norm, candle_device)| async move {
                    layer_norm.forward(&tensor).unwrap();
                    candle_device.synchronize().unwrap();
                },
                BatchSize::LargeInput,
            );
        },
    );
}

// Benchmark Self-Attention operation
fn self_attention(c: &mut Criterion) {
    use crate::Device;
    use crate::Tensor;

    let source = FileSource::HuggingFace {
        model_id: "CompendiumLabs/bge-large-en-v1.5-gguf".to_string(),
        revision: "main".to_string(),
        file: "bge-large-en-v1.5-q4_k_m.gguf".to_string(),
    };
    let bytes = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async move {
            let cache = Cache::default();
            let path = cache.get(&source, |_| {}).await.unwrap();
            tokio::fs::read(&path).await.unwrap()
        });

    for batch_size in [1, 32] {
        for seq_len in [13, 128] {
            if batch_size * seq_len >= 32 * 128 {
                continue;
            }
            let hidden_size = 1024;
            let num_heads = 16;
            let head_size = hidden_size / num_heads;

            let random_data: Vec<Vec<Vec<f32>>> = (0..batch_size)
                .map(|_| {
                    (0..seq_len)
                        .map(|_| (0..hidden_size).map(|_| rand::random()).collect())
                        .collect()
                })
                .collect();

            // Fusor Self-Attention benchmark
            {
                let mut reader = std::io::Cursor::new(&bytes);
                let mut var_builder = fusor_core::VarBuilder::from_gguf(&mut reader).unwrap();
                let device = block_on(Device::new()).unwrap();

                // Load Q, K, V weights from the model
                let query = Linear::load(&device, &mut var_builder.pp("blk.0.attn_q")).unwrap();
                let key = Linear::load(&device, &mut var_builder.pp("blk.0.attn_k")).unwrap();
                let value = Linear::load(&device, &mut var_builder.pp("blk.0.attn_v")).unwrap();

                let mut group =
                    c.benchmark_group(format!("self_attention-fusor-{batch_size}x{seq_len}"));

                let device = device.clone();
                let random_data = random_data.clone();
                group.bench_with_input(
                    BenchmarkId::new("self_attention-fusor", format!("{batch_size}x{seq_len}")),
                    &(batch_size, seq_len),
                    move |b, &(batch_size, seq_len)| {
                        let device = device.clone();
                        let random_data = random_data.clone();

                        b.to_async(FuturesExecutor).iter_custom(async |iters| {
                            let tensor = Tensor::new(&device, &random_data);
                            tensor.materialize().await;
                            let mut sum = Duration::ZERO;
                            while sum.is_zero() {
                                for _ in 0..iters {
                                    let start = std::time::Instant::now();

                                    // Q, K, V projections
                                    let q = query.forward(&tensor);
                                    let k = key.forward(&tensor);
                                    let v = value.forward(&tensor);

                                    // Transpose for multi-head attention
                                    let q = q
                                        .reshape([batch_size, seq_len, num_heads, head_size])
                                        .transpose(1, 2);
                                    let k = k
                                        .reshape([batch_size, seq_len, num_heads, head_size])
                                        .transpose(1, 2);
                                    let v = v
                                        .reshape([batch_size, seq_len, num_heads, head_size])
                                        .transpose(1, 2);

                                    // Attention scores
                                    let scores = q.mat_mul(&k.t()) / (head_size as f32).sqrt();
                                    let probs = scores.softmax_last_dim();

                                    // Context layer
                                    let context = probs.mat_mul(&v);
                                    let context = context.transpose(1, 2);
                                    let output = context.flatten_last_n::<1, _>();

                                    output.materialize().await;
                                    sum += start.elapsed();
                                }
                            }
                            sum
                        });
                    },
                );
            }

            // Candle Self-Attention benchmark
            #[cfg(target_os = "macos")]
            {
                let candle_device = candle_core::Device::Metal(MetalDevice::new(0).unwrap());
                bench_candle_self_attention(
                    &bytes,
                    batch_size,
                    seq_len,
                    hidden_size,
                    num_heads,
                    random_data.clone(),
                    candle_device,
                    "self_attention-candle-metal",
                    c,
                );
            }

            {
                let candle_device = candle_core::Device::Cpu;
                bench_candle_self_attention(
                    &bytes,
                    batch_size,
                    seq_len,
                    hidden_size,
                    num_heads,
                    random_data.clone(),
                    candle_device,
                    "self_attention-candle-cpu",
                    c,
                );
            }
        }
    }
}

fn bench_candle_self_attention(
    bytes: &[u8],
    batch_size: usize,
    seq_len: usize,
    hidden_size: usize,
    num_heads: usize,
    random_data: Vec<Vec<Vec<f32>>>,
    candle_device: candle_core::Device,
    name: &str,
    c: &mut Criterion,
) {
    use candle_transformers::quantized_nn::Linear;

    let var_builder = candle_transformers::quantized_var_builder::VarBuilder::from_gguf_buffer(
        bytes,
        &candle_device,
    )
    .unwrap();

    let q_weight = var_builder
        .pp("blk.0.attn_q")
        .get_no_shape("weight")
        .unwrap();
    let q_bias = var_builder
        .pp("blk.0.attn_q")
        .get_no_shape("bias")
        .unwrap()
        .dequantize(&candle_device)
        .unwrap();
    let k_weight = var_builder
        .pp("blk.0.attn_k")
        .get_no_shape("weight")
        .unwrap();
    let k_bias = var_builder
        .pp("blk.0.attn_k")
        .get_no_shape("bias")
        .unwrap()
        .dequantize(&candle_device)
        .unwrap();
    let v_weight = var_builder
        .pp("blk.0.attn_v")
        .get_no_shape("weight")
        .unwrap();
    let v_bias = var_builder
        .pp("blk.0.attn_v")
        .get_no_shape("bias")
        .unwrap()
        .dequantize(&candle_device)
        .unwrap();

    let q_linear = Linear::from_arc(q_weight, Some(q_bias)).unwrap();
    let k_linear = Linear::from_arc(k_weight, Some(k_bias)).unwrap();
    let v_linear = Linear::from_arc(v_weight, Some(v_bias)).unwrap();

    let head_size = hidden_size / num_heads;

    let mut group = c.benchmark_group(format!("{name}-{batch_size}x{seq_len}"));
    group.sample_size(20);

    group.bench_with_input(
        BenchmarkId::new(name, format!("{batch_size}x{seq_len}")),
        &(batch_size, seq_len),
        move |b, &(batch_size, seq_len)| {
            b.to_async(FuturesExecutor).iter_batched(
                || {
                    let candle_tensor = candle_core::Tensor::from_iter(
                        random_data
                            .iter()
                            .flat_map(|b| b.iter().flat_map(|s| s.iter().copied())),
                        &candle_device,
                    )
                    .unwrap()
                    .reshape(&[batch_size, seq_len, hidden_size])
                    .unwrap();
                    candle_device.synchronize().unwrap();
                    (
                        candle_tensor,
                        q_linear.clone(),
                        k_linear.clone(),
                        v_linear.clone(),
                        candle_device.clone(),
                    )
                },
                |(tensor, q_linear, k_linear, v_linear, candle_device)| async move {
                    // Q, K, V projections
                    let q = q_linear.forward(&tensor).unwrap();
                    let k = k_linear.forward(&tensor).unwrap();
                    let v = v_linear.forward(&tensor).unwrap();

                    // Reshape for multi-head attention
                    let q = q
                        .reshape(&[batch_size, seq_len, num_heads, head_size])
                        .unwrap()
                        .transpose(1, 2)
                        .unwrap()
                        .contiguous()
                        .unwrap();
                    let k = k
                        .reshape(&[batch_size, seq_len, num_heads, head_size])
                        .unwrap()
                        .transpose(1, 2)
                        .unwrap()
                        .contiguous()
                        .unwrap();
                    let v = v
                        .reshape(&[batch_size, seq_len, num_heads, head_size])
                        .unwrap()
                        .transpose(1, 2)
                        .unwrap()
                        .contiguous()
                        .unwrap();

                    // Attention scores
                    // k is [batch, heads, seq, head_dim], transpose to [batch, heads, head_dim, seq]
                    let k_t = k.transpose(2, 3).unwrap().contiguous().unwrap();
                    let scores = q.matmul(&k_t).unwrap();
                    let scores = (scores / (head_size as f64).sqrt()).unwrap();
                    let probs = candle_nn::ops::softmax_last_dim(&scores).unwrap();

                    // Context layer
                    let context = probs.matmul(&v).unwrap();
                    let context = context.transpose(1, 2).unwrap().contiguous().unwrap();
                    let output = context.flatten_from(2).unwrap();

                    candle_device.synchronize().unwrap();
                },
                BatchSize::LargeInput,
            );
        },
    );
}

// Benchmark FFN (Feed-Forward Network) block
fn ffn_block(c: &mut Criterion) {
    use crate::Device;
    use crate::Tensor;

    let source = FileSource::HuggingFace {
        model_id: "CompendiumLabs/bge-large-en-v1.5-gguf".to_string(),
        revision: "main".to_string(),
        file: "bge-large-en-v1.5-q4_k_m.gguf".to_string(),
    };
    let bytes = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async move {
            let cache = Cache::default();
            let path = cache.get(&source, |_| {}).await.unwrap();
            tokio::fs::read(&path).await.unwrap()
        });

    for batch_size in [1, 32, 512] {
        for seq_len in [13, 128] {
            if batch_size * seq_len >= 512 * 128 {
                // Skip too large combinations for FFN
                continue;
            }
            let hidden_size = 1024;
            let intermediate_size = 4096;

            let random_data: Vec<Vec<Vec<f32>>> = (0..batch_size)
                .map(|_| {
                    (0..seq_len)
                        .map(|_| (0..hidden_size).map(|_| rand::random()).collect())
                        .collect()
                })
                .collect();

            // Fusor FFN benchmark
            {
                let mut reader = std::io::Cursor::new(&bytes);
                let mut var_builder = fusor_core::VarBuilder::from_gguf(&mut reader).unwrap();
                let device = block_on(Device::new()).unwrap();

                // Load FFN weights from the model
                let ffn_up = Linear::load(&device, &mut var_builder.pp("blk.0.ffn_up")).unwrap();
                let ffn_down =
                    Linear::load(&device, &mut var_builder.pp("blk.0.ffn_down")).unwrap();

                let mut group =
                    c.benchmark_group(format!("ffn_block-fusor-{batch_size}x{seq_len}"));

                let device = device.clone();
                let random_data = random_data.clone();
                group.bench_with_input(
                    BenchmarkId::new("ffn_block-fusor", format!("{batch_size}x{seq_len}")),
                    &(batch_size, seq_len),
                    move |b, &(batch_size, seq_len)| {
                        let device = device.clone();
                        let random_data = random_data.clone();

                        b.to_async(FuturesExecutor).iter_custom(async |iters| {
                            let tensor = Tensor::new(&device, &random_data);
                            tensor.materialize().await;
                            let mut sum = Duration::ZERO;
                            while sum.is_zero() {
                                for _ in 0..iters {
                                    let start = std::time::Instant::now();

                                    // Intermediate (up projection + GELU)
                                    let intermediate = ffn_up.forward(&tensor);
                                    let intermediate = intermediate.gelu();

                                    // Output (down projection)
                                    let output = ffn_down.forward(&intermediate);

                                    output.materialize().await;
                                    sum += start.elapsed();
                                }
                            }
                            sum
                        });
                    },
                );
            }

            // Candle FFN benchmark
            #[cfg(target_os = "macos")]
            {
                let candle_device = candle_core::Device::Metal(MetalDevice::new(0).unwrap());
                bench_candle_ffn(
                    &bytes,
                    batch_size,
                    seq_len,
                    hidden_size,
                    random_data.clone(),
                    candle_device,
                    "ffn_block-candle-metal",
                    c,
                );
            }

            {
                let candle_device = candle_core::Device::Cpu;
                bench_candle_ffn(
                    &bytes,
                    batch_size,
                    seq_len,
                    hidden_size,
                    random_data.clone(),
                    candle_device,
                    "ffn_block-candle-cpu",
                    c,
                );
            }
        }
    }
}

fn bench_candle_ffn(
    bytes: &[u8],
    batch_size: usize,
    seq_len: usize,
    hidden_size: usize,
    random_data: Vec<Vec<Vec<f32>>>,
    candle_device: candle_core::Device,
    name: &str,
    c: &mut Criterion,
) {
    use candle_transformers::quantized_nn::Linear;

    let var_builder = candle_transformers::quantized_var_builder::VarBuilder::from_gguf_buffer(
        bytes,
        &candle_device,
    )
    .unwrap();

    let up_weight = var_builder
        .pp("blk.0.ffn_up")
        .get_no_shape("weight")
        .unwrap();
    let up_bias = var_builder
        .pp("blk.0.ffn_up")
        .get_no_shape("bias")
        .unwrap()
        .dequantize(&candle_device)
        .unwrap();
    let down_weight = var_builder
        .pp("blk.0.ffn_down")
        .get_no_shape("weight")
        .unwrap();
    let down_bias = var_builder
        .pp("blk.0.ffn_down")
        .get_no_shape("bias")
        .unwrap()
        .dequantize(&candle_device)
        .unwrap();

    let up_linear = Linear::from_arc(up_weight, Some(up_bias)).unwrap();
    let down_linear = Linear::from_arc(down_weight, Some(down_bias)).unwrap();

    let mut group = c.benchmark_group(format!("{name}-{batch_size}x{seq_len}"));
    group.sample_size(20);

    group.bench_with_input(
        BenchmarkId::new(name, format!("{batch_size}x{seq_len}")),
        &(batch_size, seq_len),
        move |b, &(batch_size, seq_len)| {
            b.to_async(FuturesExecutor).iter_batched(
                || {
                    let candle_tensor = candle_core::Tensor::from_iter(
                        random_data
                            .iter()
                            .flat_map(|b| b.iter().flat_map(|s| s.iter().copied())),
                        &candle_device,
                    )
                    .unwrap()
                    .reshape(&[batch_size, seq_len, hidden_size])
                    .unwrap();
                    candle_device.synchronize().unwrap();
                    (
                        candle_tensor,
                        up_linear.clone(),
                        down_linear.clone(),
                        candle_device.clone(),
                    )
                },
                |(tensor, up_linear, down_linear, candle_device)| async move {
                    // Intermediate (up projection + GELU)
                    let intermediate = up_linear.forward(&tensor).unwrap();
                    let intermediate = intermediate.gelu().unwrap();

                    // Output (down projection)
                    let output = down_linear.forward(&intermediate).unwrap();

                    candle_device.synchronize().unwrap();
                },
                BatchSize::LargeInput,
            );
        },
    );
}

criterion_group!(benches, layer_norm, self_attention, ffn_block);
criterion_main!(benches);
