mod config;
mod data;
mod model;

use data::{
    DatasetSplit, MidiTokenizer, SourceDataset, SourceFile, autoregressive_context,
    bootstrap_dataset, load_dataset_split, position_indexes, windows_to_token_inputs,
    windows_to_token_targets, write_tokens_to_midi_file,
};
use fusor::{Device, Tensor, ToVec2, ToVec3};
use fusor_gguf::{
    BlockQ4_0, BlockQ8_0, GgmlType, GgufMetadata, GgufTensorMetadata, GgufValue, GgufVersion,
};
use half::f16;
use model::{AdamW, NamedTensor, NanoChatModel};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::{
    env,
    fs::File,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};

use crate::config::{RuntimeConfig, SaveQuantization};

fn main() {
    pollster::block_on(async {
        let args = Args::parse();
        let runtime = RuntimeConfig::load();
        let tokenizer = MidiTokenizer::new();

        let device = if args.force_cpu {
            Device::cpu()
        } else {
            Device::gpu().await.unwrap()
        };

        let dataset_root = bootstrap_dataset(&runtime.dataset_cache_dir).await;
        let datasets = load_dataset_split(&dataset_root, &tokenizer);
        let DatasetSplit {
            train: train_dataset,
            validation: validation_dataset,
            test: test_dataset,
        } = datasets;
        assert!(
            train_dataset.num_docs() > 0,
            "NES-MDB training split is empty under {}",
            dataset_root.display()
        );

        let evaluation_dataset =
            preferred_eval_dataset(&validation_dataset, &test_dataset, &train_dataset);
        let mut rng = StdRng::seed_from_u64(runtime.seed);
        let train_position_values = position_indexes(runtime.batch_size, runtime.block_size);
        let train_position_inputs: Tensor<2, u32> = Tensor::new(&device, &train_position_values);
        let sample_position_values = position_indexes(1, runtime.block_size);
        let sample_position_inputs: Tensor<2, u32> = Tensor::new(&device, &sample_position_values);
        let mut model = NanoChatModel::new(&device, &mut rng, tokenizer.vocab_size(), &runtime);
        let causal_mask = causal_mask_tensor(
            model.graph(),
            &device,
            runtime.batch_size.max(1),
            runtime.block_size,
        );
        let sample_causal_mask = causal_mask_tensor(model.graph(), &device, 1, runtime.block_size);
        let mut optimizer = AdamW::new(&device, &model, &runtime);

        println!(
            "dataset_root={} train_files={} valid_files={} test_files={} train_tokens={} valid_tokens={} test_tokens={} train_windows={} valid_windows={} vocab={} params={} max_file_tokens={}",
            dataset_root.display(),
            train_dataset.num_docs(),
            validation_dataset.num_docs(),
            test_dataset.num_docs(),
            train_dataset.num_tokens(),
            validation_dataset.num_tokens(),
            test_dataset.num_tokens(),
            train_dataset.num_training_windows(runtime.block_size),
            validation_dataset.num_training_windows(runtime.block_size),
            tokenizer.vocab_size(),
            model.num_parameters(),
            train_dataset
                .max_tokens_per_example()
                .max(validation_dataset.max_tokens_per_example())
                .max(test_dataset.max_tokens_per_example()),
        );

        for step in 0..runtime.train_steps {
            let learning_rate = scheduled_learning_rate(step + 1, &runtime);
            optimizer.set_learning_rate(learning_rate);
            let batch = train_dataset.sample_batch(&mut rng, tokenizer.eot_token(), &runtime);
            let input_values = windows_to_token_inputs(&batch.windows);
            let target_values = windows_to_token_targets(&batch.windows);
            let token_inputs: Tensor<2, u32> = Tensor::new(&device, &input_values);
            let targets: Tensor<2, u32> = Tensor::new(&device, &target_values);
            let logits = model.forward(&token_inputs, &train_position_inputs, &causal_mask);
            let loss = masked_cross_entropy_autograd(
                model.graph(),
                &device,
                &logits,
                &targets,
                &batch.masks,
                batch.valid_tokens,
                tokenizer.vocab_size(),
            );

            let should_log = step % runtime.log_every == 0 || step + 1 == runtime.train_steps;
            let can_materialize_metrics = !device.is_gpu();
            let loss_value = if should_log && can_materialize_metrics {
                Some(loss.raw().to_scalar().await.unwrap())
            } else {
                None
            };
            let gradients = loss.backward().unwrap();
            model = optimizer.step(model, &gradients);

            if let Some(loss_value) = loss_value {
                let train_bits_per_token = nats_to_bits(loss_value);
                let validation_bits_per_token = evaluate_bits_per_token(
                    &model,
                    &device,
                    tokenizer.eot_token(),
                    evaluation_dataset,
                    &runtime,
                )
                .await;
                println!(
                    "step {:>4} | lr={learning_rate:.6} | loss={loss_value:.6} | train_bpt={train_bits_per_token:.4} | eval_bpt={validation_bits_per_token:.4}",
                    step + 1
                );
            } else if should_log {
                println!(
                    "step {:>4} | lr={learning_rate:.6} | metrics skipped on gpu",
                    step + 1
                );
            }

            if runtime.save_every_steps > 0 && (step + 1) % runtime.save_every_steps == 0 {
                let checkpoint_path = checkpoint_path(&runtime.gguf_path, step + 1);
                let checkpoint_path = resolve_output_path(&checkpoint_path);
                save_gguf(&model, &tokenizer, &runtime, &checkpoint_path).await;
                println!("saved checkpoint: {}", checkpoint_path.display());
            }
        }

        if runtime.save_final_model {
            let gguf_path = resolve_output_path(&runtime.gguf_path);
            save_gguf(&model, &tokenizer, &runtime, &gguf_path).await;
            println!("saved gguf: {}", gguf_path.display());
        } else {
            println!("skipped final gguf save");
        }

        let metrics = evaluate_continuation(
            &model,
            &device,
            &sample_position_inputs,
            &sample_causal_mask,
            &tokenizer,
            evaluation_dataset,
            &runtime,
        )
        .await;

        println!(
            "\ncontinuation token accuracy: {}/{} ({:.2}%)",
            metrics.correct_tokens,
            metrics.total_tokens,
            metrics.token_accuracy() * 100.0,
        );
        println!(
            "continuation exact match: {}/{} ({:.2}%)",
            metrics.exact_matches,
            metrics.exact_match_examples,
            metrics.exact_match_rate() * 100.0,
        );

        if let Some(file) = evaluation_dataset.files().first() {
            let prompt_tokens = continuation_prompt_tokens(&tokenizer, file, &runtime);
            let generated = generate_completion(
                &model,
                &device,
                &sample_position_inputs,
                &sample_causal_mask,
                &runtime,
                &prompt_tokens,
                tokenizer.eot_token(),
                &mut rng,
                SamplingMode::Sample,
            )
            .await;
            let expected = expected_continuation_tokens(
                file,
                prompt_tokens.len().saturating_sub(1),
                runtime.sample_tokens,
            );
            let mut sample_tokens = prompt_tokens[1..].to_vec();
            sample_tokens.extend_from_slice(&generated);
            let sample_output_path = resolve_output_path(&runtime.sample_output_path);
            write_tokens_to_midi_file(&tokenizer, &sample_tokens, &sample_output_path);
            println!("\n--- sample export ---");
            println!("source: {}", file.path());
            println!(
                "prompt: {}",
                tokenizer.describe_tokens(&prompt_tokens[1..], 48)
            );
            println!("expected: {}", tokenizer.describe_tokens(expected, 48));
            println!("generated: {}", tokenizer.describe_tokens(&generated, 48));
            println!("wrote sample MIDI: {}", sample_output_path.display());
        }
    });
}

struct Args {
    force_cpu: bool,
}

impl Args {
    fn parse() -> Self {
        let mut force_cpu = false;
        let mut args = env::args().skip(1);

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--cpu" => force_cpu = true,
                other => panic!("unknown argument: {other}"),
            }
        }

        Self { force_cpu }
    }
}

async fn save_gguf(
    model: &NanoChatModel,
    tokenizer: &MidiTokenizer,
    runtime: &RuntimeConfig,
    path: &Path,
) {
    let save_start = Instant::now();
    println!(
        "saving gguf to {} ({})...",
        path.display(),
        runtime.save_quantization.as_str()
    );
    println!("materializing model weights for save...");
    let materialize_start = Instant::now();
    let named_tensors = model.named_tensors().await;
    let materialize_elapsed = materialize_start.elapsed();
    let materialized_bytes = named_tensors
        .iter()
        .map(|tensor| tensor.values.len() * std::mem::size_of::<f32>())
        .sum::<usize>();
    println!(
        "materialized {} tensors in {:.2?} ({:.2} MiB of f32 weights)",
        named_tensors.len(),
        materialize_elapsed,
        materialized_bytes as f64 / (1024.0 * 1024.0),
    );

    println!("serializing tensors for gguf...");
    let serialize_start = Instant::now();
    let mut serialized_tensors = Vec::with_capacity(named_tensors.len());
    let mut serialized_bytes = 0usize;
    for tensor in named_tensors {
        let tensor_start = Instant::now();
        let input_bytes = tensor.values.len() * std::mem::size_of::<f32>();
        let name = tensor.name.clone();
        let shape = tensor.shape.clone();
        let serialized = serialize_tensor(tensor, runtime.save_quantization);
        serialized_bytes += serialized.data.len();
        println!(
            "serialized tensor {name} shape={shape:?} input={:.2} MiB output={:.2} MiB ty={} in {:.2?}",
            input_bytes as f64 / (1024.0 * 1024.0),
            serialized.data.len() as f64 / (1024.0 * 1024.0),
            serialized.ty,
            tensor_start.elapsed(),
        );
        serialized_tensors.push(serialized);
    }
    println!(
        "prepared {} tensors for gguf write in {:.2?} ({:.2} MiB serialized)",
        serialized_tensors.len(),
        serialize_start.elapsed(),
        serialized_bytes as f64 / (1024.0 * 1024.0),
    );

    let tensor_info_start = Instant::now();
    let mut offset = 0u64;
    let tensor_infos = serialized_tensors
        .iter()
        .map(|tensor| {
            let metadata = GgufTensorMetadata {
                ty: tensor.ty,
                shape: tensor.shape.clone().into_boxed_slice(),
                offset,
            };
            offset += tensor.data.len() as u64;
            (tensor.name.clone().into_boxed_str(), metadata)
        })
        .collect();
    println!(
        "built tensor metadata in {:.2?}",
        tensor_info_start.elapsed()
    );

    let metadata_start = Instant::now();
    let mut metadata = vec![
        (
            String::from("general.architecture").into_boxed_str(),
            GgufValue::String("nanochat".into()),
        ),
        (
            String::from("general.name").into_boxed_str(),
            GgufValue::String("fusor-nanochat-midi".into()),
        ),
        (
            String::from("general.alignment").into_boxed_str(),
            GgufValue::U32(1),
        ),
        (
            String::from("nanochat.block_size").into_boxed_str(),
            GgufValue::U32(model.block_size() as u32),
        ),
        (
            String::from("nanochat.embedding_length").into_boxed_str(),
            GgufValue::U32(model.n_embd() as u32),
        ),
        (
            String::from("nanochat.feed_forward_length").into_boxed_str(),
            GgufValue::U32(model.n_ff() as u32),
        ),
        (
            String::from("nanochat.conv_kernel_size").into_boxed_str(),
            GgufValue::U32(model.conv_kernel_size() as u32),
        ),
        (
            String::from("nanochat.head_count").into_boxed_str(),
            GgufValue::U32(model.n_head() as u32),
        ),
        (
            String::from("nanochat.block_count").into_boxed_str(),
            GgufValue::U32(model.n_layer() as u32),
        ),
        (
            String::from("nanochat.attention_period").into_boxed_str(),
            GgufValue::U32(model.attention_period() as u32),
        ),
        (
            String::from("nanochat.vocab_size").into_boxed_str(),
            GgufValue::U32(model.vocab_size() as u32),
        ),
        (
            String::from("nanochat.train_steps").into_boxed_str(),
            GgufValue::U32(runtime.train_steps.min(u32::MAX as usize) as u32),
        ),
        (
            String::from("nanochat.learning_rate").into_boxed_str(),
            GgufValue::F32(runtime.learning_rate),
        ),
        (
            String::from("nanochat.batch_size").into_boxed_str(),
            GgufValue::U32(runtime.batch_size as u32),
        ),
        (
            String::from("nanochat.training_objective").into_boxed_str(),
            GgufValue::String("autoregressive-midi".into()),
        ),
        (
            String::from("nanochat.eps").into_boxed_str(),
            GgufValue::F32(model.eps()),
        ),
        (
            String::from("nanochat.save_quantization").into_boxed_str(),
            GgufValue::String(runtime.save_quantization.as_str().into()),
        ),
    ];
    metadata.extend(
        tokenizer
            .gguf_metadata()
            .into_iter()
            .map(|(key, value)| (key.into_boxed_str(), value)),
    );
    println!(
        "built gguf metadata in {:.2?} ({} metadata entries)",
        metadata_start.elapsed(),
        metadata.len(),
    );

    let gguf_build_start = Instant::now();
    let gguf = GgufMetadata {
        version: GgufVersion::V3,
        metadata: metadata.into_iter().collect(),
        tensor_infos,
        tensor_data_offset: 0,
    };
    println!(
        "assembled gguf structure in {:.2?}",
        gguf_build_start.elapsed()
    );

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }

    let writer = File::create(path).unwrap();
    let mut writer = BufWriter::new(writer);
    println!("writing gguf bytes...");
    gguf.write(
        &mut writer,
        serialized_tensors
            .iter()
            .map(|tensor| (tensor.name.as_str(), tensor.data.as_slice())),
    )
    .unwrap();
    writer.flush().unwrap();
    println!("total save time: {:.2?}", save_start.elapsed());
}

struct SerializedTensor {
    name: String,
    shape: Vec<u32>,
    ty: GgmlType,
    data: Vec<u8>,
}

fn serialize_tensor(
    tensor: NamedTensor,
    preferred_quantization: SaveQuantization,
) -> SerializedTensor {
    let (ty, data) = match preferred_quantization {
        SaveQuantization::F32 => (
            GgmlType::F32,
            bytemuck::cast_slice::<f32, u8>(&tensor.values).to_vec(),
        ),
        SaveQuantization::F16 => (GgmlType::F16, f16_bytes(&tensor.values)),
        SaveQuantization::Q4_0 => serialize_q4_0_or_f16(&tensor.shape, &tensor.values),
        SaveQuantization::Q8_0 => serialize_q8_0_or_f16(&tensor.shape, &tensor.values),
    };

    SerializedTensor {
        name: tensor.name,
        shape: tensor.shape,
        ty,
        data,
    }
}

fn f16_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<f16>());
    for value in values {
        bytes.extend_from_slice(&f16::from_f32(*value).to_bits().to_le_bytes());
    }
    bytes
}

fn serialize_q8_0_or_f16(shape: &[u32], values: &[f32]) -> (GgmlType, Vec<u8>) {
    let Some(&inner_dim) = shape.last() else {
        return (GgmlType::F16, f16_bytes(values));
    };

    if inner_dim as usize % BlockQ8_0::BLOCK_SIZE != 0
        || !values.len().is_multiple_of(BlockQ8_0::BLOCK_SIZE)
    {
        return (GgmlType::F16, f16_bytes(values));
    }

    let mut bytes =
        Vec::with_capacity(values.len() / BlockQ8_0::BLOCK_SIZE * std::mem::size_of::<BlockQ8_0>());
    for chunk in values.chunks_exact(BlockQ8_0::BLOCK_SIZE) {
        let block = BlockQ8_0::quantize(chunk.try_into().unwrap());
        bytes.extend_from_slice(bytemuck::bytes_of(&block));
    }
    (GgmlType::Q8_0, bytes)
}

fn serialize_q4_0_or_f16(shape: &[u32], values: &[f32]) -> (GgmlType, Vec<u8>) {
    let Some(&inner_dim) = shape.last() else {
        return (GgmlType::F16, f16_bytes(values));
    };

    if inner_dim as usize % BlockQ4_0::BLOCK_SIZE != 0
        || !values.len().is_multiple_of(BlockQ4_0::BLOCK_SIZE)
    {
        return (GgmlType::F16, f16_bytes(values));
    }

    let mut bytes =
        Vec::with_capacity(values.len() / BlockQ4_0::BLOCK_SIZE * std::mem::size_of::<BlockQ4_0>());
    for chunk in values.chunks_exact(BlockQ4_0::BLOCK_SIZE) {
        let block = BlockQ4_0::quantize(chunk.try_into().unwrap());
        bytes.extend_from_slice(bytemuck::bytes_of(&block));
    }
    (GgmlType::Q4_0, bytes)
}

fn checkpoint_path(base_path: &Path, step: usize) -> PathBuf {
    let parent = base_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_default();
    let stem = base_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("nanochat");
    let extension = base_path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("gguf");
    parent.join(format!("{stem}-step-{step:05}.{extension}"))
}

fn resolve_output_path(path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(path)
    }
}

async fn generate_completion(
    model: &NanoChatModel,
    device: &Device,
    position_inputs: &Tensor<2, u32>,
    causal_mask: &fusor::autograd::Tensor<3>,
    runtime: &RuntimeConfig,
    prompt_tokens: &[u32],
    stop_token: u32,
    rng: &mut StdRng,
    mode: SamplingMode,
) -> Vec<u32> {
    let mut tokens = prompt_tokens.to_vec();

    for _ in 0..runtime.sample_tokens {
        let (context, last_index) = autoregressive_context(&tokens, stop_token, runtime.block_size);
        let token_inputs: Tensor<2, u32> = Tensor::new(device, &[context]);
        let logits = model.forward(&token_inputs, position_inputs, causal_mask);
        let logits = logits.raw().clone().as_slice().await.unwrap().to_vec3();
        let next = match mode {
            SamplingMode::Greedy => argmax_from_logits(&logits[0][last_index]),
            SamplingMode::Sample => sample_from_logits(
                rng,
                &logits[0][last_index],
                runtime.sample_temperature,
                runtime.sample_top_k,
            ),
        };
        tokens.push(next);
        if next == stop_token {
            break;
        }
    }

    tokens[prompt_tokens.len()..]
        .iter()
        .copied()
        .take_while(|&token| token != stop_token)
        .collect()
}

fn argmax_from_logits(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index as u32)
        .unwrap()
}

fn sample_from_logits(rng: &mut StdRng, logits: &[f32], temperature: f32, top_k: usize) -> u32 {
    if temperature <= 0.0 || top_k <= 1 {
        return argmax_from_logits(logits);
    }

    let mut candidates = logits.iter().copied().enumerate().collect::<Vec<_>>();
    candidates.sort_unstable_by(|(_, left), (_, right)| right.total_cmp(left));
    candidates.truncate(top_k.min(candidates.len()).max(1));

    let max_logit = candidates[0].1;
    let mut weighted = Vec::with_capacity(candidates.len());
    let mut total = 0.0f32;
    for (index, logit) in candidates {
        let weight = ((logit - max_logit) / temperature).exp();
        total += weight;
        weighted.push((index as u32, weight));
    }

    let mut threshold = rng.random::<f32>() * total.max(f32::MIN_POSITIVE);
    for (index, weight) in weighted {
        threshold -= weight;
        if threshold <= 0.0 {
            return index;
        }
    }

    argmax_from_logits(logits)
}

fn causal_mask_tensor(
    graph: &fusor::autograd::Graph,
    device: &Device,
    batch_size: usize,
    block_size: usize,
) -> fusor::autograd::Tensor<3> {
    let mut causal_mask_values = vec![vec![vec![0.0f32; block_size]; block_size]; batch_size];
    for batch in &mut causal_mask_values {
        for i in 0..block_size {
            for j in (i + 1)..block_size {
                batch[i][j] = f32::NEG_INFINITY;
            }
        }
    }
    fusor::autograd::Tensor::constant_from_raw(
        graph,
        fusor::Tensor::new(device, &causal_mask_values),
    )
}

fn masked_cross_entropy_autograd(
    graph: &fusor::autograd::Graph,
    device: &fusor::Device,
    logits: &fusor::autograd::Tensor<3>,
    targets: &fusor::Tensor<2, u32>,
    mask: &[Vec<f32>],
    valid_tokens: f32,
    vocab_size: usize,
) -> fusor::autograd::Tensor<0> {
    let batch_size = logits.shape()[0];
    let block_size = logits.shape()[1];
    let target_values = pollster::block_on(targets.clone().as_slice())
        .unwrap()
        .to_vec2();
    let mut flat_targets = Vec::with_capacity(batch_size * block_size);
    for batch in 0..batch_size {
        for position in 0..block_size {
            flat_targets.push(target_values[batch][position]);
        }
    }

    let target_indices =
        fusor::Tensor::from_slice(device, [batch_size * block_size], &flat_targets);
    let mask = fusor::autograd::Tensor::constant_from_raw(graph, fusor::Tensor::new(device, mask));

    let log_norm = logits.exp().sum_keepdim(2).log();
    let log_probs = logits.sub(&log_norm.broadcast_as([batch_size, block_size, vocab_size]));
    let flat_log_probs = log_probs.reshape([batch_size * block_size, vocab_size]);
    let selected = flat_log_probs
        .gather_last(&target_indices)
        .reshape([batch_size, block_size]);
    let token_nll = selected.neg().mul(&mask);
    token_nll.sum(1).sum().div_scalar(valid_tokens.max(1.0))
}

async fn evaluate_bits_per_token(
    model: &NanoChatModel,
    device: &Device,
    pad_token: u32,
    dataset: &SourceDataset,
    runtime: &RuntimeConfig,
) -> f32 {
    let batches = dataset.evaluation_batches(pad_token, runtime);
    if batches.is_empty() {
        return 0.0;
    }

    let mut total_nats = 0.0;
    let mut total_valid_tokens = 0.0;
    for batch in batches {
        let batch_size = batch.windows.len();
        let batch_causal_mask =
            causal_mask_tensor(model.graph(), device, batch_size, runtime.block_size);
        let position_values = position_indexes(batch_size, runtime.block_size);
        let position_inputs: Tensor<2, u32> = Tensor::new(device, &position_values);
        let input_values = windows_to_token_inputs(&batch.windows);
        let target_values = windows_to_token_targets(&batch.windows);
        let token_inputs: Tensor<2, u32> = Tensor::new(device, &input_values);
        let targets: Tensor<2, u32> = Tensor::new(device, &target_values);
        let logits = model.forward(&token_inputs, &position_inputs, &batch_causal_mask);
        let loss = masked_cross_entropy_autograd(
            model.graph(),
            device,
            &logits,
            &targets,
            &batch.masks,
            batch.valid_tokens,
            model.vocab_size(),
        );
        total_nats += loss.raw().to_scalar().await.unwrap() * batch.valid_tokens.max(1.0);
        total_valid_tokens += batch.valid_tokens;
    }

    nats_to_bits(total_nats / total_valid_tokens.max(1.0))
}

fn scheduled_learning_rate(step: usize, runtime: &RuntimeConfig) -> f32 {
    if runtime.train_steps <= 1 {
        return runtime.learning_rate;
    }

    let warmup_steps = runtime.warmup_steps.max(1).min(runtime.train_steps);
    if step <= warmup_steps {
        return runtime.learning_rate * (step as f32 / warmup_steps as f32);
    }

    let decay_steps = runtime.train_steps.saturating_sub(warmup_steps).max(1);
    let progress = (step.saturating_sub(warmup_steps) as f32 / decay_steps as f32).clamp(0.0, 1.0);
    let cosine = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
    runtime.min_learning_rate + cosine * (runtime.learning_rate - runtime.min_learning_rate)
}

fn nats_to_bits(nats: f32) -> f32 {
    nats / std::f32::consts::LN_2
}

struct PrefixMetrics {
    correct_tokens: usize,
    total_tokens: usize,
    exact_matches: usize,
    exact_match_examples: usize,
}

impl PrefixMetrics {
    fn token_accuracy(&self) -> f32 {
        self.correct_tokens as f32 / self.total_tokens.max(1) as f32
    }

    fn exact_match_rate(&self) -> f32 {
        self.exact_matches as f32 / self.exact_match_examples.max(1) as f32
    }
}

async fn evaluate_continuation(
    model: &NanoChatModel,
    device: &Device,
    position_inputs: &Tensor<2, u32>,
    causal_mask: &fusor::autograd::Tensor<3>,
    tokenizer: &MidiTokenizer,
    dataset: &SourceDataset,
    runtime: &RuntimeConfig,
) -> PrefixMetrics {
    let mut correct_tokens = 0;
    let mut total_tokens = 0;
    let mut exact_matches = 0;
    let exact_match_examples = dataset.files().len();

    println!(
        "evaluating continuation token accuracy across {} files...",
        dataset.files().len()
    );

    for file in dataset.files() {
        let prompt_tokens = continuation_prompt_tokens(tokenizer, file, runtime);
        let predicted = generate_completion(
            model,
            device,
            position_inputs,
            causal_mask,
            runtime,
            &prompt_tokens,
            tokenizer.eot_token(),
            &mut StdRng::seed_from_u64(0),
            SamplingMode::Greedy,
        )
        .await;

        let continuation = expected_continuation_tokens(
            file,
            prompt_tokens.len().saturating_sub(1),
            runtime.sample_tokens,
        );
        for (position, target) in continuation.iter().copied().enumerate() {
            total_tokens += 1;
            if predicted.get(position).copied() == Some(target) {
                correct_tokens += 1;
            }
        }
    }

    for file in dataset.files() {
        let prompt_tokens = continuation_prompt_tokens(tokenizer, file, runtime);
        let generated = generate_completion(
            model,
            device,
            position_inputs,
            causal_mask,
            runtime,
            &prompt_tokens,
            tokenizer.eot_token(),
            &mut StdRng::seed_from_u64(0),
            SamplingMode::Greedy,
        )
        .await;
        let expected = expected_continuation_tokens(
            file,
            prompt_tokens.len().saturating_sub(1),
            runtime.sample_tokens,
        );

        if generated == expected {
            exact_matches += 1;
        }
    }

    PrefixMetrics {
        correct_tokens,
        total_tokens,
        exact_matches,
        exact_match_examples,
    }
}

fn continuation_prompt_tokens(
    tokenizer: &MidiTokenizer,
    file: &SourceFile,
    runtime: &RuntimeConfig,
) -> Vec<u32> {
    let mut prompt = vec![tokenizer.bos_token()];
    prompt.extend_from_slice(file.completion_prompt_tokens(
        runtime.sample_prefix_tokens,
        runtime.block_size,
        runtime.sample_tokens,
    ));
    prompt
}

fn expected_continuation_tokens(
    file: &SourceFile,
    prompt_tokens: usize,
    sample_tokens: usize,
) -> &[u32] {
    let start = prompt_tokens.min(file.target_tokens().len());
    let end = (start + sample_tokens).min(file.target_tokens().len());
    &file.target_tokens()[start..end]
}

fn preferred_eval_dataset<'a>(
    validation: &'a SourceDataset,
    test: &'a SourceDataset,
    train: &'a SourceDataset,
) -> &'a SourceDataset {
    if validation.num_docs() > 0 {
        validation
    } else if test.num_docs() > 0 {
        test
    } else {
        train
    }
}

#[derive(Clone, Copy)]
enum SamplingMode {
    Greedy,
    Sample,
}
