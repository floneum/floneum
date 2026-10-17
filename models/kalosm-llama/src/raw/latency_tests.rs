//! Small real-model reproduction of latency spikes at KV-capacity boundaries.
use super::*;
use std::time::Instant;

fn decode_model(device: &Device) -> Model {
    let config = Arc::new(LlamaConfig {
        rope_freq_weight: None,
        rope_theta: 10_000.,
        context_length: 1024,
        head_dimension: 128,
        n_head: 1,
        n_layer: 1,
        start_token_string: String::new(),
        stop_token: 0,
        stop_token_string: String::new(),
        chat_template: None,
        rope_scaling: None,
        sliding_window_type: None,
        sliding_window_size: None,
        mrope_sections: None,
        vision_start_token: None,
        image_pad_token: None,
    });
    let weight = |rows, cols, seed| {
        let values: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 17 + seed) % 41) as f32 * 0.0015 - 0.03)
            .collect();
        Weight::Dense(Tensor::from_slice(device, [rows, cols], &values).into_dyn())
    };
    let norm = || RmsNorm::new(Some(Tensor::from_slice(device, [128], &[1.; 128])), 1e-5);
    let layers = (0..config.n_layer)
        .map(|layer| {
            let seed = layer * 7;
            LlamaAttention {
                attention_variant: AttentionVariant::Separate(Box::new(SeparateAttention {
                    attention_wq: weight(128, 128, seed + 1),
                    attention_qkv: None,
                    attention_qk: None,
                    attention_q_norm: None,
                    attention_wk: weight(128, 128, seed + 2),
                    attention_k_norm: None,
                    attention_wv: weight(128, 128, seed + 3),
                    bias: None,
                    interleaved_rope: false,
                })),
                attention_wo: weight(128, 128, seed + 4),
                attention_norm: norm(),
                post_attention_norm: None,
                feed_forward_variant: FeedForwardVariant::Llama(Box::new(LlamaFeedForward::new(
                    weight(128, 128, seed + 5),
                    weight(128, 128, seed + 6),
                    weight(128, 128, seed + 7),
                ))),
                ffn_norm: norm(),
                post_ffn_norm: None,
                n_head: 1,
                n_kv_head: 1,
                head_dim: 128,
                hidden_size: 128,
                rope_cache: RopeImplementation::new(&config, config.rope_theta, device),
                sliding_window_size: None,
            }
        })
        .collect();
    Model {
        tok_embeddings: weight(16, 128, 23),
        tok_embedding_scale: None,
        layers,
        norm: norm(),
        output: weight(16, 128, 31),
        masks: Mutex::new(MaskCache::new()),
        step_inputs: Default::default(),
        embed_inputs: Default::default(),
        config,
        #[cfg(feature = "vision")]
        vision_encoder: None,
    }
}

#[test]
#[ignore = "requires a live GPU; bounded one-layer model, no downloaded weights"]
fn decode_latency_across_cache_growth() {
    let device = pollster::block_on(Device::gpu_inference()).unwrap();
    assert_eq!(
        device.execution_profile(),
        fusor_ir::execution::ExecutionProfile::LowLatency,
        "this regression requires an adapter supporting the inference profile"
    );
    let target = device.backend().gpu_target().unwrap();
    let model = decode_model(&device);
    let mut cache = LlamaCache::new(&model.config, device.execution_profile());
    // Exercise the production prefill/decode path, then cross 64, 128 and 256.
    let prompt: Vec<_> = (0..48).map(|i| (i % 16) as u32).collect();
    model
        .forward(&prompt, &[], &device, Some(&mut cache))
        .unwrap()
        .to_vec_f32();
    let mut stable = Vec::new();
    let mut boundaries = Vec::new();
    let mut boundary_compiles = 0;
    for step in 0..212 {
        let before = cache.blocks[0].storage_capacity().unwrap();
        let compiles = target.launcher().pipeline_compiles();
        let start = Instant::now();
        KvCache::reserve_all(&mut cache.blocks, 1).unwrap();
        let reserve_ms = start.elapsed().as_secs_f64() * 1000.;
        let reserve_compiles = target.launcher().pipeline_compiles() - compiles;
        let values = model
            .forward(&[(step % 16) as u32], &[], &device, Some(&mut cache))
            .unwrap()
            .to_vec_f32();
        let ms = start.elapsed().as_secs_f64() * 1000.;
        let after = cache.blocks[0].storage_capacity().unwrap();
        let compiled = target.launcher().pipeline_compiles() - compiles;
        assert!(values.iter().all(|v| v.is_finite()));
        eprintln!(
            "DECODE_LATENCY token={} capacity={before}->{after} ms={ms:.3} compiles={compiled} reserve_ms={reserve_ms:.3} reserve_compiles={reserve_compiles}",
            49 + step
        );
        if before != after {
            boundaries.push(ms);
            boundary_compiles += compiled;
        } else if step > 2 {
            stable.push(ms);
        }
    }
    stable.sort_by(f64::total_cmp);
    let median = stable[stable.len() / 2];
    assert_eq!(
        boundaries.len(),
        3,
        "exercise all three cache growth boundaries"
    );
    let worst = boundaries.into_iter().fold(0.0_f64, f64::max);
    eprintln!("DECODE_LATENCY median_ms={median:.3} worst_boundary_ms={worst:.3}");
    assert_eq!(
        boundary_compiles, 0,
        "cache growth must not compile new decode shaders"
    );
    assert!(
        worst < 100. && worst < median * 20.,
        "cache growth stalled decode: {worst:.3}ms vs {median:.3}ms typical"
    );
}
