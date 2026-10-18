#![recursion_limit = "256"]

use kalosm_llama::prelude::*;
use kalosm_model_types::ModelLoadingProgress;

async fn measure_stream<S>(mut stream: S, warmup: usize, measured: usize)
where
    S: futures_util::Stream<Item = String> + Unpin,
{
    // A text callback may combine tokens (for example while decoding UTF-8).
    // Report stream chunks explicitly instead of treating them as token IDs.
    let mut text = String::new();
    for _ in 0..warmup {
        let Some(chunk) = stream.next().await else {
            panic!("stream ended during warmup; no throughput sample was measured");
        };
        text.push_str(&chunk);
    }

    let warmup_bytes = text.len();
    let start = std::time::Instant::now();
    let mut last = start;
    let mut gaps_ms = Vec::with_capacity(measured);
    while gaps_ms.len() < measured {
        let Some(chunk) = stream.next().await else {
            break;
        };
        let now = std::time::Instant::now();
        gaps_ms.push(now.duration_since(last).as_secs_f64() * 1_000.);
        last = now;
        text.push_str(&chunk);
    }
    let elapsed = last.duration_since(start);
    let chunks = gaps_ms.len();
    assert!(
        chunks > 0,
        "stream ended before a throughput sample was measured"
    );
    let per_chunk_ms = elapsed.as_secs_f64() * 1_000. / chunks as f64;
    let mut sorted = gaps_ms.clone();
    sorted.sort_by(f64::total_cmp);
    let p50_ms = sorted[(chunks - 1) / 2];
    let p95_ms = sorted[(chunks * 95).div_ceil(100) - 1];
    let max_ms = sorted[chunks - 1];
    println!(
        "llama_forward_profile chunks={chunks} requested={measured} warmup_bytes={warmup_bytes} elapsed={elapsed:?} per_chunk_ms={per_chunk_ms:.3} p50_ms={p50_ms:.3} p95_ms={p95_ms:.3} max_ms={max_ms:.3}"
    );
    println!("llama_forward_gaps_ms {gaps_ms:?}");
    println!("llama_forward_output {text:?}");
    assert_eq!(
        chunks, measured,
        "stream ended early; incomplete benchmark sample"
    );
    if let Ok(max_ms) = std::env::var("KALOSM_PROFILE_LLAMA_MAX_MS") {
        let max_ms = max_ms.parse::<f64>().expect("MAX_MS must be a number");
        assert!(
            per_chunk_ms <= max_ms,
            "decode regression: {per_chunk_ms:.3} ms/chunk exceeds {max_ms:.3} ms/chunk"
        );
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn source() -> LlamaSource {
    if let Ok(path) = std::env::var("KALOSM_PROFILE_LLAMA_LOCAL_PATH") {
        return LlamaSource::new(FileSource::local(path.into()));
    }

    if let (Ok(model_id), Ok(file)) = (
        std::env::var("KALOSM_PROFILE_LLAMA_HF_REPO"),
        std::env::var("KALOSM_PROFILE_LLAMA_HF_FILE"),
    ) {
        let revision =
            std::env::var("KALOSM_PROFILE_LLAMA_HF_REVISION").unwrap_or_else(|_| "main".into());
        return LlamaSource::new(FileSource::huggingface(model_id, revision, file));
    }

    match std::env::var("KALOSM_PROFILE_LLAMA_SOURCE").as_deref() {
        Ok("default-chat") => LlamaSource::llama_3_1_8b_chat(),
        Ok("llama-8b") => LlamaSource::llama_8b(),
        Ok("llama-8b-chat") => LlamaSource::llama_8b_chat(),
        Ok("llama-3.1-8b-chat") => LlamaSource::llama_3_1_8b_chat(),
        Ok("tiny-llama") => LlamaSource::tiny_llama_1_1b_chat(),
        Ok("qwen2.5-0.5b") => LlamaSource::qwen_2_5_0_5b_instruct(),
        Ok("qwen2.5-1.5b") => LlamaSource::qwen_2_5_1_5b_instruct(),
        Ok("qwen3-0.6b") => LlamaSource::qwen_3_0_6b_instruct(),
        Ok("gemma3-270m") => LlamaSource::gemma_3_270m_chat(),
        Ok("gemma3-1b") => LlamaSource::gemma_3_1b_chat(),
        Ok("gemma3-4b") => LlamaSource::gemma_3_4b_chat(),
        _ => LlamaSource::new(FileSource::huggingface(
            "unsloth/SmolLM2-135M-Instruct-GGUF",
            "main",
            "SmolLM2-135M-Instruct-Q4_K_M.gguf",
        )),
    }
}

fn main() {
    let _ = tracing_subscriber::fmt::try_init();

    pollster::block_on(async {
        let warmup = env_usize("KALOSM_PROFILE_LLAMA_WARMUP", 4);
        let measured = env_usize("KALOSM_PROFILE_LLAMA_TOKENS", 16);
        let seed = env_usize("KALOSM_PROFILE_LLAMA_SEED", 0) as u64;
        assert!((1..=512).contains(&warmup) && (1..=512).contains(&measured));
        let prompt = std::env::var("KALOSM_PROFILE_LLAMA_PROMPT")
            .unwrap_or_else(|_| "Write one compact Rust performance tip:".into());

        let model_source = source();
        let source_description = format!("{model_source:?}");
        let model = Llama::builder()
            .with_source(model_source)
            .build_with_loading_handler(|_: ModelLoadingProgress| {})
            .await
            .unwrap();

        let prerun = env_usize("KALOSM_PROFILE_LLAMA_PRERUN", 0);
        assert!(prerun <= 512);
        if prerun > 0 {
            let prerun_prompt = std::env::var("KALOSM_PROFILE_LLAMA_PRERUN_PROMPT")
                .unwrap_or_else(|_| "Warm up the model with a short answer.".into());
            let prerun_sampler = GenerationParameters::default()
                .with_max_length(prerun as u32)
                .with_seed(seed);
            let prerun_stream = model
                .complete(&prerun_prompt)
                .with_sampler(prerun_sampler)
                .take(prerun);
            measure_stream(prerun_stream, 0, prerun).await;
        }

        let prompt_tokens = model
            .tokenizer()
            .encode(prompt.as_str(), true)
            .unwrap()
            .len();
        let sampler = if std::env::var_os("KALOSM_PROFILE_LLAMA_UNBOUNDED").is_some() {
            GenerationParameters::default()
        } else {
            GenerationParameters::default()
                .with_max_length((prompt_tokens + warmup + measured) as u32)
        };
        let sampler = match std::env::var("KALOSM_PROFILE_LLAMA_TOP_K")
            .ok()
            .and_then(|value| value.parse().ok())
        {
            Some(top_k) => sampler.with_top_k(top_k),
            None => sampler,
        };
        let repeats = env_usize("KALOSM_PROFILE_LLAMA_REPEATS", 1);
        assert!((1..=10).contains(&repeats));
        println!(
            "llama_forward_config warmup={warmup} measured={measured} seed={seed} repeats={repeats} chat={} prompt={prompt:?} source={source_description} sampler={sampler:?}",
            std::env::var_os("KALOSM_PROFILE_LLAMA_CHAT").is_some()
        );
        for _ in 0..repeats {
            if std::env::var_os("KALOSM_PROFILE_LLAMA_CHAT").is_some() {
                let mut chat = model.chat();
                let stream = chat
                    .add_message(prompt.clone())
                    .with_sampler(sampler.clone().with_seed(seed))
                    .take(warmup + measured);
                measure_stream(stream, warmup, measured).await;
            } else {
                let stream = model
                    .complete(&prompt)
                    .with_sampler(sampler.clone().with_seed(seed))
                    .take(warmup + measured);
                measure_stream(stream, warmup, measured).await;
            }
        }
    });
}
